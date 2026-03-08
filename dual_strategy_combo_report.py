import io
import contextlib
from dataclasses import dataclass
import numpy as np
import pandas as pd

from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester
from config import (
    SYMBOL,
    START,
    END,
    INITIAL_CASH,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    SLIPPAGE_BPS,
)


@dataclass
class RunOut:
    result: pd.DataFrame
    stats: dict
    closed_pnls: list


def run_strategy(df_raw: pd.DataFrame, cash: float, kind: str, regime_filter=False) -> RunOut:
    if kind == "conservative":
        strat = BTCPerpTrendStrategy1H(
            fast=5,
            slow=15,
            atr_pct_threshold=0.0045,
            use_regime_filter=False,
        )
        bt_kwargs = dict(
            leverage=2.0,
            max_pos=0.8,
            cooldown_bars=3,
            stop_atr=1.5,
            take_R=3.5,
            trail_start_R=1.5,
            trail_atr=2.0,
            entry_is_maker=False,
        )
    elif kind == "aggressive":
        strat = BTCPerpTrendStrategy1H(
            fast=5,
            slow=15,
            atr_pct_threshold=0.0038,
            use_regime_filter=regime_filter,
            adx_threshold_4h=32,
            trend_strength_threshold_4h=0.010,
            slow_slope_lookback_4h=2,
        )
        bt_kwargs = dict(
            leverage=3.6,
            max_pos=0.8,
            cooldown_bars=3,
            stop_atr=1.3,
            take_R=2.7,
            trail_start_R=0.8,
            trail_atr=3.0,
            entry_is_maker=True,
        )
    else:
        raise ValueError(kind)

    with contextlib.redirect_stdout(io.StringIO()):
        sig = strat.generate_signals(df_raw)

    portfolio = PerpPortfolio(
        initial_cash=cash,
        leverage=bt_kwargs["leverage"],
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=portfolio,
        strategy=strat,
        max_pos=bt_kwargs["max_pos"],
        cooldown_bars=bt_kwargs["cooldown_bars"],
        stop_atr=bt_kwargs["stop_atr"],
        take_R=bt_kwargs["take_R"],
        trail_start_R=bt_kwargs["trail_start_R"],
        trail_atr=bt_kwargs["trail_atr"],
        use_trailing=True,
        check_liq=True,
        entry_is_maker=bt_kwargs["entry_is_maker"],
        funding_rate_per_8h=0.0,
    )
    result = bt.run(sig)
    stats = result.attrs.get("stats", {})

    return RunOut(result=result, stats=stats, closed_pnls=list(bt.closed_trade_pnls))


def combine_metrics(con_out: RunOut, agg_out: RunOut):
    c = con_out.result[["equity"]].rename(columns={"equity": "eq_con"})
    a = agg_out.result[["equity"]].rename(columns={"equity": "eq_agg"})
    df = c.join(a, how="outer").sort_index().ffill()

    if df["eq_con"].isna().all():
        df["eq_con"] = 0.0
    else:
        df["eq_con"] = df["eq_con"].fillna(df["eq_con"].dropna().iloc[0])

    if df["eq_agg"].isna().all():
        df["eq_agg"] = 0.0
    else:
        df["eq_agg"] = df["eq_agg"].fillna(df["eq_agg"].dropna().iloc[0])

    df["equity"] = df["eq_con"] + df["eq_agg"]

    total_return = float(df["equity"].iloc[-1] / INITIAL_CASH - 1)
    max_dd = float((df["equity"] / df["equity"].cummax() - 1).min())

    all_pnls = np.array(con_out.closed_pnls + agg_out.closed_pnls, dtype=float)
    wins = all_pnls[all_pnls > 0]
    losses = all_pnls[all_pnls < 0]
    win_rate = float((len(wins) / len(all_pnls)) if len(all_pnls) else 0.0)
    pnl_ratio = float((wins.mean() / abs(losses.mean())) if (len(wins) and len(losses)) else np.nan)
    profit_factor = float((wins.sum() / abs(losses.sum())) if len(losses) else np.nan)

    trades = int(con_out.stats.get("trade_count", 0) + agg_out.stats.get("trade_count", 0))

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "trades": trades,
        "win_rate": win_rate,
        "pnl_ratio": pnl_ratio,
        "profit_factor": profit_factor,
    }


def build_regime_log(df_raw: pd.DataFrame):
    # Aggressive candidate signals (no regime)
    strat_no = BTCPerpTrendStrategy1H(
        fast=5, slow=15, atr_pct_threshold=0.0038, use_regime_filter=False
    )
    # Aggressive with regime
    strat_yes = BTCPerpTrendStrategy1H(
        fast=5,
        slow=15,
        atr_pct_threshold=0.0038,
        use_regime_filter=True,
        adx_threshold_4h=32,
        trend_strength_threshold_4h=0.010,
        slow_slope_lookback_4h=2,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        s0 = strat_no.generate_signals(df_raw)
    with contextlib.redirect_stdout(io.StringIO()):
        s1 = strat_yes.generate_signals(df_raw)

    idx = s0.index.intersection(s1.index)
    s0 = s0.loc[idx]
    s1 = s1.loc[idx]

    enabled = (s1["signal"] != 0)
    candidate_trade = (s0["trade_signal"] != 0)
    enabled_trade = (s1["trade_signal"] != 0)
    avoided_trade = candidate_trade & (~enabled_trade)

    log = pd.DataFrame({
        "aggressive_enabled": enabled,
        "candidate_trade": candidate_trade,
        "enabled_trade": enabled_trade,
        "avoided_trade": avoided_trade,
    }, index=idx)

    # 连续启用时段
    seg = log["aggressive_enabled"].ne(log["aggressive_enabled"].shift()).cumsum()
    periods = (
        log.assign(seg=seg)
        .groupby(["seg", "aggressive_enabled"], as_index=False)
        .agg(start=("aggressive_enabled", lambda x: x.index.min()), end=("aggressive_enabled", lambda x: x.index.max()))
    )
    periods = periods[periods["aggressive_enabled"]].drop(columns=["seg"])

    summary = {
        "enabled_bars": int(log["aggressive_enabled"].sum()),
        "total_bars": int(len(log)),
        "enabled_ratio": float(log["aggressive_enabled"].mean()),
        "candidate_trades": int(log["candidate_trade"].sum()),
        "enabled_trades": int(log["enabled_trade"].sum()),
        "avoided_trades": int(log["avoided_trade"].sum()),
    }

    return log, periods, summary


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    allocations = [
        (1.0, 0.0),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
    ]

    rows = []
    for w_con, w_agg in allocations:
        con = run_strategy(df, INITIAL_CASH * w_con, kind="conservative", regime_filter=False)
        agg = run_strategy(df, INITIAL_CASH * w_agg, kind="aggressive", regime_filter=True)
        m = combine_metrics(con, agg)
        rows.append({
            "conservative_weight": w_con,
            "aggressive_weight": w_agg,
            **m,
        })

    report = pd.DataFrame(rows)
    report.to_csv("dual_strategy_combo_report.csv", index=False)

    log, periods, summary = build_regime_log(df)
    log.to_csv("aggressive_regime_log.csv")
    periods.to_csv("aggressive_regime_periods.csv", index=False)

    print("=== COMBO REPORT ===")
    print(report.to_string(index=False))

    print("\n=== REGIME SWITCH SUMMARY ===")
    print(summary)
    print("\nSaved:")
    print("- dual_strategy_combo_report.csv")
    print("- aggressive_regime_log.csv")
    print("- aggressive_regime_periods.csv")


if __name__ == "__main__":
    main()
