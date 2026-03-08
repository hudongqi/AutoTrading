import io
import contextlib
import numpy as np
import pandas as pd

from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester
from config import SYMBOL, START, END, INITIAL_CASH, MAKER_FEE_RATE, TAKER_FEE_RATE, SLIPPAGE_BPS


AGGR = dict(
    atr_pct_threshold=0.0038,
    use_regime_filter=True,
    adx_threshold_4h=32,
    trend_strength_threshold_4h=0.010,
    slow_slope_lookback_4h=2,
    leverage=3.6,
    max_pos=0.8,
    cooldown_bars=3,
    stop_atr=1.3,
    take_R=2.7,
    trail_start_R=0.8,
    trail_atr=3.0,
    entry_is_maker=True,
)


def run_backtest(df_raw, strat, sig, cash, cfg):
    pf = PerpPortfolio(
        initial_cash=cash,
        leverage=cfg["leverage"],
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=pf,
        strategy=strat,
        max_pos=cfg["max_pos"],
        cooldown_bars=cfg["cooldown_bars"],
        stop_atr=cfg["stop_atr"],
        take_R=cfg["take_R"],
        trail_start_R=cfg["trail_start_R"],
        trail_atr=cfg["trail_atr"],
        use_trailing=True,
        check_liq=True,
        entry_is_maker=cfg["entry_is_maker"],
        funding_rate_per_8h=0.0,
    )
    res = bt.run(sig)
    return res, res.attrs.get("stats", {}), list(bt.closed_trade_pnls)


def build_conservative_signal(df_raw, variant):
    # baseline conservative (current)
    atr_th = 0.0045
    strat = BTCPerpTrendStrategy1H(
        fast=5,
        slow=15,
        atr_pct_threshold=atr_th,
        use_regime_filter=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sig = strat.generate_signals(df_raw)

    # NOTE: 当前策略本身已含基础4h方向 + 7d突破，这里做“增强版”
    if variant == "A_4H_SLOPE_FILTER":
        long_ok = sig["ma_slow_slope_4h"] > 0
        short_ok = sig["ma_slow_slope_4h"] < 0
        sig.loc[(sig["signal"] == 1) & (~long_ok), "signal"] = 0
        sig.loc[(sig["signal"] == -1) & (~short_ok), "signal"] = 0

    elif variant == "B_ATR_0.0045":
        pass

    elif variant == "B_ATR_0.005":
        strat = BTCPerpTrendStrategy1H(
            fast=5,
            slow=15,
            atr_pct_threshold=0.005,
            use_regime_filter=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            sig = strat.generate_signals(df_raw)

    elif variant == "C_BREAKOUT_BUFFER_0.1ATR":
        # 增加突破缓冲：close > res + 0.1*atr / close < sup - 0.1*atr
        long_keep = (sig["signal"] == 1) & (sig["close"] > (sig["resistance_7d"] + 0.1 * sig["atr"]))
        short_keep = (sig["signal"] == -1) & (sig["close"] < (sig["support_7d"] - 0.1 * sig["atr"]))
        sig.loc[(sig["signal"] == 1) & (~long_keep), "signal"] = 0
        sig.loc[(sig["signal"] == -1) & (~short_keep), "signal"] = 0

    elif variant == "D_LEV_1.5":
        pass

    elif variant == "D_MAXPOS_0.6":
        pass

    else:
        raise ValueError(variant)

    sig["trade_signal"] = sig["signal"].diff().fillna(0)
    return strat, sig


def conservative_cfg(variant):
    cfg = dict(
        leverage=2.0,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.5,
        take_R=3.5,
        trail_start_R=1.5,
        trail_atr=2.0,
        entry_is_maker=False,
    )
    if variant == "D_LEV_1.5":
        cfg["leverage"] = 1.5
    if variant == "D_MAXPOS_0.6":
        cfg["max_pos"] = 0.6
    return cfg


def combine(con_res, con_stats, con_pnls, agg_res, agg_stats, agg_pnls):
    c = con_res[["equity"]].rename(columns={"equity": "eq_con"})
    a = agg_res[["equity"]].rename(columns={"equity": "eq_agg"})
    df = c.join(a, how="outer").sort_index().ffill()
    df["eq_con"] = df["eq_con"].bfill()
    df["eq_agg"] = df["eq_agg"].bfill()
    df["equity"] = df["eq_con"] + df["eq_agg"]

    total_return = float(df["equity"].iloc[-1] / INITIAL_CASH - 1)
    max_dd = float((df["equity"] / df["equity"].cummax() - 1).min())

    pnls = np.array(con_pnls + agg_pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    pnl_ratio = float((wins.mean() / abs(losses.mean())) if (len(wins) and len(losses)) else np.nan)
    profit_factor = float((wins.sum() / abs(losses.sum())) if len(losses) else np.nan)

    trades = int(con_stats.get("trade_count", 0) + agg_stats.get("trade_count", 0))

    return dict(
        total_return=total_return,
        max_drawdown=max_dd,
        trades=trades,
        win_rate=win_rate,
        pnl_ratio=pnl_ratio,
        profit_factor=profit_factor,
    )


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    # Aggressive fixed (as requested)
    aggr_strat = BTCPerpTrendStrategy1H(
        fast=5,
        slow=15,
        atr_pct_threshold=AGGR["atr_pct_threshold"],
        use_regime_filter=AGGR["use_regime_filter"],
        adx_threshold_4h=AGGR["adx_threshold_4h"],
        trend_strength_threshold_4h=AGGR["trend_strength_threshold_4h"],
        slow_slope_lookback_4h=AGGR["slow_slope_lookback_4h"],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        aggr_sig = aggr_strat.generate_signals(df)

    variants = [
        "A_4H_SLOPE_FILTER",
        "B_ATR_0.0045",
        "B_ATR_0.005",
        "C_BREAKOUT_BUFFER_0.1ATR",
        "D_LEV_1.5",
        "D_MAXPOS_0.6",
    ]

    allocs = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]

    rows = []

    for v in variants:
        c_strat, c_sig = build_conservative_signal(df, v)
        c_cfg = conservative_cfg(v)

        for w_c, w_a in allocs:
            con_res, con_stats, con_pnls = run_backtest(df, c_strat, c_sig, INITIAL_CASH * w_c, c_cfg)
            agg_res, agg_stats, agg_pnls = run_backtest(df, aggr_strat, aggr_sig, INITIAL_CASH * w_a, AGGR)
            m = combine(con_res, con_stats, con_pnls, agg_res, agg_stats, agg_pnls)
            rows.append({
                "variant": v,
                "conservative_weight": w_c,
                "aggressive_weight": w_a,
                **m,
            })

    out = pd.DataFrame(rows)
    out.to_csv("conservative_tuning_combo_report.csv", index=False)

    print(out.to_string(index=False))
    print("\nSaved: conservative_tuning_combo_report.csv")


if __name__ == "__main__":
    main()
