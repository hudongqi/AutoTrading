import io
import contextlib
from dataclasses import dataclass
import itertools
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


# 固定 aggressive 主参数（按你的要求不改）
AGGR_PARAMS = dict(
    fast=5,
    slow=15,
    atr_pct_threshold=0.0038,
    entry_is_maker=True,
    funding_rate_per_8h=0.0,
    leverage=3.6,
    max_pos=0.8,
    cooldown_bars=3,
    stop_atr=1.3,
    take_R=2.7,
    trail_start_R=0.8,
    trail_atr=3.0,
)


@dataclass
class Window:
    name: str
    start: str
    end: str


def run_one(df, use_regime_filter, adx_threshold_4h, trend_strength_threshold_4h, slow_slope_lookback_4h):
    strat = BTCPerpTrendStrategy1H(
        fast=AGGR_PARAMS["fast"],
        slow=AGGR_PARAMS["slow"],
        atr_pct_threshold=AGGR_PARAMS["atr_pct_threshold"],
        use_regime_filter=use_regime_filter,
        adx_threshold_4h=adx_threshold_4h,
        trend_strength_threshold_4h=trend_strength_threshold_4h,
        slow_slope_lookback_4h=slow_slope_lookback_4h,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        sig = strat.generate_signals(df)

    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=AGGR_PARAMS["leverage"],
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )

    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=portfolio,
        strategy=strat,
        max_pos=AGGR_PARAMS["max_pos"],
        cooldown_bars=AGGR_PARAMS["cooldown_bars"],
        stop_atr=AGGR_PARAMS["stop_atr"],
        take_R=AGGR_PARAMS["take_R"],
        trail_start_R=AGGR_PARAMS["trail_start_R"],
        trail_atr=AGGR_PARAMS["trail_atr"],
        use_trailing=True,
        check_liq=True,
        entry_is_maker=AGGR_PARAMS["entry_is_maker"],
        funding_rate_per_8h=AGGR_PARAMS["funding_rate_per_8h"],
    )

    result = bt.run(sig)
    eq = result["equity"]
    ret = float(eq.iloc[-1] / INITIAL_CASH - 1)
    max_dd = float((eq / eq.cummax() - 1).min())
    stats = result.attrs.get("stats", {})

    pnls = np.array(bt.closed_trade_pnls, dtype=float)
    gross_profit = float(pnls[pnls > 0].sum()) if pnls.size else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if pnls.size else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    return {
        "return": ret,
        "max_drawdown": max_dd,
        "trades": int(stats.get("trade_count", 0)),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "pnl_ratio": float(stats.get("pnl_ratio", np.nan)),
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
    }


def main():
    ds = CCXTDataSource()

    # 至少覆盖：趋势 / 震荡 / 最近样本外
    windows = [
        Window("TREND_2025Q4", "2025-09-01", "2025-12-15"),
        Window("RANGE_2025H1", "2025-02-01", "2025-06-30"),
        Window("OOS_RECENT", "2026-01-01", END),
        Window("FULL_2025_TO_TODAY", START, END),
    ]

    adx_grid = [24, 28, 32, 36]
    trend_strength_grid = [0.004, 0.006, 0.008, 0.010]
    slope_lookback_grid = [2, 3, 4, 6]

    rows = []

    # baseline without regime filter (for reference)
    for w in windows:
        df = ds.load_ohlcv(SYMBOL, w.start, w.end)
        m = run_one(
            df,
            use_regime_filter=False,
            adx_threshold_4h=0,
            trend_strength_threshold_4h=0.0,
            slow_slope_lookback_4h=3,
        )
        rows.append({
            "window": w.name,
            "use_regime_filter": False,
            "adx_threshold_4h": 0,
            "trend_strength_threshold_4h": 0.0,
            "slow_slope_lookback_4h": 3,
            **m,
        })

    # grid with regime filter on
    combos = list(itertools.product(adx_grid, trend_strength_grid, slope_lookback_grid))
    for w in windows:
        df = ds.load_ohlcv(SYMBOL, w.start, w.end)
        for adx_th, ts_th, slope_lb in combos:
            m = run_one(
                df,
                use_regime_filter=True,
                adx_threshold_4h=adx_th,
                trend_strength_threshold_4h=ts_th,
                slow_slope_lookback_4h=slope_lb,
            )
            rows.append({
                "window": w.name,
                "use_regime_filter": True,
                "adx_threshold_4h": adx_th,
                "trend_strength_threshold_4h": ts_th,
                "slow_slope_lookback_4h": slope_lb,
                **m,
            })

    out = pd.DataFrame(rows)

    # 每区间最优（以 return 最大，且 dd 不差于 -40%）
    per_window_best = []
    for w in windows:
        x = out[(out["window"] == w.name) & (out["use_regime_filter"])]
        x2 = x[x["max_drawdown"] >= -0.40]
        if x2.empty:
            x2 = x
        best = x2.sort_values(["return", "profit_factor", "win_rate"], ascending=False).head(1)
        per_window_best.append(best)
    per_window_best_df = pd.concat(per_window_best, ignore_index=True)

    # 全区间综合稳健性：
    # 1) 先按参数聚合四个窗口
    # 2) 最大化 worst_return，其次最小化 worst_dd，其次最大化 mean_return
    g = out[out["use_regime_filter"]].groupby(
        ["adx_threshold_4h", "trend_strength_threshold_4h", "slow_slope_lookback_4h"],
        as_index=False,
    ).agg(
        worst_return=("return", "min"),
        mean_return=("return", "mean"),
        worst_dd=("max_drawdown", "min"),
        mean_dd=("max_drawdown", "mean"),
        mean_pf=("profit_factor", "mean"),
        mean_wr=("win_rate", "mean"),
        mean_trades=("trades", "mean"),
    )

    robust = g.sort_values(
        ["worst_return", "worst_dd", "mean_return", "mean_pf"],
        ascending=[False, False, False, False],
    ).head(1)

    out.to_csv("regime_grid_all_results.csv", index=False)
    per_window_best_df.to_csv("regime_grid_per_window_best.csv", index=False)
    g.to_csv("regime_grid_robust_ranking.csv", index=False)

    print("=== BASE RANGE ===")
    print(f"SYMBOL={SYMBOL}, START={START}, END={END}")

    print("\n=== PER-WINDOW BEST ===")
    print(per_window_best_df[[
        "window", "adx_threshold_4h", "trend_strength_threshold_4h", "slow_slope_lookback_4h",
        "return", "max_drawdown", "trades", "win_rate", "pnl_ratio", "profit_factor"
    ]].to_string(index=False))

    print("\n=== MOST ROBUST ACROSS WINDOWS ===")
    print(robust.to_string(index=False))

    print("\nSaved:")
    print("- regime_grid_all_results.csv")
    print("- regime_grid_per_window_best.csv")
    print("- regime_grid_robust_ranking.csv")


if __name__ == "__main__":
    main()
