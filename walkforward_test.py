#!/usr/bin/env python3
import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def run_segment(df_seg):
    strat = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=30,
        trend_strength_threshold_4h=0.006,
        breakout_confirm_atr=0.15,
        pullback_bars=3,
        atr_pct_low=0.0035,
        atr_pct_high=0.015,
    )
    sig = strat.generate_signals(df_seg)
    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=2.0,
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=portfolio,
        strategy=strat,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.4,
        take_R=2.6,
        trail_start_R=1.0,
        trail_atr=2.2,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=0.0075,
        enable_risk_position_sizing=True,
        allow_reentry=True,
    )
    out = bt.run(sig)
    stats = out.attrs.get("stats", {})
    ret = out["equity"].iloc[-1] / INITIAL_CASH - 1
    dd = (out["equity"] / out["equity"].cummax() - 1).min()
    return {
        "return": ret,
        "max_dd": dd,
        "win_rate": stats.get("win_rate", 0.0),
        "profit_factor": stats.get("profit_factor", float("nan")),
        "expectancy": stats.get("expectancy_per_trade", 0.0),
        "trades": stats.get("trade_count", 0),
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    # 简易 rolling windows: 90d 窗口，每30d 滚动
    idx = df.index
    start = idx.min()
    end = idx.max()

    rows = []
    cur = start
    while cur < end:
        seg_end = cur + pd.Timedelta(days=90)
        seg = df[(df.index >= cur) & (df.index < seg_end)]
        if len(seg) < 24 * 40:
            break
        res = run_segment(seg)
        res["from"] = str(cur)
        res["to"] = str(seg_end)
        rows.append(res)
        cur = cur + pd.Timedelta(days=30)

    rep = pd.DataFrame(rows)
    if rep.empty:
        print("No enough data for walk-forward")
        return

    print("\n=== WALK-FORWARD REPORT ===")
    print(rep.to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_dd": lambda x: f"{x:.2%}",
        "win_rate": lambda x: f"{x:.2%}",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy": lambda x: f"{x:.4f}",
    }))


if __name__ == "__main__":
    main()
