#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def load_overlay(event_file="event_signals.json"):
    defaults = {
        "block_new_entries": False,
        "reduce_risk": False,
        "risk_mode": "NORMAL",
        "market_bias": "NEUTRAL",
        "leverage_mult": 1.0,
        "max_pos_mult": 1.0,
    }
    p = Path(event_file)
    if not p.exists():
        return defaults
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    macro = data.get("macro", {})
    geo = data.get("geopolitics", {})
    risk_mode = str(data.get("risk_mode", "NORMAL")).upper()
    reduce_risk = bool(macro.get("reduce_risk", False) or geo.get("reduce_risk", False) or risk_mode == "REDUCE_RISK")
    return {
        "block_new_entries": bool(macro.get("block", False) or geo.get("block_new_entries", False)),
        "reduce_risk": reduce_risk,
        "risk_mode": risk_mode,
        "market_bias": str(data.get("market_bias", "NEUTRAL")).upper(),
        "leverage_mult": 0.6 if reduce_risk else 1.0,
        "max_pos_mult": 0.5 if reduce_risk else 1.0,
    }


def run_one(df, overlay, leverage, risk_per_trade):
    strat = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=30,
        trend_strength_threshold_4h=0.006,
        breakout_confirm_atr=0.15,
        breakout_body_atr=0.25,
        pullback_bars=3,
        pullback_max_depth_atr=0.75,
        first_pullback_only=False,
        max_pullbacks_long=2,
        max_pullbacks_short=1,
        rejection_wick_ratio_long=0.55,
        rejection_wick_ratio_short=0.80,
        allow_short=False,
        allow_same_bar_entry=False,
        breakout_valid_bars=10,
        atr_pct_low=0.0035,
        atr_pct_high=0.015,
    )
    sig = strat.generate_signals(df)

    lev = leverage * overlay["leverage_mult"]
    max_pos = 0.8 * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        max_pos = 0.0

    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=lev,
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=portfolio,
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=3,
        stop_atr=1.4,
        take_R=2.6,
        trail_start_R=1.0,
        trail_atr=2.2,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=risk_per_trade,
        enable_risk_position_sizing=True,
        allow_reentry=True,
    )
    out = bt.run(sig)
    stats = out.attrs.get("stats", {})
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    return {
        "base_leverage": leverage,
        "effective_leverage": lev,
        "risk_per_trade": risk_per_trade,
        "final_equity": float(eq.iloc[-1]),
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(dd),
        "fees": float(stats.get("total_fees", 0.0)),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "pnl_ratio": stats.get("pnl_ratio", float("nan")),
        "profit_factor": stats.get("profit_factor", float("nan")),
        "expectancy_per_trade": float(stats.get("expectancy_per_trade", 0.0)),
        "fees_per_trade": float(stats.get("fees_per_trade", 0.0)),
        "trade_count": int(stats.get("closed_trade_count", 0)),
        "time_in_market": float(stats.get("time_in_market", 0.0)),
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()

    tests = [
        (2.0, 0.0075),
        (2.5, 0.0090),
        (3.0, 0.0100),
        (3.5, 0.0100),
    ]

    rows = [run_one(df, overlay, lev, rpt) for lev, rpt in tests]
    rep = pd.DataFrame(rows).sort_values("return", ascending=False)
    print("\n==== V4 LEVERAGE / RISK SWEEP ====")
    print(rep.to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "win_rate": lambda x: f"{x:.2%}",
        "time_in_market": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "fees_per_trade": lambda x: f"{x:.2f}",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "pnl_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy_per_trade": lambda x: f"{x:.2f}",
        "final_equity": lambda x: f"{x:.2f}",
        "base_leverage": lambda x: f"{x:.1f}",
        "effective_leverage": lambda x: f"{x:.2f}",
        "risk_per_trade": lambda x: f"{x:.2%}",
    }))


if __name__ == "__main__":
    main()
