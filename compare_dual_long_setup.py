#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from strategy_profiles import get_strategy_profile
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def load_overlay(event_file="event_signals.json"):
    defaults = {"block_new_entries": False, "reduce_risk": False, "risk_mode": "NORMAL", "market_bias": "NEUTRAL", "leverage_mult": 1.0, "max_pos_mult": 1.0}
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


def run(profile_name):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()
    strat = BTCPerpPullbackStrategy1H(**get_strategy_profile(profile_name))
    sig = strat.generate_signals(df)

    leverage = 3.0 * overlay["leverage_mult"]
    max_pos = 1.2 * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        max_pos = 0.0

    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=PerpPortfolio(INITIAL_CASH, leverage=leverage, taker_fee_rate=TAKER_FEE_RATE, maker_fee_rate=MAKER_FEE_RATE, maint_margin_rate=0.005),
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=2,
        stop_atr=1.4,
        take_R=2.6,
        trail_start_R=1.0,
        trail_atr=2.2,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=0.015,
        enable_risk_position_sizing=True,
        allow_reentry=True,
    )
    out = bt.run(sig)
    st = out.attrs.get("stats", {})
    trades = pd.DataFrame(out.attrs.get("closed_trades", []))
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    remove_best = None
    setup_split = {}
    if not trades.empty and "realized_net" in trades.columns:
        trades = trades.copy()
        trades["realized_net"] = trades["realized_net"].astype(float)
        if len(trades) > 1:
            remove_best = float(trades.drop(trades["realized_net"].idxmax())["realized_net"].sum())
        if "entry_reason" in trades.columns:
            setup_split = trades.groupby("entry_reason")["realized_net"].agg(["count", "mean", "sum"]).to_dict("index")
    return {
        "profile": profile_name,
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(dd),
        "trade_count": int(st.get("closed_trade_count", 0)),
        "fees": float(st.get("total_fees", 0.0)),
        "gross_closed_pnl": float(st.get("gross_closed_pnl", 0.0)),
        "net_closed_pnl": float(st.get("net_closed_pnl", 0.0)),
        "profit_factor": st.get("profit_factor", float("nan")),
        "expectancy_per_trade": float(st.get("expectancy_per_trade", 0.0)),
        "remove_best_trade_net": remove_best,
        "setup_split": setup_split,
    }


def main():
    rows = [run("v6_2_sample_up"), run("v7_dual_long")]
    rep = pd.DataFrame(rows)
    print("\n=== DUAL LONG SETUP COMPARE ===")
    print(rep.drop(columns=["setup_split"]).to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "gross_closed_pnl": lambda x: f"{x:.2f}",
        "net_closed_pnl": lambda x: f"{x:.2f}",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy_per_trade": lambda x: f"{x:.2f}",
        "remove_best_trade_net": lambda x: f"{x:.2f}" if x is not None and pd.notna(x) else "N/A",
    }))
    print("\nSetup split:")
    for r in rows:
        print(r["profile"], r["setup_split"])


if __name__ == "__main__":
    main()
