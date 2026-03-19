#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from strategy_profiles import get_strategy_profile, BACKTEST_COMMON
from exit_profiles import list_exit_profiles, get_exit_profile
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


def run(profile="v6_1_default", exit_profile="exit_baseline"):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()
    strat = BTCPerpPullbackStrategy1H(**get_strategy_profile(profile))
    sig = strat.generate_signals(df)

    cfg = {**BACKTEST_COMMON, **get_exit_profile(exit_profile)}
    leverage = cfg["leverage"] * overlay["leverage_mult"]
    max_pos = cfg["max_pos"] * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        max_pos = 0.0

    portfolio = PerpPortfolio(INITIAL_CASH, leverage=leverage, taker_fee_rate=TAKER_FEE_RATE, maker_fee_rate=MAKER_FEE_RATE, maint_margin_rate=0.005)
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=portfolio,
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=cfg["cooldown_bars"],
        stop_atr=cfg["stop_atr"],
        take_R=cfg["take_R"],
        trail_start_R=cfg["trail_start_R"],
        trail_atr=cfg["trail_atr"],
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=cfg["risk_per_trade"],
        enable_risk_position_sizing=True,
        allow_reentry=True,
        partial_take_R=cfg["partial_take_R"],
        partial_take_frac=cfg["partial_take_frac"],
        break_even_after_partial=cfg["break_even_after_partial"],
        break_even_R=cfg["break_even_R"],
    )
    out = bt.run(sig)
    st = out.attrs.get("stats", {})
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    return {
        "exit_profile": exit_profile,
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(dd),
        "trade_count": int(st.get("closed_trade_count", 0)),
        "fees": float(st.get("total_fees", 0.0)),
        "win_rate": float(st.get("win_rate", 0.0)),
        "pnl_ratio": st.get("pnl_ratio", float("nan")),
        "profit_factor": st.get("profit_factor", float("nan")),
        "expectancy_per_trade": float(st.get("expectancy_per_trade", 0.0)),
        "gross_closed_pnl": float(st.get("gross_closed_pnl", 0.0)),
        "net_closed_pnl": float(st.get("net_closed_pnl", 0.0)),
        "mfe_capture_ratio": st.get("mfe_capture_ratio", float("nan")),
        "give_back_ratio": st.get("give_back_ratio", float("nan")),
        "avg_R_realized": st.get("avg_R_realized", float("nan")),
        "exit_reason_split": st.get("exit_reason_split", {}),
        "partial_take_effectiveness": st.get("partial_take_effectiveness", float("nan")),
    }


def main():
    rows = [run("v6_1_default", ep) for ep in list_exit_profiles()]
    rep = pd.DataFrame(rows)
    print("\n=== EXIT PROFILE COMPARE (v6_1_default) ===")
    print(rep.drop(columns=["exit_reason_split"]).to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "win_rate": lambda x: f"{x:.2%}",
        "pnl_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy_per_trade": lambda x: f"{x:.2f}",
        "gross_closed_pnl": lambda x: f"{x:.2f}",
        "net_closed_pnl": lambda x: f"{x:.2f}",
        "mfe_capture_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "give_back_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "avg_R_realized": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "partial_take_effectiveness": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
    }))
    print("\nExit reason split:")
    for r in rows:
        print(r["exit_profile"], r["exit_reason_split"])


if __name__ == "__main__":
    main()
