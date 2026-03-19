#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
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


def run_variant(name, strat_kwargs, exit_cfg, leverage, max_pos, risk_per_trade):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()

    strat = BTCPerpPullbackStrategy1H(**strat_kwargs)
    sig = strat.generate_signals(df)

    lev = leverage * overlay["leverage_mult"]
    mp = max_pos * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        mp = 0.0

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
        max_pos=mp,
        cooldown_bars=2,
        stop_atr=exit_cfg["stop_atr"],
        take_R=exit_cfg["take_R"],
        trail_start_R=exit_cfg["trail_start_R"],
        trail_atr=exit_cfg["trail_atr"],
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=risk_per_trade,
        enable_risk_position_sizing=True,
        allow_reentry=True,
        partial_take_R=exit_cfg["partial_take_R"],
        partial_take_frac=exit_cfg["partial_take_frac"],
        break_even_after_partial=exit_cfg["break_even_after_partial"],
        break_even_R=exit_cfg["break_even_R"],
    )
    out = bt.run(sig)
    st = out.attrs.get("stats", {})
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    return {
        "variant": name,
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(dd),
        "trade_count": int(st.get("closed_trade_count", 0)),
        "fees": float(st.get("total_fees", 0.0)),
        "gross_closed_pnl": float(st.get("gross_closed_pnl", 0.0)),
        "net_closed_pnl": float(st.get("net_closed_pnl", 0.0)),
        "profit_factor": st.get("profit_factor", float("nan")),
        "expectancy_per_trade": float(st.get("expectancy_per_trade", 0.0)),
        "effective_leverage": lev,
        "risk_per_trade": risk_per_trade,
    }


def main():
    # 不恢复 short；只更激进地恢复样本 + 放大利润 + 提高风险使用
    entry_base = {
        "adx_threshold_4h": 26,
        "trend_strength_threshold_4h": 0.0050,
        "breakout_confirm_atr": 0.10,
        "breakout_body_atr": 0.18,
        "pullback_bars": 4,
        "pullback_max_depth_atr": 0.45,
        "first_pullback_only": False,
        "max_pullbacks_long": 4,
        "max_pullbacks_short": 1,
        "min_breakout_age_long": 1,
        "rejection_wick_ratio_long": 0.55,
        "rejection_wick_ratio_short": 0.80,
        "allow_short": False,
        "allow_same_bar_entry": False,
        "breakout_valid_bars": 14,
        "atr_pct_low": 0.0030,
        "atr_pct_high": 0.018,
    }

    runner_exit = {
        "stop_atr": 1.5,
        "take_R": 4.0,
        "trail_start_R": 1.8,
        "trail_atr": 3.2,
        "partial_take_R": 2.5,
        "partial_take_frac": 0.25,
        "break_even_after_partial": False,
        "break_even_R": 1.5,
    }

    aggressive_runner_exit = {
        "stop_atr": 1.6,
        "take_R": 5.0,
        "trail_start_R": 2.2,
        "trail_atr": 3.8,
        "partial_take_R": 3.0,
        "partial_take_frac": 0.20,
        "break_even_after_partial": False,
        "break_even_R": 1.8,
    }

    variants = [
        ("P20_A_ENTRY_PLUS", entry_base, runner_exit, 3.0, 1.2, 0.015),
        ("P20_B_RISK_PLUS", entry_base, runner_exit, 4.0, 1.5, 0.020),
        ("P20_C_MAX_PUSH", entry_base, aggressive_runner_exit, 5.0, 1.8, 0.025),
    ]

    rows = [run_variant(*v) for v in variants]
    rep = pd.DataFrame(rows).sort_values("return", ascending=False)
    print("\n=== PROFIT PUSH TOWARD 20% ===")
    print(rep.to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "gross_closed_pnl": lambda x: f"{x:.2f}",
        "net_closed_pnl": lambda x: f"{x:.2f}",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy_per_trade": lambda x: f"{x:.2f}",
        "effective_leverage": lambda x: f"{x:.2f}",
        "risk_per_trade": lambda x: f"{x:.2%}",
    }))


if __name__ == "__main__":
    main()
