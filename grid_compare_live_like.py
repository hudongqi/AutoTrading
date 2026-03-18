#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def load_overlay(use_research_overlay: bool = True):
    if not use_research_overlay:
        return {
            "block_new_entries": False,
            "reduce_risk": False,
            "risk_mode": "NORMAL",
            "market_bias": "NEUTRAL",
            "leverage_mult": 1.0,
            "max_pos_mult": 1.0,
        }

    p = Path("event_signals.json")
    if not p.exists():
        return {
            "block_new_entries": False,
            "reduce_risk": False,
            "risk_mode": "NORMAL",
            "market_bias": "NEUTRAL",
            "leverage_mult": 1.0,
            "max_pos_mult": 1.0,
        }

    data = json.loads(p.read_text(encoding="utf-8"))
    macro = data.get("macro", {})
    geo = data.get("geopolitics", {})
    risk_mode = str(data.get("risk_mode", "NORMAL")).upper()
    reduce = bool(macro.get("reduce_risk", False) or geo.get("reduce_risk", False) or risk_mode == "REDUCE_RISK")
    return {
        "block_new_entries": bool(macro.get("block", False) or geo.get("block_new_entries", False)),
        "reduce_risk": reduce,
        "risk_mode": risk_mode,
        "market_bias": str(data.get("market_bias", "NEUTRAL")).upper(),
        "leverage_mult": 0.6 if reduce else 1.0,
        "max_pos_mult": 0.5 if reduce else 1.0,
    }


def run_case(name, df, p, overlay):
    strat = BTCPerpTrendStrategy1H(
        fast=5,
        slow=15,
        atr_pct_threshold=p["atr_pct_threshold"],
        use_regime_filter=p["use_regime_filter"],
        adx_threshold_4h=p["adx_threshold_4h"],
        trend_strength_threshold_4h=p["trend_strength_threshold_4h"],
        slow_slope_lookback_4h=3,
    )
    df_sig = strat.generate_signals(df)

    leverage = p["leverage"] * overlay["leverage_mult"]
    max_pos = p["max_pos"] * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        max_pos = 0.0

    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=leverage,
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    broker = SimBroker(slippage_bps=SLIPPAGE_BPS)

    bt = Backtester(
        broker=broker,
        portfolio=portfolio,
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=p["cooldown_bars"],
        stop_atr=p["stop_atr"],
        take_R=p["take_R"],
        trail_start_R=p["trail_start_R"],
        trail_atr=p["trail_atr"],
        use_trailing=True,
        check_liq=True,
        entry_is_maker=False,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
    )

    out = bt.run(df_sig)
    stats = out.attrs.get("stats", {})
    ret = out["equity"].iloc[-1] / INITIAL_CASH - 1
    dd = (out["equity"] / out["equity"].cummax() - 1).min()
    fees = stats.get("total_fees", 0.0)
    win = stats.get("win_rate", 0.0)
    pnlr = stats.get("pnl_ratio", float("nan"))

    return {
        "case": name,
        "return": ret,
        "max_dd": dd,
        "fees": fees,
        "win_rate": win,
        "pnl_ratio": pnlr,
        "final_equity": float(out["equity"].iloc[-1]),
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    overlay_on = load_overlay(use_research_overlay=True)
    overlay_off = load_overlay(use_research_overlay=False)

    configs = [
        (
            "A_CONSERVATIVE",
            {
                "atr_pct_threshold": 0.0052,
                "use_regime_filter": True,
                "adx_threshold_4h": 36,
                "trend_strength_threshold_4h": 0.008,
                "cooldown_bars": 8,
                "stop_atr": 1.4,
                "take_R": 2.4,
                "trail_start_R": 0.9,
                "trail_atr": 2.2,
                "leverage": 2.0,
                "max_pos": 0.8,
            },
        ),
        (
            "B_BALANCED",
            {
                "atr_pct_threshold": 0.0046,
                "use_regime_filter": True,
                "adx_threshold_4h": 34,
                "trend_strength_threshold_4h": 0.007,
                "cooldown_bars": 6,
                "stop_atr": 1.4,
                "take_R": 2.6,
                "trail_start_R": 0.9,
                "trail_atr": 2.5,
                "leverage": 2.0,
                "max_pos": 0.8,
            },
        ),
        (
            "C_CURRENT_TUNED",
            {
                "atr_pct_threshold": 0.0045,
                "use_regime_filter": False,
                "adx_threshold_4h": 32,
                "trend_strength_threshold_4h": 0.006,
                "cooldown_bars": 6,
                "stop_atr": 1.5,
                "take_R": 3.0,
                "trail_start_R": 1.0,
                "trail_atr": 2.0,
                "leverage": 2.0,
                "max_pos": 0.8,
            },
        ),
    ]

    rows = []
    for name, cfg in configs:
        rows.append(run_case(name + "_OVERLAY_ON", df, cfg, overlay_on))
        rows.append(run_case(name + "_OVERLAY_OFF", df, cfg, overlay_off))

    rep = pd.DataFrame(rows).sort_values("return", ascending=False)
    pd.set_option("display.max_columns", 20)
    print("\n=== GRID COMPARE (LIVE-LIKE) ===")
    print(rep.to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_dd": lambda x: f"{x:.2%}",
        "win_rate": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "pnl_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "final_equity": lambda x: f"{x:.2f}",
    }))


if __name__ == "__main__":
    main()
