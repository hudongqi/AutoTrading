#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from strategy_profiles import list_profiles, get_strategy_profile, BACKTEST_COMMON
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


def run_segment(df_seg, profile):
    overlay = load_overlay()
    strat = BTCPerpPullbackStrategy1H(**get_strategy_profile(profile))
    sig = strat.generate_signals(df_seg)
    if sig.empty:
        return None, pd.DataFrame()
    leverage = BACKTEST_COMMON["leverage"] * overlay["leverage_mult"]
    max_pos = BACKTEST_COMMON["max_pos"] * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        max_pos = 0.0
    portfolio = PerpPortfolio(INITIAL_CASH, leverage=leverage, taker_fee_rate=TAKER_FEE_RATE, maker_fee_rate=MAKER_FEE_RATE, maint_margin_rate=0.005)
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=portfolio,
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=BACKTEST_COMMON["cooldown_bars"],
        stop_atr=BACKTEST_COMMON["stop_atr"],
        take_R=BACKTEST_COMMON["take_R"],
        trail_start_R=BACKTEST_COMMON["trail_start_R"],
        trail_atr=BACKTEST_COMMON["trail_atr"],
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=BACKTEST_COMMON["risk_per_trade"],
        enable_risk_position_sizing=True,
        allow_reentry=True,
    )
    out = bt.run(sig)
    stats = out.attrs.get("stats", {})
    trades = pd.DataFrame(out.attrs.get("closed_trades", []))
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    return {
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_dd": float(dd),
        "profit_factor": stats.get("profit_factor", float("nan")),
        "trades": int(stats.get("closed_trade_count", 0)),
    }, trades


def summarize(rep):
    nonzero = rep[rep["trades"] > 0].copy()
    return {
        "avg_trade_count_per_window": float(rep["trades"].mean()),
        "non_zero_window_ratio": float((rep["trades"] > 0).mean()),
        "median_return_non_zero": float(nonzero["return"].median()) if not nonzero.empty else 0.0,
        "median_profit_factor_non_zero": float(nonzero["profit_factor"].dropna().median()) if not nonzero["profit_factor"].dropna().empty else np.nan,
    }


def remove_best_trade_net(df):
    if df.empty or "realized_net" not in df.columns or len(df) <= 1:
        return None
    d = df.copy()
    d["realized_net"] = d["realized_net"].astype(float)
    return float(d.drop(d["realized_net"].idxmax())["realized_net"].sum())


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    idx = df.index
    start = idx.min(); end = idx.max()

    summary_rows = []
    for profile in list_profiles():
        rows = []
        trades_all = []
        cur = start
        while cur < end:
            seg_end = cur + pd.Timedelta(days=90)
            seg = df[(df.index >= cur) & (df.index < seg_end)]
            if len(seg) < 24 * 40:
                break
            r, t = run_segment(seg, profile)
            if r is not None:
                rows.append(r)
                if not t.empty:
                    trades_all.append(t)
            cur += pd.Timedelta(days=30)
        rep = pd.DataFrame(rows)
        trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
        s = summarize(rep)
        summary_rows.append({
            "profile": profile,
            **s,
            "remove_best_trade_net": remove_best_trade_net(trades),
        })

    out = pd.DataFrame(summary_rows)
    print("\n=== WALKFORWARD PROFILE SUMMARY ===")
    print(out.to_string(index=False, formatters={
        "avg_trade_count_per_window": lambda x: f"{x:.2f}",
        "non_zero_window_ratio": lambda x: f"{x:.2%}",
        "median_return_non_zero": lambda x: f"{x:.2%}",
        "median_profit_factor_non_zero": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "remove_best_trade_net": lambda x: f"{x:.2f}" if x is not None and pd.notna(x) else "N/A",
    }))


if __name__ == "__main__":
    main()
