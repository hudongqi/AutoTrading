#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from strategy_profiles import get_strategy_profile, BACKTEST_COMMON
from exit_profiles import get_exit_profile
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester

STRATEGY_PROFILE = "v6_1_default"
EXIT_PROFILE = "exit_baseline"


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


def run_backtest(df):
    overlay = load_overlay()
    strat = BTCPerpPullbackStrategy1H(**get_strategy_profile(STRATEGY_PROFILE))
    sig = strat.generate_signals(df)
    if sig.empty:
        return None, pd.DataFrame()

    cfg = {**BACKTEST_COMMON, **get_exit_profile(EXIT_PROFILE)}
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
    stats = out.attrs.get("stats", {})
    trades = pd.DataFrame(out.attrs.get("closed_trades", []))
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    row = {
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_dd": float(dd),
        "trade_count": int(stats.get("closed_trade_count", 0)),
        "fees": float(stats.get("total_fees", 0.0)),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "pnl_ratio": stats.get("pnl_ratio", float("nan")),
        "profit_factor": stats.get("profit_factor", float("nan")),
        "expectancy": float(stats.get("expectancy_per_trade", 0.0)),
        "gross_closed_pnl": float(stats.get("gross_closed_pnl", 0.0)),
        "net_closed_pnl": float(stats.get("net_closed_pnl", 0.0)),
        "fees_per_trade": float(stats.get("fees_per_trade", 0.0)),
    }
    return row, trades


def remove_best_trade_net(trades):
    if trades.empty or "realized_net" not in trades.columns or len(trades) <= 1:
        return None
    t = trades.copy()
    t["realized_net"] = t["realized_net"].astype(float)
    return float(t.drop(t["realized_net"].idxmax())["realized_net"].sum())


def walkforward(df, window_days=90, step_days=30):
    rows = []
    trades_all = []
    cur = df.index.min()
    end = df.index.max()
    while cur < end:
        seg_end = cur + pd.Timedelta(days=window_days)
        seg = df[(df.index >= cur) & (df.index < seg_end)]
        if len(seg) < 24 * 40:
            break
        r, t = run_backtest(seg)
        if r is not None:
            r["from"] = str(cur)
            r["to"] = str(seg_end)
            rows.append(r)
            if not t.empty:
                trades_all.append(t)
        cur += pd.Timedelta(days=step_days)
    rep = pd.DataFrame(rows)
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    return rep, trades


def summarize_wf(rep):
    if rep.empty:
        return {}
    nonzero = rep[rep["trade_count"] > 0].copy()
    return {
        "window_count": len(rep),
        "positive_windows": int((rep["return"] > 0).sum()),
        "pf_gt_1_windows": int((rep["profit_factor"].fillna(0) > 1).sum()),
        "avg_trade_count_per_window": float(rep["trade_count"].mean()),
        "non_zero_window_ratio": float((rep["trade_count"] > 0).mean()),
        "median_return_non_zero": float(nonzero["return"].median()) if not nonzero.empty else 0.0,
        "median_profit_factor_non_zero": float(nonzero["profit_factor"].dropna().median()) if not nonzero["profit_factor"].dropna().empty else np.nan,
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    overall, overall_trades = run_backtest(df)
    print("\n=== CURRENT BASELINE OVERALL ===")
    print(overall)
    print("remove_best_trade_net:", remove_best_trade_net(overall_trades))

    for wd, sd in [(60, 20), (90, 30), (120, 30)]:
        rep, tr = walkforward(df, window_days=wd, step_days=sd)
        print(f"\n=== WALKFORWARD {wd}d/{sd}d ===")
        print(summarize_wf(rep))
        if not rep.empty:
            print(rep[["from", "to", "return", "trade_count", "profit_factor"]].to_string(index=False, formatters={
                "return": lambda x: f"{x:.2%}",
                "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            }))
        print("remove_best_trade_net:", remove_best_trade_net(tr))


if __name__ == "__main__":
    main()
