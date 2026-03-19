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
from strategy_profiles import get_strategy_profile, BACKTEST_COMMON
from exit_profiles import get_exit_profile


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


def run_variant(name, strat_kwargs, exit_profile, leverage, max_pos, risk_per_trade):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()

    strat = BTCPerpPullbackStrategy1H(**strat_kwargs)
    sig = strat.generate_signals(df)

    ep = get_exit_profile(exit_profile)
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
        cooldown_bars=BACKTEST_COMMON["cooldown_bars"],
        stop_atr=ep["stop_atr"],
        take_R=ep["take_R"],
        trail_start_R=ep["trail_start_R"],
        trail_atr=ep["trail_atr"],
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=risk_per_trade,
        enable_risk_position_sizing=True,
        allow_reentry=True,
        partial_take_R=ep["partial_take_R"],
        partial_take_frac=ep["partial_take_frac"],
        break_even_after_partial=ep["break_even_after_partial"],
        break_even_R=ep["break_even_R"],
    )
    out = bt.run(sig)
    st = out.attrs.get("stats", {})
    trades = pd.DataFrame(out.attrs.get("closed_trades", []))
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()

    remove_best = None
    avg_net_per_trade = 0.0
    if not trades.empty and "realized_net" in trades.columns:
        trades = trades.copy()
        trades["realized_net"] = trades["realized_net"].astype(float)
        avg_net_per_trade = float(trades["realized_net"].mean())
        if len(trades) > 1:
            remove_best = float(trades.drop(trades["realized_net"].idxmax())["realized_net"].sum())

    fees = float(st.get("total_fees", 0.0))
    gross = float(st.get("gross_closed_pnl", 0.0))
    gross_fee_ratio = gross / fees if fees > 0 else np.nan

    return {
        "variant": name,
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(dd),
        "trade_count": int(st.get("closed_trade_count", 0)),
        "fees": fees,
        "gross_closed_pnl": gross,
        "net_closed_pnl": float(st.get("net_closed_pnl", 0.0)),
        "avg_net_pnl_per_trade": avg_net_per_trade,
        "remove_best_trade_net": remove_best,
        "gross_to_fee": gross_fee_ratio,
        "profit_factor": st.get("profit_factor", float("nan")),
        "expectancy_per_trade": float(st.get("expectancy_per_trade", 0.0)),
        "effective_leverage": lev,
        "risk_per_trade": risk_per_trade,
    }


def main():
    base = get_strategy_profile("v6_1_default")

    variants = [
        (
            "V6.2_SAMPLE_UP",
            {**base, "min_breakout_age_long": 1},
            "exit_baseline",
            2.0,
            0.8,
            0.0075,
        ),
        (
            "V6.3_PROFIT_UP",
            {**base, "min_breakout_age_long": 1},
            "exit_loose_runner",
            2.0,
            0.8,
            0.0075,
        ),
        (
            "V6.4_SCALE_UP_A",
            {**base, "min_breakout_age_long": 1},
            "exit_loose_runner",
            2.5,
            1.0,
            0.0125,
        ),
        (
            "V6.4_SCALE_UP_B",
            {**base, "min_breakout_age_long": 1},
            "exit_loose_runner",
            3.0,
            1.2,
            0.0150,
        ),
    ]

    rows = [run_variant(*v) for v in variants]
    rep = pd.DataFrame(rows).sort_values("net_closed_pnl", ascending=False)
    print("\n=== PROFIT-FIRST COMPARE ===")
    print(rep.to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "gross_closed_pnl": lambda x: f"{x:.2f}",
        "net_closed_pnl": lambda x: f"{x:.2f}",
        "avg_net_pnl_per_trade": lambda x: f"{x:.2f}",
        "remove_best_trade_net": lambda x: f"{x:.2f}" if x is not None and pd.notna(x) else "N/A",
        "gross_to_fee": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy_per_trade": lambda x: f"{x:.2f}",
        "effective_leverage": lambda x: f"{x:.2f}",
        "risk_per_trade": lambda x: f"{x:.2%}",
    }))


if __name__ == "__main__":
    main()
