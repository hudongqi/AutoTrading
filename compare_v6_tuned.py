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


def run_case(name, df, strat, overlay):
    sig = strat.generate_signals(df)
    leverage = 2.0 * overlay["leverage_mult"]
    max_pos = 0.8 * overlay["max_pos_mult"]
    if overlay["block_new_entries"]:
        max_pos = 0.0
    portfolio = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=leverage,
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
        risk_per_trade=0.0075,
        enable_risk_position_sizing=True,
        allow_reentry=True,
    )
    out = bt.run(sig)
    st = out.attrs["stats"]
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    return {
        "case": name,
        "final_equity": float(eq.iloc[-1]),
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(dd),
        "fees": float(st.get("total_fees", 0.0)),
        "win_rate": float(st.get("win_rate", 0.0)),
        "pnl_ratio": st.get("pnl_ratio", float("nan")),
        "profit_factor": st.get("profit_factor", float("nan")),
        "expectancy_per_trade": float(st.get("expectancy_per_trade", 0.0)),
        "fees_per_trade": float(st.get("fees_per_trade", 0.0)),
        "gross_closed_pnl": float(st.get("gross_closed_pnl", 0.0)),
        "net_closed_pnl": float(st.get("net_closed_pnl", 0.0)),
        "trade_count": int(st.get("closed_trade_count", 0)),
        "time_in_market": float(st.get("time_in_market", 0.0)),
        "rejected_reason_count": st.get("rejected_reason_count", {}),
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()

    v6 = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=28,
        trend_strength_threshold_4h=0.0055,
        breakout_confirm_atr=0.12,
        breakout_body_atr=0.20,
        pullback_bars=4,
        pullback_max_depth_atr=0.90,
        first_pullback_only=False,
        max_pullbacks_long=3,
        max_pullbacks_short=1,
        rejection_wick_ratio_long=0.40,
        rejection_wick_ratio_short=0.80,
        allow_short=False,
        allow_same_bar_entry=False,
        breakout_valid_bars=12,
        atr_pct_low=0.0030,
        atr_pct_high=0.016,
    )

    tuned = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=28,
        trend_strength_threshold_4h=0.0055,
        breakout_confirm_atr=0.12,
        breakout_body_atr=0.20,
        pullback_bars=4,
        pullback_max_depth_atr=0.40,
        first_pullback_only=False,
        max_pullbacks_long=3,
        max_pullbacks_short=1,
        min_breakout_age_long=3,
        rejection_wick_ratio_long=0.65,
        rejection_wick_ratio_short=0.80,
        allow_short=False,
        allow_same_bar_entry=False,
        breakout_valid_bars=12,
        atr_pct_low=0.0030,
        atr_pct_high=0.016,
    )

    rows = [run_case("V6_BASE", df, v6, overlay), run_case("V6_TUNED", df, tuned, overlay)]
    rep = pd.DataFrame(rows)
    print("\n==== V6 DIAGNOSED TUNING COMPARE ====")
    print(rep.drop(columns=["rejected_reason_count"]).to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "win_rate": lambda x: f"{x:.2%}",
        "pnl_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "expectancy_per_trade": lambda x: f"{x:.2f}",
        "fees_per_trade": lambda x: f"{x:.2f}",
        "gross_closed_pnl": lambda x: f"{x:.2f}",
        "net_closed_pnl": lambda x: f"{x:.2f}",
        "time_in_market": lambda x: f"{x:.2%}",
        "final_equity": lambda x: f"{x:.2f}",
    }))
    print("\nRejected reasons V6_BASE:", rows[0]["rejected_reason_count"])
    print("Rejected reasons V6_TUNED:", rows[1]["rejected_reason_count"])


if __name__ == "__main__":
    main()
