#!/usr/bin/env python3
import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def run_segment(df_seg, min_breakout_age_long=3):
    strat = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=28,
        trend_strength_threshold_4h=0.0055,
        breakout_confirm_atr=0.12,
        breakout_body_atr=0.20,
        pullback_bars=4,
        pullback_max_depth_atr=0.40,
        first_pullback_only=False,
        max_pullbacks_long=3,
        max_pullbacks_short=1,
        min_breakout_age_long=min_breakout_age_long,
        rejection_wick_ratio_long=0.65,
        rejection_wick_ratio_short=0.80,
        allow_short=False,
        allow_same_bar_entry=False,
        breakout_valid_bars=12,
        atr_pct_low=0.0030,
        atr_pct_high=0.016,
    )
    sig = strat.generate_signals(df_seg)
    if sig.empty:
        return None, pd.DataFrame()

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
    trades = pd.DataFrame(out.attrs.get("closed_trades", []))
    eq = out["equity"]
    dd = (eq / eq.cummax() - 1).min()
    row = {
        "return": float(eq.iloc[-1] / INITIAL_CASH - 1),
        "max_dd": float(dd),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "profit_factor": stats.get("profit_factor", float("nan")),
        "expectancy": float(stats.get("expectancy_per_trade", 0.0)),
        "trades": int(stats.get("closed_trade_count", 0)),
        "fees": float(stats.get("total_fees", 0.0)),
        "net_closed_pnl": float(stats.get("net_closed_pnl", 0.0)),
    }
    return row, trades


def summarize_windows(rep: pd.DataFrame):
    nonzero = rep[rep["trades"] > 0].copy()
    return {
        "windows": len(rep),
        "nonzero_windows": int((rep["trades"] > 0).sum()),
        "nonzero_window_ratio": float((rep["trades"] > 0).mean()),
        "avg_trades_per_window": float(rep["trades"].mean()),
        "median_return_nonzero": float(nonzero["return"].median()) if not nonzero.empty else 0.0,
        "median_pf_nonzero": float(nonzero["profit_factor"].dropna().median()) if not nonzero["profit_factor"].dropna().empty else np.nan,
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    idx = df.index
    start = idx.min()
    end = idx.max()

    rows_v6 = []
    rows_v61 = []
    trades_v6 = []
    trades_v61 = []

    cur = start
    while cur < end:
        seg_end = cur + pd.Timedelta(days=90)
        seg = df[(df.index >= cur) & (df.index < seg_end)]
        if len(seg) < 24 * 40:
            break

        r6, t6 = run_segment(seg, min_breakout_age_long=3)
        r61, t61 = run_segment(seg, min_breakout_age_long=2)

        if r6 is not None:
            r6["from"] = str(cur); r6["to"] = str(seg_end); rows_v6.append(r6)
            if not t6.empty:
                t6 = t6.copy(); t6["window_from"] = str(cur); trades_v6.append(t6)
        if r61 is not None:
            r61["from"] = str(cur); r61["to"] = str(seg_end); rows_v61.append(r61)
            if not t61.empty:
                t61 = t61.copy(); t61["window_from"] = str(cur); trades_v61.append(t61)

        cur += pd.Timedelta(days=30)

    rep6 = pd.DataFrame(rows_v6)
    rep61 = pd.DataFrame(rows_v61)
    tr6 = pd.concat(trades_v6, ignore_index=True) if trades_v6 else pd.DataFrame()
    tr61 = pd.concat(trades_v61, ignore_index=True) if trades_v61 else pd.DataFrame()

    print("\n=== V6 vs V6.1 WALK-FORWARD SUMMARY ===")
    s6 = summarize_windows(rep6)
    s61 = summarize_windows(rep61)
    print("V6:", s6)
    print("V6.1:", s61)

    if not rep61.empty:
        print("\n--- V6.1 windows ---")
        print(rep61.to_string(index=False, formatters={
            "return": lambda x: f"{x:.2%}",
            "max_dd": lambda x: f"{x:.2%}",
            "win_rate": lambda x: f"{x:.2%}",
            "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "expectancy": lambda x: f"{x:.2f}",
            "fees": lambda x: f"{x:.2f}",
        }))

    def remove_best_trade(df):
        if df.empty or "realized_net" not in df.columns or len(df) <= 1:
            return None
        d = df.copy()
        d["realized_net"] = d["realized_net"].astype(float)
        best = d["realized_net"].idxmax()
        d = d.drop(best)
        return float(d["realized_net"].sum())

    print("\n=== Best-trade sensitivity ===")
    print("V6 remove best trade net:", remove_best_trade(tr6))
    print("V6.1 remove best trade net:", remove_best_trade(tr61))

    if not tr61.empty:
        tr61 = tr61.copy()
        tr61["pullback_depth"] = tr61["entry_pullback_depth"].astype(float)
        tr61["breakout_age"] = tr61["breakout_age_at_entry"].astype(float)
        tr61["fees"] = tr61["fee_accum"].astype(float)
        tr61["gross_pnl"] = tr61["realized_gross"].astype(float)
        tr61["net_pnl"] = tr61["realized_net"].astype(float)
        tr61["weak_followthrough"] = tr61["mfe"].astype(float) < (tr61["fees"] * 3)
        print("\n=== V6.1 trade diagnostics ===")
        print(tr61[["entry_time","net_pnl","gross_pnl","fees","pullback_depth","breakout_age","mfe","mae","weak_followthrough"]].to_string(index=False))


if __name__ == "__main__":
    main()
