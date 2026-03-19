#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
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


def run_v6():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()

    strat = BTCPerpPullbackStrategy1H(
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
    stats = out.attrs.get("stats", {})
    trades = pd.DataFrame(out.attrs.get("closed_trades", []))
    return out, stats, trades


def bucket_stats(df, col, bucket_name):
    rows = []
    for label, g in df.groupby(bucket_name, dropna=False):
        if len(g) == 0:
            continue
        pnls = g["realized_net"].astype(float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) else np.nan
        rows.append({
            "bucket": str(label),
            "count": len(g),
            "win_rate": float((pnls > 0).mean()),
            "avg_gross_pnl": float(g["realized_gross"].mean()),
            "avg_net_pnl": float(g["realized_net"].mean()),
            "profit_factor": float(pf) if pd.notna(pf) else np.nan,
            "avg_MFE": float(g["mfe"].mean()),
            "avg_MAE": float(g["mae"].mean()),
            "avg_holding_bars": float(g["holding_bars"].mean()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "dimension", col)
    return out


def main():
    out, stats, trades = run_v6()
    if trades.empty:
        print("No closed trades")
        return

    trades = trades.copy()
    trades["fees"] = trades["fee_accum"].astype(float)
    trades["funding"] = trades["funding_accum"].astype(float)
    trades["gross_pnl"] = trades["realized_gross"].astype(float)
    trades["net_pnl"] = trades["realized_net"].astype(float)
    trades["breakout_age"] = trades["breakout_age_at_entry"].astype(float)
    trades["pullback_depth"] = trades["entry_pullback_depth"].astype(float)
    trades["adx_4h"] = trades["entry_adx_4h"].astype(float)
    trades["low_followthrough"] = trades["mfe"].astype(float) < (trades["fees"].astype(float) * 3)

    # rejection quality via wick ratio proxy: MFE/MAE 不是 rejection 本身，但 trade log 目前未保留 exact wick ratio，先用 pullback+MFE/MAE 结合排查
    # 这里构造 rejection_quality 桶：用 MFE/MAE 做后验跟随质量分层
    ratio = trades["mfe"] / trades["mae"].replace(0, np.nan)
    trades["rejection_quality_bucket"] = pd.cut(
        ratio,
        bins=[-np.inf, 0.8, 1.5, np.inf],
        labels=["weak_rejection", "medium_rejection", "strong_rejection"]
    )
    trades["pullback_depth_bucket"] = pd.cut(
        trades["pullback_depth"],
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=["<=0.4ATR", "0.4~0.7ATR", ">0.7ATR"]
    )
    trades["breakout_age_bucket"] = pd.cut(
        trades["breakout_age"],
        bins=[0, 2, 4, np.inf],
        labels=["1~2 bars", "3~4 bars", "5+ bars"]
    )

    print("\n==== V6 CURRENT DIAGNOSIS ====")
    print("closed trades:", len(trades))
    print("avg gross pnl per trade:", round(float(trades["gross_pnl"].mean()), 4))
    print("avg net pnl per trade:", round(float(trades["net_pnl"].mean()), 4))
    print("fees per trade:", round(float(trades["fees"].mean()), 4))
    gross_fee_ratio = float(trades["gross_pnl"].mean()) / max(1e-9, float(trades["fees"].mean()))
    net_fee_ratio = float(trades["net_pnl"].mean()) / max(1e-9, float(trades["fees"].mean()))
    print("gross-to-fee ratio:", round(gross_fee_ratio, 4))
    print("net-to-fee ratio:", round(net_fee_ratio, 4))

    sorted_tr = trades.sort_values("net_pnl")
    worst_n = max(1, int(len(sorted_tr) * 0.2))
    worst = sorted_tr.head(worst_n)
    print("\n==== WORST TRADES (bottom 20%) ====")
    print(worst[[
        "entry_time", "net_pnl", "gross_pnl", "fees", "pullback_depth", "breakout_age",
        "adx_4h", "mfe", "mae", "low_followthrough", "holding_bars"
    ]].to_string(index=False))

    print("\n==== BUCKET ANALYSIS ====")
    bucket_frames = [
        bucket_stats(trades, "pullback_depth", "pullback_depth_bucket"),
        bucket_stats(trades, "breakout_age", "breakout_age_bucket"),
        bucket_stats(trades, "rejection_quality", "rejection_quality_bucket"),
    ]
    report = pd.concat([x for x in bucket_frames if not x.empty], ignore_index=True)
    print(report.to_string(index=False, formatters={
        "win_rate": lambda x: f"{x:.2%}",
        "avg_gross_pnl": lambda x: f"{x:.2f}",
        "avg_net_pnl": lambda x: f"{x:.2f}",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "avg_MFE": lambda x: f"{x:.2f}",
        "avg_MAE": lambda x: f"{x:.2f}",
        "avg_holding_bars": lambda x: f"{x:.2f}",
    }))

    # 去掉最好1笔后的稳健性
    if len(trades) >= 2:
        ex_best = trades.drop(trades["net_pnl"].idxmax())
        print("\n==== ROBUSTNESS CHECK ====")
        print("remove best trade => net pnl:", round(float(ex_best["net_pnl"].sum()), 4))
        print("remove best trade => avg net pnl/trade:", round(float(ex_best["net_pnl"].mean()), 4))
        print("best-trade share of total net:", round(float(trades["net_pnl"].max() / max(1e-9, trades["net_pnl"].sum())), 4))


if __name__ == "__main__":
    main()
