import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H, BTCPerpPullbackStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def load_research_overlay(event_file="event_signals.json"):
    """
    交易过滤层（研究层）:
    - 技术策略决定 setup
    - 研究层决定是否允许开仓、是否降风险
    """
    defaults = {
        "block_new_entries": False,
        "reduce_risk": False,
        "risk_mode": "NORMAL",
        "market_bias": "NEUTRAL",
        "leverage_mult": 1.0,
        "max_pos_mult": 1.0,
        "reason": "default",
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

    block_new_entries = bool(macro.get("block", False) or geo.get("block_new_entries", False))
    reduce_risk = bool(macro.get("reduce_risk", False) or geo.get("reduce_risk", False) or risk_mode == "REDUCE_RISK")

    leverage_mult = 1.0
    max_pos_mult = 1.0
    if reduce_risk:
        leverage_mult = 0.6
        max_pos_mult = 0.5

    return {
        "block_new_entries": block_new_entries,
        "reduce_risk": reduce_risk,
        "risk_mode": risk_mode,
        "market_bias": str(data.get("market_bias", "NEUTRAL")).upper(),
        "leverage_mult": leverage_mult,
        "max_pos_mult": max_pos_mult,
        "reason": f"macro={macro.get('reason', '')}; geo={geo.get('reason', '')}",
    }


def run_case(
    name: str,
    df_sig,
    strat,
    entry_is_maker=False,
    funding_rate_per_8h=0.0,
    leverage=2.0,
    max_pos=0.8,
    cooldown_bars=3,
    stop_atr=1.5,
    take_R=3.5,
    trail_start_R=1.5,
    trail_atr=2.0,
    risk_per_trade=0.0075,
    enable_risk_position_sizing=True,
    allow_reentry=True,
    show_result_tail=False,
    debug_breakpoint=False,
    research_overlay=None,
):
    research_overlay = research_overlay or load_research_overlay()

    if research_overlay["block_new_entries"]:
        max_pos = 0.0

    leverage = max(1.0, leverage * research_overlay["leverage_mult"])
    max_pos = max(0.0, max_pos * research_overlay["max_pos_mult"])

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
        cooldown_bars=cooldown_bars,
        stop_atr=stop_atr,
        take_R=take_R,
        trail_start_R=trail_start_R,
        trail_atr=trail_atr,
        use_trailing=True,
        check_liq=True,
        entry_is_maker=entry_is_maker,
        funding_rate_per_8h=funding_rate_per_8h,
        risk_per_trade=risk_per_trade,
        enable_risk_position_sizing=enable_risk_position_sizing,
        allow_reentry=allow_reentry,
    )

    result = bt.run(df_sig)
    stats = result.attrs.get("stats", {})
    eq = result["equity"]
    max_dd = (eq / eq.cummax() - 1).min()

    if show_result_tail:
        print(result[["close", "position", "equity", "margin_used", "free_margin", "exit_reason"]].tail(20))

    if debug_breakpoint:
        breakpoint()

    print(f"\n==== {name} ====")
    print("overlay:", research_overlay)
    print("final_equity:", f"{result['equity'].iloc[-1]:.4f}")
    print("return:", f"{(result['equity'].iloc[-1] / INITIAL_CASH - 1):.2%}")
    print("max_drawdown:", f"{max_dd:.2%}")
    print("fees:", f"{stats.get('total_fees', 0.0):.4f}")
    print("funding_total:", f"{stats.get('funding_total', 0.0):.4f}")
    print("win_rate:", f"{stats.get('win_rate', 0.0):.2%}")

    pnl_ratio = stats.get("pnl_ratio", float("nan"))
    print("pnl_ratio:", f"{pnl_ratio:.2f}" if pd.notna(pnl_ratio) else "N/A")
    print("expectancy_per_trade:", f"{stats.get('expectancy_per_trade', 0.0):.4f}")
    pf = stats.get("profit_factor", float("nan"))
    print("profit_factor:", f"{pf:.2f}" if pd.notna(pf) else "N/A")
    print("avg_holding_bars:", f"{stats.get('avg_holding_bars', 0.0):.2f}")
    print("time_in_market:", f"{stats.get('time_in_market', 0.0):.2%}")
    print("long_short_split:", stats.get("long_short_split", {}))
    print("mfe_avg:", f"{stats.get('mfe_avg', 0.0):.2f}", "mae_avg:", f"{stats.get('mae_avg', 0.0):.2f}")
    print("fees_per_trade:", f"{stats.get('fees_per_trade', 0.0):.4f}")
    print("gross_closed_pnl:", f"{stats.get('gross_closed_pnl', 0.0):.2f}", "net_closed_pnl:", f"{stats.get('net_closed_pnl', 0.0):.2f}")
    print("long_side:", stats.get("long_side", {}))
    print("short_side:", stats.get("short_side", {}))
    print("worst_trades_summary:", stats.get("worst_trades_summary", {}))
    print("rejected_reason_count:", stats.get("rejected_reason_count", {}))

    return {
        "case": name,
        "final_equity": float(result['equity'].iloc[-1]),
        "return": float(result['equity'].iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(max_dd),
        "fees": float(stats.get('total_fees', 0.0)),
        "win_rate": float(stats.get('win_rate', 0.0)),
        "pnl_ratio": float(stats.get('pnl_ratio', float('nan'))) if pd.notna(stats.get('pnl_ratio', float('nan'))) else np.nan,
        "profit_factor": float(stats.get('profit_factor', float('nan'))) if pd.notna(stats.get('profit_factor', float('nan'))) else np.nan,
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    overlay = load_research_overlay("event_signals.json")

    reports = []

    # 改造前（宽松版）
    strat_v2 = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=30,
        trend_strength_threshold_4h=0.006,
        breakout_confirm_atr=0.15,
        breakout_body_atr=0.25,
        pullback_bars=3,
        pullback_max_depth_atr=0.60,
        first_pullback_only=False,
        atr_pct_low=0.0035,
        atr_pct_high=0.015,
    )
    df_sig_v2 = strat_v2.generate_signals(df)
    reports.append(run_case(
        "LIVE_LIKE_PULLBACK_RISK_V2_BASELINE",
        df_sig_v2,
        strat_v2,
        entry_is_maker=False,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        leverage=2.0,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.4,
        take_R=2.6,
        trail_start_R=1.0,
        trail_atr=2.2,
        show_result_tail=False,
        debug_breakpoint=False,
        research_overlay=overlay,
    ))

    # 改造后（严格过滤版）
    strat_v3 = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=32,
        trend_strength_threshold_4h=0.0065,
        breakout_confirm_atr=0.20,
        breakout_body_atr=0.35,
        pullback_bars=3,
        pullback_max_depth_atr=0.45,
        first_pullback_only=True,
        atr_pct_low=0.0035,
        atr_pct_high=0.013,
    )
    df_sig_v3 = strat_v3.generate_signals(df)
    reports.append(run_case(
        "LIVE_LIKE_PULLBACK_RISK_V3_FILTERED",
        df_sig_v3,
        strat_v3,
        entry_is_maker=False,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        leverage=2.0,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.4,
        take_R=2.6,
        trail_start_R=1.0,
        trail_atr=2.2,
        show_result_tail=True,
        debug_breakpoint=False,
        research_overlay=overlay,
    ))

    print("\n==== BEFORE vs AFTER SUMMARY ====")
    rep = pd.DataFrame(reports)
    print(rep.to_string(index=False, formatters={
        "return": lambda x: f"{x:.2%}",
        "max_drawdown": lambda x: f"{x:.2%}",
        "fees": lambda x: f"{x:.2f}",
        "win_rate": lambda x: f"{x:.2%}",
        "pnl_ratio": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "profit_factor": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "final_equity": lambda x: f"{x:.2f}",
    }))


if __name__ == "__main__":
    main()
