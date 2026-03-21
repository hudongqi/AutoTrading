import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import *
from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H, SOLMeanReversionStrategy1H
from strategy_profiles import get_strategy_profile, get_sol_backtest_profile, list_profiles, BACKTEST_COMMON
from exit_profiles import get_exit_profile, list_exit_profiles
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester


def load_research_overlay(event_file="event_signals.json"):
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


def run_case(name: str, df_sig, strat, research_overlay=None, show_result_tail=True, debug_breakpoint=False, exit_profile="exit_baseline", **params):
    research_overlay = research_overlay or load_research_overlay()
    cfg = {**BACKTEST_COMMON, **get_exit_profile(exit_profile), **params}

    leverage = max(1.0, cfg["leverage"] * research_overlay["leverage_mult"])
    max_pos = max(0.0, cfg["max_pos"] * research_overlay["max_pos_mult"])
    if research_overlay["block_new_entries"]:
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
        cooldown_bars=cfg["cooldown_bars"],
        stop_atr=cfg["stop_atr"],
        take_R=cfg["take_R"],
        trail_start_R=cfg["trail_start_R"],
        trail_atr=cfg["trail_atr"],
        use_trailing=cfg.get("use_trailing", True),
        check_liq=True,
        entry_is_maker=cfg["entry_is_maker"],
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=cfg["risk_per_trade"],
        enable_risk_position_sizing=cfg["enable_risk_position_sizing"],
        allow_reentry=cfg["allow_reentry"],
        partial_take_R=cfg["partial_take_R"],
        partial_take_frac=cfg["partial_take_frac"],
        break_even_after_partial=cfg.get("break_even_after_partial", False),
        break_even_R=cfg.get("break_even_R", 0.0),
        use_signal_exit_targets=cfg.get("use_signal_exit_targets", False),
        max_hold_bars=cfg.get("max_hold_bars", 0),
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
    mc = stats.get('mfe_capture_ratio', float('nan'))
    gb = stats.get('give_back_ratio', float('nan'))
    ar = stats.get('avg_R_realized', float('nan'))
    print("mfe_capture_ratio:", f"{mc:.2f}" if pd.notna(mc) else "N/A")
    print("give_back_ratio:", f"{gb:.2f}" if pd.notna(gb) else "N/A")
    print("avg_R_realized:", f"{ar:.2f}" if pd.notna(ar) else "N/A")
    print("exit_reason_split:", stats.get('exit_reason_split', {}))
    print("partial_take_effectiveness:", stats.get('partial_take_effectiveness', float('nan')))

    return {
        "profile": name,
        "final_equity": float(result['equity'].iloc[-1]),
        "return": float(result['equity'].iloc[-1] / INITIAL_CASH - 1),
        "max_drawdown": float(max_dd),
        "trade_count": int(stats.get('closed_trade_count', 0)),
        "fees": float(stats.get('total_fees', 0.0)),
        "win_rate": float(stats.get('win_rate', 0.0)),
        "pnl_ratio": float(stats.get('pnl_ratio', float('nan'))) if pd.notna(stats.get('pnl_ratio', float('nan'))) else np.nan,
        "profit_factor": float(stats.get('profit_factor', float('nan'))) if pd.notna(stats.get('profit_factor', float('nan'))) else np.nan,
        "expectancy_per_trade": float(stats.get('expectancy_per_trade', 0.0)),
        "gross_closed_pnl": float(stats.get('gross_closed_pnl', 0.0)),
        "net_closed_pnl": float(stats.get('net_closed_pnl', 0.0)),
        "fees_per_trade": float(stats.get('fees_per_trade', 0.0)),
        "mfe_capture_ratio": float(stats.get('mfe_capture_ratio', float('nan'))) if pd.notna(stats.get('mfe_capture_ratio', float('nan'))) else np.nan,
        "give_back_ratio": float(stats.get('give_back_ratio', float('nan'))) if pd.notna(stats.get('give_back_ratio', float('nan'))) else np.nan,
        "avg_R_realized": float(stats.get('avg_R_realized', float('nan'))) if pd.notna(stats.get('avg_R_realized', float('nan'))) else np.nan,
        "exit_reason_split": stats.get('exit_reason_split', {}),
    }


def build_strategy(profile_name: str):
    cfg = get_strategy_profile(profile_name)
    if profile_name.startswith("sol_"):
        return SOLMeanReversionStrategy1H(**cfg)
    return BTCPerpPullbackStrategy1H(**cfg)


def resolve_market(profile_name: str, symbol: str | None, start: str | None, end: str | None):
    resolved_symbol = symbol or ("SOL/USDT:USDT" if profile_name.startswith("sol_") else SYMBOL)
    resolved_start = start or START
    resolved_end = end or END
    return resolved_symbol, resolved_start, resolved_end


def main():
    parser = argparse.ArgumentParser(description="Run main backtest for a strategy profile")
    parser.add_argument("--profile", default="v6_2_sample_up", choices=list_profiles())
    parser.add_argument("--exit-profile", default="exit_loose_runner", choices=list_exit_profiles())
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--max-pos", type=float, default=1.2)
    parser.add_argument("--risk-per-trade", type=float, default=0.015)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    resolved_symbol, resolved_start, resolved_end = resolve_market(args.profile, args.symbol, args.start, args.end)
    ds = CCXTDataSource()
    df = ds.load_ohlcv(resolved_symbol, resolved_start, resolved_end)
    overlay = load_research_overlay("event_signals.json")

    strat = build_strategy(args.profile)
    df_sig = strat.generate_signals(df)
    extra_cfg = {}
    if args.profile.startswith("sol_"):
        extra_cfg = get_sol_backtest_profile(args.profile)
        extra_cfg.update({
            "leverage": args.leverage,
            "max_pos": args.max_pos,
            "risk_per_trade": args.risk_per_trade,
        })

    run_params = {
        "leverage": args.leverage,
        "max_pos": args.max_pos,
        "risk_per_trade": args.risk_per_trade,
    }
    run_params.update(extra_cfg)

    run_case(
        f"LIVE_LIKE_{args.profile}_{args.exit_profile}_{resolved_symbol}",
        df_sig,
        strat,
        show_result_tail=True,
        research_overlay=overlay,
        exit_profile=args.exit_profile,
        **run_params,
    )


if __name__ == "__main__":
    main()
