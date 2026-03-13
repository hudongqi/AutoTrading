#!/usr/bin/env python3
"""
全自动执行入口（第一版）
流程：
1) 读取 event_signals.json（研究层结论）
2) 拉取最新 K 线并生成技术信号
3) decision_engine 生成动作
4) risk_manager 计算仓位
5) live_broker 执行下单 + 保护单
6) trade_logger 记录审计日志

默认 DRY_RUN=True，先模拟执行，再切换实盘。
"""

import argparse
import json
from datetime import datetime, timezone

from config import SYMBOL
from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from live_broker import LiveBroker
from decision_engine import DecisionEngine
from risk_manager import RiskManager, RiskConfig
from trade_logger import TradeLogger


def load_event_signals(path="event_signals.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def symbol_raw_from_ccxt_symbol(symbol_ccxt: str) -> str:
    # BTC/USDT:USDT -> BTCUSDT
    base = symbol_ccxt.split("/")[0]
    return f"{base}USDT"


def build_reason_lines(symbol_raw, decision, event_signals, tech_row, risk_info):
    macro = event_signals.get("macro", {})
    geo = event_signals.get("geopolitics", {})
    sym = event_signals.get("symbols", {}).get(symbol_raw, {})
    return [
        f"macro.block={macro.get('block')} macro.reduce_risk={macro.get('reduce_risk')}",
        f"geopolitics.block_new_entries={geo.get('block_new_entries')} geopolitics.reduce_risk={geo.get('reduce_risk')}",
        f"market_bias={event_signals.get('market_bias')} risk_mode={event_signals.get('risk_mode')}",
        f"symbol.bias={sym.get('bias')} strength={sym.get('strength')} recommended_action={sym.get('recommended_action')}",
        f"whale_bias={sym.get('whale_bias')} whale_score={sym.get('whale_score')} whale_reason={sym.get('whale_reason')}",
        f"technical.signal={tech_row.get('signal')} technical.trade_signal={tech_row.get('trade_signal')}",
        f"decision.action={decision.action} decision.side={decision.side} confidence={decision.confidence:.2f}",
        f"risk.qty={risk_info.get('qty')} risk_pct={risk_info.get('risk_pct')} stop_distance={risk_info.get('stop_distance')}",
    ]


def run_once(dry_run: bool = True, long_only: bool = False):
    event_signals = load_event_signals("event_signals.json")

    ds = CCXTDataSource()
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = "2026-01-01"
    df = ds.load_ohlcv(SYMBOL, start, end, timeframe="1h")

    strat = BTCPerpTrendStrategy1H(fast=5, slow=15)
    sig = strat.generate_signals(df)
    latest = sig.iloc[-1].to_dict()
    price = float(latest["close"])
    atr = float(latest.get("atr", 0.0))

    symbol_raw = symbol_raw_from_ccxt_symbol(SYMBOL)

    # 若研究层未覆盖当前交易标的，降级使用市场级判断（避免系统中断）
    event_signals.setdefault("symbols", {})
    if symbol_raw not in event_signals["symbols"]:
        mb = str(event_signals.get("market_bias", "NEUTRAL")).upper()
        default_action = "WAIT"
        default_bias = "NEUTRAL"
        if mb in {"BULLISH", "LONG"}:
            default_action, default_bias = "LONG", "LONG"
        elif mb in {"BEARISH", "SHORT"}:
            default_action, default_bias = "SHORT", "SHORT"
        event_signals["symbols"][symbol_raw] = {
            "block": False,
            "reduce_risk": False,
            "reason": "fallback from market_bias",
            "news_signals": [],
            "whale_score": event_signals.get("whale", {}).get("whale_score", 0.0),
            "whale_bias": event_signals.get("whale", {}).get("whale_bias", "NEUTRAL"),
            "whale_reason": "fallback aggregate whale signal",
            "bias": default_bias,
            "strength": "MEDIUM",
            "recommended_action": default_action,
        }

    decision_engine = DecisionEngine(min_strength="MEDIUM", require_technical_confirm=True)
    decision = decision_engine.decide(symbol_raw, event_signals, latest)

    broker = LiveBroker(dry_run=dry_run)
    balance = broker.fetch_balance()

    risk_mgr = RiskManager(RiskConfig(allow_long_only=long_only))
    risk_info = risk_mgr.compute_order_size(price, atr, decision.action, decision.confidence, event_signals, balance)

    logger = TradeLogger("logs/trade_audit.jsonl")

    exec_result = {"status": "skipped", "side": decision.side, "qty": 0}

    if decision.action in {"buy", "sell", "reduce_only"} and risk_info.get("ok", False):
        if long_only and decision.side == "sell":
            exec_result = {"status": "blocked_long_only", "side": "sell", "qty": 0}
        else:
            side = "buy" if decision.side == "buy" else "sell"
            qty = float(risk_info["qty"])
            entry = broker.create_market_order(SYMBOL, side, qty)
            sl, tp = risk_mgr.build_exit_levels(side, price, atr)
            sl_o = broker.set_stop_loss(SYMBOL, side, qty, sl)
            tp_o = broker.set_take_profit(SYMBOL, side, qty, tp)
            exec_result = {
                "status": "executed" if entry.ok else "failed",
                "side": side,
                "qty": qty,
                "entry_price": entry.price or price,
                "stop_loss": sl,
                "take_profit": tp,
                "entry_order": entry.raw,
                "sl_order": sl_o.raw,
                "tp_order": tp_o.raw,
            }
    elif decision.action in {"block_new_entry", "hold"}:
        exec_result = {"status": "no_trade", "side": "none", "qty": 0}

    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol_raw,
        "action": decision.action,
        "reason_lines": build_reason_lines(symbol_raw, decision, event_signals, latest, risk_info),
        "execution": exec_result,
    }
    logger.print_decision(payload)
    logger.log(payload)


def main():
    parser = argparse.ArgumentParser(description="Run automated trading execution")
    parser.add_argument("--live", action="store_true", help="execute real orders (default dry-run)")
    parser.add_argument("--long-only", action="store_true", help="allow long only")
    args = parser.parse_args()

    run_once(dry_run=not args.live, long_only=args.long_only)


if __name__ == "__main__":
    main()
