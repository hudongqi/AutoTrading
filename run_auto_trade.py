#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from market_data import MarketDataClient
from strategy import BTCPerpTrendStrategy1H
from decision_engine import DecisionEngine
from risk_manager import RiskManager, RiskConfig
from trade_logger import TradeLogger
from portfolio_state import PortfolioStateStore
from paper_broker import PaperBroker
from exchange_broker import ExchangeBroker
from execution_engine import ExecutionEngine


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_event_signals(path="event_signals.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_ccxt_symbol(raw: str) -> str:
    if "/" in raw:
        return raw
    base = raw.replace("USDT", "")
    return f"{base}/USDT:USDT"


def from_ccxt_symbol(sym: str) -> str:
    return sym.split("/")[0] + "USDT"


def ensure_symbol_overlay(event_signals: dict, symbol_raw: str):
    event_signals.setdefault("symbols", {})
    if symbol_raw not in event_signals["symbols"]:
        mb = str(event_signals.get("market_bias", "NEUTRAL")).upper()
        default_action, default_bias = "WAIT", "NEUTRAL"
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


def compute_latest_signal(md: MarketDataClient, ccxt_symbol: str, timeframe: str, limit: int = 5000) -> dict:
    df = md.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=limit)
    strat = BTCPerpTrendStrategy1H(fast=5, slow=15)
    sig = strat.generate_signals(df)
    if sig.empty:
        # 数据不足时也返回可解释结构，避免静默
        last = df.iloc[-1].to_dict()
        atr = max(1e-9, float(last["high"] - last["low"]))
        return {
            "close": float(last["close"]),
            "atr": atr,
            "signal": 0,
            "trade_signal": 0.0,
            "bar_time": df.index[-1].isoformat(),
            "tech_note": "insufficient bars for full strategy confirmation",
        }
    row = sig.iloc[-1].to_dict()
    row["bar_time"] = sig.index[-1].isoformat()
    return row


def run_backtest_mode(cfg: dict):
    # 保持现有 backtest 入口
    import main as backtest_main
    backtest_main.main()


def run_loop(mode: str, cfg: dict):
    symbols = [to_ccxt_symbol(s) for s in cfg.get("symbols", ["BTCUSDT"])]
    timeframe = cfg.get("timeframe", "1m")
    poll = int(cfg.get("polling_interval_seconds", 30))
    max_loops = int(cfg.get("max_loops", 1))

    md = MarketDataClient(exchange=cfg.get("exchange", "binanceusdm"), use_testnet=bool(cfg.get("use_testnet", False)))

    if mode == "paper":
        store = PortfolioStateStore(cfg.get("state_path", "state/paper_state.json"))
        broker = PaperBroker(
            state_store=store,
            initial_cash=float(cfg.get("initial_cash", 10000)),
            fee_rate=float(cfg.get("fee_rate", 0.0005)),
            slippage_bps=float(cfg.get("slippage_bps", 5)),
        )
    else:
        broker = ExchangeBroker(exchange=cfg.get("exchange", "binanceusdm"), mode=mode)

    de = DecisionEngine(
        min_strength=str(cfg.get("min_strength", "MEDIUM")),
        require_technical_confirm=bool(cfg.get("require_technical_confirm", True)),
    )
    rm = RiskManager(RiskConfig(
        risk_per_trade=float(cfg.get("risk_per_trade", 0.01)),
        max_total_exposure=float(cfg.get("max_total_exposure", 0.3)),
        reduce_risk_multiplier=float(cfg.get("reduce_risk_factor", 0.5)),
        allow_long_only=bool(cfg.get("allow_long", True)) and not bool(cfg.get("allow_short", True)),
    ))
    logger = TradeLogger(cfg.get("audit_log", "logs/trade_audit.jsonl"))
    ex_engine = ExecutionEngine(de, rm, logger)

    last_bar = {}

    for i in range(max_loops):
        event_signals = load_event_signals(cfg.get("event_signals_path", "event_signals.json"))

        for sym in symbols:
            raw = from_ccxt_symbol(sym)
            ensure_symbol_overlay(event_signals, raw)

            tech = compute_latest_signal(md, sym, timeframe=timeframe)
            if not tech:
                continue

            bar_time = tech.get("bar_time")
            # 只在新K线决策，避免重复
            if last_bar.get(sym) == bar_time:
                continue
            last_bar[sym] = bar_time

            price = float(tech.get("close", 0.0))
            atr = float(tech.get("atr", 0.0))

            payload = ex_engine.execute_symbol(mode, raw, sym, event_signals, tech, broker, price, atr)

            if mode == "paper":
                broker.mark_to_market(sym, price)

        if i < max_loops - 1:
            time.sleep(max(1, poll))


def main():
    parser = argparse.ArgumentParser(description="Auto trading runner (backtest/paper/testnet)")
    parser.add_argument("--mode", choices=["backtest", "paper", "testnet"], default=None)
    parser.add_argument("--config", default="config/trading.yaml")
    parser.add_argument("--loops", type=int, default=None, help="override max_loops")
    args = parser.parse_args()

    base = load_yaml(args.config)
    mode = args.mode or base.get("mode", "paper")

    mode_cfg = {}
    if mode == "paper":
        mode_cfg = load_yaml("config/paper.yaml")
    elif mode == "testnet":
        mode_cfg = load_yaml("config/testnet.yaml")

    cfg = {**base, **mode_cfg}
    if args.loops is not None:
        cfg["max_loops"] = args.loops

    print(f"[AUTO-TRADE] mode={mode} exchange={cfg.get('exchange')} timeframe={cfg.get('timeframe')}")

    if mode == "backtest":
        run_backtest_mode(cfg)
        return

    run_loop(mode, cfg)


if __name__ == "__main__":
    main()
