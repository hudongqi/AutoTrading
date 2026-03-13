from datetime import datetime, timezone
from typing import Dict, Any

from decision_engine import DecisionEngine
from risk_manager import RiskManager


class ExecutionEngine:
    """统一执行协调器：决策 -> 风控 -> broker -> 日志"""

    def __init__(self, decision_engine: DecisionEngine, risk_manager: RiskManager, logger):
        self.decision_engine = decision_engine
        self.risk_manager = risk_manager
        self.logger = logger

    def execute_symbol(self, mode: str, symbol_raw: str, symbol_ccxt: str, event_signals: dict, tech_row: dict, broker, price: float, atr: float):
        decision = self.decision_engine.decide(symbol_raw, event_signals, tech_row)
        balance = broker.fetch_balance()
        risk = self.risk_manager.compute_order_size(price, atr, decision.action, decision.confidence, event_signals, balance)

        ex = {"status": "no_trade", "side": "none", "qty": 0.0}
        action = decision.action

        if action in {"buy", "sell", "reduce_only"} and risk.get("ok", False):
            side = "buy" if decision.side == "buy" else "sell"
            qty = float(risk["qty"])

            if mode == "paper":
                entry = broker.create_market_order(symbol_ccxt, side, qty, last_price=price)
            else:
                entry = broker.create_market_order(symbol_ccxt, side, qty)

            sl, tp = self.risk_manager.build_exit_levels(side, price, atr)
            sl_o = broker.set_stop_loss(symbol_ccxt, side, qty, sl)
            tp_o = broker.set_take_profit(symbol_ccxt, side, qty, tp)

            ex = {
                "status": "executed" if entry.ok else "failed",
                "side": side,
                "qty": qty,
                "entry_price": entry.price or price,
                "fee": getattr(entry, "fee", None),
                "stop_loss": sl,
                "take_profit": tp,
                "entry_order": entry.raw,
                "sl_order": sl_o.raw,
                "tp_order": tp_o.raw,
            }
        elif action in {"block_new_entry", "hold"}:
            ex = {"status": "no_trade", "side": "none", "qty": 0.0}

        reason_lines = [
            f"mode={mode}",
            f"market_bias={event_signals.get('market_bias')} risk_mode={event_signals.get('risk_mode')}",
            f"symbol.bias={event_signals.get('symbols', {}).get(symbol_raw, {}).get('bias')}",
            f"symbol.strength={event_signals.get('symbols', {}).get(symbol_raw, {}).get('strength')}",
            f"symbol.recommended_action={event_signals.get('symbols', {}).get(symbol_raw, {}).get('recommended_action')}",
            f"whale_bias={event_signals.get('symbols', {}).get(symbol_raw, {}).get('whale_bias')} whale_score={event_signals.get('symbols', {}).get(symbol_raw, {}).get('whale_score')}",
            f"technical.signal={tech_row.get('signal')} trade_signal={tech_row.get('trade_signal')}",
            f"risk.ok={risk.get('ok')} qty={risk.get('qty')} risk_pct={risk.get('risk_pct')}",
            f"final_action={decision.action} side={decision.side} confidence={decision.confidence:.2f}",
        ]

        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "symbol": symbol_raw,
            "action": decision.action,
            "reason_lines": reason_lines,
            "execution": ex,
        }
        self.logger.print_decision(payload)
        self.logger.log(payload)
        return payload
