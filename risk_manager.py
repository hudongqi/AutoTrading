from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_total_exposure: float = 0.30
    reduce_risk_multiplier: float = 0.5
    max_consecutive_losses: int = 3
    daily_max_drawdown: float = 0.05
    allow_long_only: bool = False


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def _available_equity(self, balance: dict) -> float:
        usdt = balance.get("USDT", {}) if isinstance(balance, dict) else {}
        return float(usdt.get("free", usdt.get("total", 0.0)))

    def compute_order_size(self, price: float, atr: float, decision_action: str, confidence: float,
                           event_signals: dict, balance: dict) -> Dict[str, Any]:
        equity = self._available_equity(balance)
        if equity <= 0 or price <= 0:
            return {"ok": False, "qty": 0.0, "reason": "invalid equity/price"}

        risk_pct = self.cfg.risk_per_trade * max(0.2, confidence)

        # 风险模式缩仓
        if decision_action == "reduce_only" or str(event_signals.get("risk_mode", "NORMAL")).upper() == "REDUCE_RISK":
            risk_pct *= self.cfg.reduce_risk_multiplier

        stop_distance = max(price * 0.003, atr * 1.5 if atr > 0 else price * 0.005)
        risk_amount = equity * risk_pct
        qty = risk_amount / stop_distance

        # 总敞口上限
        max_notional = equity * self.cfg.max_total_exposure
        qty_cap = max_notional / price
        qty = min(qty, qty_cap)

        qty = max(0.0, qty)

        return {
            "ok": qty > 0,
            "qty": qty,
            "risk_pct": risk_pct,
            "risk_amount": risk_amount,
            "stop_distance": stop_distance,
            "max_notional": max_notional,
            "equity": equity,
        }

    def build_exit_levels(self, side: str, price: float, atr: float):
        atr = atr if atr and atr > 0 else price * 0.005
        if side == "buy":
            sl = price - 1.8 * atr
            tp = price + 3.0 * atr
        else:
            sl = price + 1.8 * atr
            tp = price - 3.0 * atr
        return sl, tp
