from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Decision:
    action: str  # buy/sell/hold/reduce_only/close_position/block_new_entry
    side: str
    confidence: float
    reason: Dict[str, Any]


class DecisionEngine:
    """研究层 + 技术层 的统一决策引擎"""

    def __init__(self, min_strength: str = "MEDIUM", require_technical_confirm: bool = True):
        self.min_strength = min_strength
        self.require_technical_confirm = require_technical_confirm
        self._strength_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def decide(self, symbol: str, event_signals: dict, tech_row: dict) -> Decision:
        macro = event_signals.get("macro", {})
        geo = event_signals.get("geopolitics", {})
        sym = event_signals.get("symbols", {}).get(symbol, {})
        sentiment = event_signals.get("sentiment", {})
        risk_mode = str(event_signals.get("risk_mode", "NORMAL")).upper()

        # 第一层：市场总开关
        if macro.get("block", False):
            return Decision("block_new_entry", "none", 0.0, {
                "market": "macro.block=true",
                "macro": macro,
            })

        if geo.get("block_new_entries", False):
            return Decision("block_new_entry", "none", 0.0, {
                "market": "geopolitics.block_new_entries=true",
                "geopolitics": geo,
            })

        if sym.get("block", False):
            return Decision("block_new_entry", "none", 0.0, {
                "symbol": f"{symbol}.block=true",
                "symbol_state": sym,
            })

        # 第二层：币种筛选
        strength = str(sym.get("strength", "LOW")).upper()
        if self._strength_rank.get(strength, 1) < self._strength_rank.get(self.min_strength, 2):
            return Decision("hold", "none", 0.2, {
                "symbol": f"strength<{self.min_strength}",
                "symbol_state": sym,
            })

        bias = str(sym.get("bias", "NEUTRAL")).upper()
        rec = str(sym.get("recommended_action", "WAIT")).upper()

        desired_side = "none"
        if bias in {"LONG", "BULLISH", "BUY"} and rec in {"LONG", "BUY"}:
            desired_side = "buy"
        elif bias in {"SHORT", "BEARISH", "SELL"} and rec in {"SHORT", "SELL"}:
            desired_side = "sell"

        if desired_side == "none":
            return Decision("hold", "none", 0.25, {
                "symbol": "bias/recommended_action not aligned",
                "symbol_state": sym,
            })

        # whale 反向时降级
        whale_bias = str(sym.get("whale_bias", "NEUTRAL")).upper()
        whale_score = float(sym.get("whale_score", 0.0))
        whale_penalty = 0.0
        if (desired_side == "buy" and whale_bias == "BEARISH") or (desired_side == "sell" and whale_bias == "BULLISH"):
            whale_penalty = min(0.25, abs(whale_score) * 0.25)

        # 第三层：技术面确认
        ts = float(tech_row.get("trade_signal", 0.0))
        signal = int(tech_row.get("signal", 0))
        tech_ok = False
        if desired_side == "buy" and (signal == 1 or ts > 0):
            tech_ok = True
        if desired_side == "sell" and (signal == -1 or ts < 0):
            tech_ok = True

        if self.require_technical_confirm and not tech_ok:
            return Decision("hold", "none", 0.3, {
                "technical": "no valid entry confirmation",
                "tech": {"signal": signal, "trade_signal": ts},
                "symbol_state": sym,
            })

        action = desired_side
        if macro.get("reduce_risk", False) or geo.get("reduce_risk", False) or sym.get("reduce_risk", False) or risk_mode == "REDUCE_RISK":
            action = "reduce_only"

        base_conf = 0.7 if strength == "HIGH" else 0.55
        base_conf -= whale_penalty
        if sentiment.get("bias", "NEUTRAL").upper() in {"BULLISH", "BEARISH"}:
            base_conf += 0.05
        base_conf = max(0.05, min(0.95, base_conf))

        return Decision(action, desired_side, base_conf, {
            "market": {
                "risk_mode": risk_mode,
                "macro_reduce_risk": macro.get("reduce_risk", False),
                "geo_reduce_risk": geo.get("reduce_risk", False),
            },
            "symbol": sym,
            "technical": {"signal": signal, "trade_signal": ts},
            "sentiment": sentiment,
            "whale_penalty": whale_penalty,
        })
