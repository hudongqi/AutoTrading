#!/usr/bin/env python3
"""
Smart-money whale signal ingestion (辅助增强因子)

设计原则:
- 只接入“历史胜率高、经常提前布局”的 smart money 观察结果
- 输出结构化字段: whale_score / whale_bias / whale_reason
- 仅用于候选筛选、仓位调节、入场置信度增强
- 不单独触发开仓/平仓，不因单笔大额转账直接下交易结论
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

POOL_SYMBOLS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]


@dataclass
class WhaleSignal:
    whale_score: float = 0.0              # [-1, 1]
    whale_bias: str = "NEUTRAL"           # BULLISH / BEARISH / NEUTRAL
    whale_reason: str = "No validated smart-money signal"


class WhaleSignalCollector:
    def __init__(self, root: Path):
        self.root = root
        self.manual_file = root / "research" / "whale" / "manual_smart_money_signals.json"
        self.daily_dir = root / "research" / "whale" / "daily"
        self.event_file = root / "event_signals.json"

    def _ensure_files(self):
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.manual_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.manual_file.exists():
            template = {
                "note": "仅填写经过历史验证的 smart money 观点；不要基于单笔大额转账直接给结论",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "symbols": {
                    s: {
                        "whale_score": 0.0,
                        "whale_bias": "NEUTRAL",
                        "whale_reason": "No validated smart-money signal",
                    }
                    for s in POOL_SYMBOLS
                },
            }
            self.manual_file.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_signal(raw: dict) -> WhaleSignal:
        score = float(raw.get("whale_score", 0.0))
        score = max(-1.0, min(1.0, score))

        bias = str(raw.get("whale_bias", "NEUTRAL")).upper().strip()
        if bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            bias = "NEUTRAL"

        reason = str(raw.get("whale_reason", "No validated smart-money signal")).strip()
        if not reason:
            reason = "No validated smart-money signal"

        return WhaleSignal(whale_score=score, whale_bias=bias, whale_reason=reason)

    def collect(self) -> Dict[str, WhaleSignal]:
        self._ensure_files()
        raw = json.loads(self.manual_file.read_text(encoding="utf-8"))
        symbols = raw.get("symbols", {})

        out = {}
        for s in POOL_SYMBOLS:
            out[s] = self._normalize_signal(symbols.get(s, {}))
        return out

    def write_daily_snapshot(self, signals: Dict[str, WhaleSignal]) -> Path:
        now = datetime.now(timezone.utc)
        fp = self.daily_dir / f"{now.strftime('%Y-%m-%d')}_whale_signals.json"
        data = {
            "generated_at": now.isoformat(),
            "note": "Smart-money whale signals are auxiliary only; never standalone entry/exit triggers.",
            "symbols": {k: asdict(v) for k, v in signals.items()},
        }
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return fp

    def merge_into_event_signals(self, signals: Dict[str, WhaleSignal]) -> Path:
        if not self.event_file.exists():
            raise FileNotFoundError(f"Missing {self.event_file}")

        data = json.loads(self.event_file.read_text(encoding="utf-8"))
        data.setdefault("whale", {})
        avg_score = sum(v.whale_score for v in signals.values()) / max(1, len(signals))
        if avg_score > 0.2:
            agg_bias = "BULLISH"
        elif avg_score < -0.2:
            agg_bias = "BEARISH"
        else:
            agg_bias = "NEUTRAL"

        data["whale"].update({
            "enabled": True,
            "mode": "auxiliary",
            "note": "Only for sizing/confidence/candidate filtering, never standalone trade trigger.",
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "whale_score": round(avg_score, 4),
            "whale_bias": agg_bias,
            "whale_reason": "Aggregated from validated smart-money symbol signals",
        })

        data.setdefault("market_bias", "NEUTRAL")
        data.setdefault("risk_mode", "NORMAL")
        data.setdefault("sentiment", {"bias": "NEUTRAL", "strength": "LOW", "recommended_action": "WAIT", "reason": "not computed yet"})
        data.setdefault("geopolitics", {"reduce_risk": False, "block_new_entries": False, "alts_bias": "NEUTRAL", "reason": "No major geopolitical escalation"})

        data.setdefault("symbols", {})
        for s in POOL_SYMBOLS:
            data["symbols"].setdefault(s, {})
            ws = signals[s]
            data["symbols"][s].setdefault("block", False)
            data["symbols"][s].setdefault("reduce_risk", False)
            data["symbols"][s].setdefault("reason", "")
            data["symbols"][s].setdefault("news_signals", [])
            data["symbols"][s].setdefault("bias", "NEUTRAL")
            data["symbols"][s].setdefault("strength", "LOW")
            data["symbols"][s].setdefault("recommended_action", "WAIT")
            data["symbols"][s]["whale_score"] = ws.whale_score
            data["symbols"][s]["whale_bias"] = ws.whale_bias
            data["symbols"][s]["whale_reason"] = ws.whale_reason

        self.event_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.event_file


def main():
    root = Path(__file__).resolve().parent
    collector = WhaleSignalCollector(root)
    signals = collector.collect()
    snap = collector.write_daily_snapshot(signals)
    event_file = collector.merge_into_event_signals(signals)

    print(f"whale snapshot: {snap}")
    print(f"event signals updated: {event_file}")


if __name__ == "__main__":
    main()
