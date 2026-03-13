import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


class TradeLogger:
    def __init__(self, path: str = "logs/trade_audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def print_decision(self, payload: Dict[str, Any]):
        print("\n==== 交易决策 ====")
        print(f"时间: {payload.get('time')}")
        print(f"标的: {payload.get('symbol')}")
        print(f"动作: {payload.get('action')}")
        print("判断依据:")
        for line in payload.get("reason_lines", []):
            print(f"- {line}")
        print("执行结果:")
        ex = payload.get("execution", {})
        for k in ["side", "qty", "entry_price", "stop_loss", "take_profit", "status"]:
            print(f"- {k}: {ex.get(k)}")
