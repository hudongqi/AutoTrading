import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


@dataclass
class SymbolPosition:
    qty: float = 0.0
    avg_price: float = 0.0
    side: str = "flat"


@dataclass
class PortfolioState:
    cash: float = 10000.0
    equity: float = 10000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    consecutive_losses: int = 0
    day_start_equity: float = 10000.0
    last_execution_time: str = ""
    last_action_by_symbol: Dict[str, str] = field(default_factory=dict)
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    open_orders: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_trade_count: int = 0


class PortfolioStateStore:
    def __init__(self, path: str = "state/portfolio_state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, initial_cash: float) -> PortfolioState:
        if not self.path.exists():
            st = PortfolioState(cash=initial_cash, equity=initial_cash, day_start_equity=initial_cash)
            self.save(st)
            return st
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return PortfolioState(**data)

    def save(self, st: PortfolioState):
        self.path.write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
