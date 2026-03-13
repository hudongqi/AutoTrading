import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import ccxt


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    status: str
    side: str
    amount: float
    price: Optional[float]
    raw: Dict[str, Any]


class LiveBroker:
    """真实执行层（支持 dry-run）"""

    def __init__(self, exchange_id: str = "binanceusdm", dry_run: bool = True):
        self.dry_run = dry_run
        key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")

        ex_cls = getattr(ccxt, exchange_id)
        self.ex = ex_cls({
            "enableRateLimit": True,
            "apiKey": key,
            "secret": secret,
            "options": {"defaultType": "future"},
        })

    def fetch_balance(self) -> dict:
        if self.dry_run:
            return {"USDT": {"free": 10000.0, "total": 10000.0}}
        return self.ex.fetch_balance()

    def fetch_positions(self, symbols=None):
        if self.dry_run:
            return []
        try:
            return self.ex.fetch_positions(symbols)
        except Exception:
            return []

    def fetch_ticker(self, symbol: str) -> dict:
        return self.ex.fetch_ticker(symbol)

    def create_market_order(self, symbol: str, side: str, amount: float, params=None) -> OrderResult:
        params = params or {}
        if self.dry_run:
            return OrderResult(True, "DRY_MARKET", "closed", side, amount, None, {"dry_run": True, "params": params})
        od = self.ex.create_order(symbol, "market", side, amount, None, params)
        return OrderResult(True, str(od.get("id")), str(od.get("status", "unknown")), side, amount, od.get("average"), od)

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params=None) -> OrderResult:
        params = params or {}
        if self.dry_run:
            return OrderResult(True, "DRY_LIMIT", "open", side, amount, price, {"dry_run": True, "params": params})
        od = self.ex.create_order(symbol, "limit", side, amount, price, params)
        return OrderResult(True, str(od.get("id")), str(od.get("status", "unknown")), side, amount, price, od)

    def set_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> OrderResult:
        """简化：用 stop_market 保护单（Binance USDM）"""
        params = {"stopPrice": stop_price, "reduceOnly": True}
        order_side = "sell" if side.lower() == "buy" else "buy"
        if self.dry_run:
            return OrderResult(True, "DRY_SL", "open", order_side, amount, stop_price, {"dry_run": True})
        od = self.ex.create_order(symbol, "stop_market", order_side, amount, None, params)
        return OrderResult(True, str(od.get("id")), str(od.get("status", "open")), order_side, amount, stop_price, od)

    def set_take_profit(self, symbol: str, side: str, amount: float, take_price: float) -> OrderResult:
        params = {"stopPrice": take_price, "reduceOnly": True}
        order_side = "sell" if side.lower() == "buy" else "buy"
        if self.dry_run:
            return OrderResult(True, "DRY_TP", "open", order_side, amount, take_price, {"dry_run": True})
        od = self.ex.create_order(symbol, "take_profit_market", order_side, amount, None, params)
        return OrderResult(True, str(od.get("id")), str(od.get("status", "open")), order_side, amount, take_price, od)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        if self.dry_run:
            return {"id": order_id, "status": "closed", "dry_run": True}
        return self.ex.fetch_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        if self.dry_run:
            return {"id": order_id, "status": "canceled", "dry_run": True}
        return self.ex.cancel_order(order_id, symbol)
