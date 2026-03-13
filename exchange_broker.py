import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import ccxt


@dataclass
class ExchangeOrderResult:
    ok: bool
    order_id: Optional[str]
    status: str
    side: str
    amount: float
    price: Optional[float]
    raw: Dict[str, Any]


class ExchangeBroker:
    """交易所执行层：testnet/live 统一入口"""

    def __init__(self, exchange: str = "binanceusdm", mode: str = "testnet"):
        key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")
        ex_cls = getattr(ccxt, exchange)
        self.ex = ex_cls({
            "enableRateLimit": True,
            "apiKey": key,
            "secret": secret,
            "options": {"defaultType": "future"},
        })
        self.mode = mode
        if mode == "testnet" and hasattr(self.ex, "set_sandbox_mode"):
            self.ex.set_sandbox_mode(True)

    def fetch_balance(self) -> dict:
        return self.ex.fetch_balance()

    def fetch_positions(self, symbols=None):
        try:
            return self.ex.fetch_positions(symbols)
        except Exception:
            return []

    def create_market_order(self, symbol: str, side: str, amount: float, params=None) -> ExchangeOrderResult:
        od = self.ex.create_order(symbol, "market", side, amount, None, params or {})
        return ExchangeOrderResult(True, str(od.get("id")), str(od.get("status", "unknown")), side, amount, od.get("average"), od)

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params=None) -> ExchangeOrderResult:
        od = self.ex.create_order(symbol, "limit", side, amount, price, params or {})
        return ExchangeOrderResult(True, str(od.get("id")), str(od.get("status", "unknown")), side, amount, price, od)

    def set_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> ExchangeOrderResult:
        order_side = "sell" if side == "buy" else "buy"
        params = {"stopPrice": stop_price, "reduceOnly": True}
        od = self.ex.create_order(symbol, "stop_market", order_side, amount, None, params)
        return ExchangeOrderResult(True, str(od.get("id")), str(od.get("status", "open")), order_side, amount, stop_price, od)

    def set_take_profit(self, symbol: str, side: str, amount: float, take_price: float) -> ExchangeOrderResult:
        order_side = "sell" if side == "buy" else "buy"
        params = {"stopPrice": take_price, "reduceOnly": True}
        od = self.ex.create_order(symbol, "take_profit_market", order_side, amount, None, params)
        return ExchangeOrderResult(True, str(od.get("id")), str(od.get("status", "open")), order_side, amount, take_price, od)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return self.ex.fetch_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self.ex.cancel_order(order_id, symbol)
