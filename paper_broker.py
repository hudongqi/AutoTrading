from dataclasses import dataclass
from typing import Dict, Any, Optional
import uuid

from portfolio_state import PortfolioState, PortfolioStateStore, now_iso


@dataclass
class PaperOrderResult:
    ok: bool
    order_id: str
    status: str
    side: str
    amount: float
    price: float
    fee: float
    raw: Dict[str, Any]


class PaperBroker:
    """真实行情驱动 + 本地模拟成交，不发交易所订单"""

    def __init__(self, state_store: PortfolioStateStore, initial_cash: float, fee_rate: float = 0.0005, slippage_bps: float = 5):
        self.state_store = state_store
        self.state = self.state_store.load(initial_cash=initial_cash)
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_rate

    def _fill_price(self, last_price: float, side: str) -> float:
        slip = last_price * self.slippage_bps / 10000.0
        return last_price + slip if side == "buy" else last_price - slip

    def fetch_balance(self) -> dict:
        return {"USDT": {"free": self.state.cash, "total": self.state.equity}}

    def fetch_positions(self, symbols=None):
        out = []
        for sym, p in self.state.positions.items():
            if symbols and sym not in symbols:
                continue
            out.append({"symbol": sym, **p})
        return out

    def create_market_order(self, symbol: str, side: str, amount: float, last_price: float, params=None) -> PaperOrderResult:
        params = params or {}
        if amount <= 0:
            return PaperOrderResult(False, "", "rejected", side, amount, last_price, 0.0, {"reason": "amount<=0"})

        fill = self._fill_price(last_price, side)
        notional = fill * amount
        fee = self._fee(notional)

        p = self.state.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0, "side": "flat"})
        old_qty = float(p["qty"])
        signed = amount if side == "buy" else -amount
        new_qty = old_qty + signed

        realized = 0.0
        old_avg = float(p["avg_price"]) if old_qty != 0 else fill

        if old_qty == 0 or (old_qty > 0 and signed > 0) or (old_qty < 0 and signed < 0):
            # 开仓或同向加仓
            total_cost = abs(old_qty) * old_avg + abs(signed) * fill
            avg = total_cost / max(1e-12, abs(new_qty))
            p = {"qty": new_qty, "avg_price": avg, "side": "long" if new_qty > 0 else "short" if new_qty < 0 else "flat"}
        else:
            # 反向：先平后反手
            close_qty = min(abs(old_qty), abs(signed))
            if old_qty > 0:
                realized = (fill - old_avg) * close_qty
            else:
                realized = (old_avg - fill) * close_qty

            if new_qty == 0:
                p = {"qty": 0.0, "avg_price": 0.0, "side": "flat"}
            elif (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
                p = {"qty": new_qty, "avg_price": fill, "side": "long" if new_qty > 0 else "short"}
            else:
                p = {"qty": new_qty, "avg_price": old_avg, "side": "long" if new_qty > 0 else "short"}

        self.state.positions[symbol] = p
        self.state.cash += realized - fee
        self.state.realized_pnl += realized - fee
        self.state.daily_trade_count += 1
        self.state.last_execution_time = now_iso()
        self.state.last_action_by_symbol[symbol] = side

        oid = f"PAPER-{uuid.uuid4().hex[:12]}"
        self.state.open_orders[oid] = {
            "id": oid,
            "symbol": symbol,
            "type": "market",
            "status": "closed",
            "side": side,
            "amount": amount,
            "price": fill,
            "fee": fee,
            "params": params,
            "ts": now_iso(),
        }
        self.state_store.save(self.state)

        return PaperOrderResult(True, oid, "closed", side, amount, fill, fee, self.state.open_orders[oid])

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params=None) -> PaperOrderResult:
        # 简化：记录 open limit，后续 execution_engine 在触发条件下撮合
        params = params or {}
        oid = f"PAPER-{uuid.uuid4().hex[:12]}"
        self.state.open_orders[oid] = {
            "id": oid, "symbol": symbol, "type": "limit", "status": "open", "side": side,
            "amount": amount, "price": price, "fee": 0.0, "params": params, "ts": now_iso(),
        }
        self.state_store.save(self.state)
        return PaperOrderResult(True, oid, "open", side, amount, price, 0.0, self.state.open_orders[oid])

    def set_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> PaperOrderResult:
        return self.create_limit_order(symbol, "sell" if side == "buy" else "buy", amount, stop_price, params={"kind": "sl", "reduceOnly": True})

    def set_take_profit(self, symbol: str, side: str, amount: float, take_price: float) -> PaperOrderResult:
        return self.create_limit_order(symbol, "sell" if side == "buy" else "buy", amount, take_price, params={"kind": "tp", "reduceOnly": True})

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return self.state.open_orders.get(order_id, {"id": order_id, "status": "not_found", "symbol": symbol})

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        od = self.state.open_orders.get(order_id)
        if od:
            od["status"] = "canceled"
            self.state_store.save(self.state)
            return od
        return {"id": order_id, "status": "not_found", "symbol": symbol}

    def mark_to_market(self, symbol: str, last_price: float):
        p = self.state.positions.get(symbol)
        upnl = 0.0
        if p and abs(float(p.get("qty", 0.0))) > 0:
            qty = float(p["qty"])
            upnl = (last_price - float(p["avg_price"])) * qty
        self.state.unrealized_pnl = upnl
        self.state.equity = self.state.cash + upnl
        self.state_store.save(self.state)
