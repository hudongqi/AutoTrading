from typing import Dict, Any, List, Optional

import ccxt
import pandas as pd


class MarketDataClient:
    """统一市场数据层：paper/testnet/live 都从这里读行情"""

    def __init__(self, exchange: str = "binanceusdm", use_testnet: bool = False):
        ex_cls = getattr(ccxt, exchange)
        self.ex = ex_cls({"enableRateLimit": True})
        if use_testnet and hasattr(self.ex, "set_sandbox_mode"):
            self.ex.set_sandbox_mode(True)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("time").drop(columns=["ts"]).sort_index()

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.ex.fetch_ticker(symbol)

    def fetch_last_price(self, symbol: str) -> float:
        t = self.fetch_ticker(symbol)
        return float(t.get("last") or t.get("close") or 0.0)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        return self.ex.fetch_order_book(symbol, limit=limit)

    def fetch_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.ex.fetch_trades(symbol, limit=limit)

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            fr = self.ex.fetch_funding_rate(symbol)
            return float(fr.get("fundingRate")) if fr.get("fundingRate") is not None else None
        except Exception:
            return None
