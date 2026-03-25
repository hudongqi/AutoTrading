import ccxt
import pandas as pd

class CCXTDataSource:
    def __init__(self, exchange_id="binanceusdm"):
        exchange_class = getattr(ccxt, exchange_id)
        self.ex = exchange_class({"enableRateLimit": True})

    def load_ohlcv(self, symbol, start, end, timeframe="1h", limit_per_call=500):

        since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

        rows = []

        while True:
            bars = self.ex.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit_per_call
            )

            if not bars:
                break

            rows.extend(bars)

            last_ts = bars[-1][0]
            if last_ts >= end_ms:
                break

            since = last_ts + 1

        df = pd.DataFrame(
            rows,
            columns=["ts", "open", "high", "low", "close", "volume"]
        )

        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("time").drop(columns=["ts"])

        end_ts = pd.Timestamp(end, tz="UTC")
        df = df[df.index <= end_ts]

        return df.sort_index()

    def load_funding_rates(self, symbol, start, end, limit_per_call=500):
        """拉取历史资金费率（每 8 小时一条），返回以 UTC 时间为索引的 Series。"""
        since  = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)
        rows = []
        while True:
            batch = self.ex.fetch_funding_rate_history(symbol, since=since, limit=limit_per_call)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if last_ts >= end_ms:
                break
            since = last_ts + 1

        if not rows:
            return pd.Series(dtype=float, name="funding_rate")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("time")[["fundingRate"]].sort_index()
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df["fundingRate"].rename("funding_rate")
