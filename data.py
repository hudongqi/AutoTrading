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
        df = df[df.index <= end]

        return df.sort_index()
