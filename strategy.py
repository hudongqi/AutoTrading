import pandas as pd
import numpy as np

class BTCPerpTrendStrategy1H:

    def __init__(self, fast=20, slow=60, atr_period=14,min_volatility=0.002):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.min_volatility = min_volatility

    @staticmethod
    def _atr(df, period):

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean()

    def generate_signals(self, df):

        out = df.copy()

        out["ma_fast"] = out["close"].rolling(self.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.slow).mean()

        out["atr"] = self._atr(out, self.atr_period)

        # ===== 波动率（归一化）=====
        out["volatility"] = out["atr"] / out["close"]

        # ===== 波动过滤 =====
        out["vol_ok"] = out["volatility"] >= self.min_volatility

        out["signal"] = 0

        long_cond = (out["ma_fast"] > out["ma_slow"]) & out["vol_ok"]
        short_cond = (out["ma_fast"] < out["ma_slow"]) & out["vol_ok"]

        out.loc[long_cond, "signal"] = 1
        out.loc[short_cond, "signal"] = -1

        out["trade_signal"] = out["signal"].diff().fillna(0)

        return out.dropna()
