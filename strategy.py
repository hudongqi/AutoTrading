import pandas as pd
import numpy as np

class BTCPerpTrendStrategy1H:

    def __init__(self, fast=5, slow=15, atr_period=14,min_volatility=0.002):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.min_volatility = min_volatility
        self.trail_start_atr = 1.5  # 盈利达到 1.5 ATR 才启动 trailing
        self.entry_price = None

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

        return tr.ewm(alpha=1/period, adjust=False).mean()

    def generate_signals(self, df):

        out = df.copy()

        out["ma_fast"] = out["close"].rolling(self.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.slow).mean()

        out["atr"] = self._atr(out, self.atr_period)

        # ===== 波动率（归一化）=====
        out["volatility"] = out["atr"] / out["close"]

        # ===== 波动过滤 =====
        out["vol_ok"] = out["volatility"] >= self.min_volatility
        #
        # # ===== 7天支撑/压力（1h = 168根）=====
        WINDOW = 7 * 24
        q = 0.975

        out["resistance_7d"] = out["high"].rolling(WINDOW, min_periods=WINDOW).quantile(q)
        out["support_7d"] = out["low"].rolling(WINDOW, min_periods=WINDOW).quantile(1 - q)
        #
        # # ===== 关键位缓冲（避免贴边交易）=====
        # # 可做成参数：self.level_buffer_atr = 0.3
        # level_buf = 0.3 * out["atr"]
        #
        # # ===== “确认离开关键位”条件 =====
        # # 做空：上一根还在压力位附近，这一根收盘确认跌回压力位下方 buffer
        # short_level_ok = (
        #         out["resistance_7d"].notna() &
        #         (out["high"].shift(1) >= out["resistance_7d"].shift(1) - level_buf.shift(1)) &
        #         (out["close"] < out["resistance_7d"] - level_buf)
        # )
        #
        # # 做多：上一根还在支撑位附近，这一根收盘确认站回支撑位上方 buffer
        # long_level_ok = (
        #         out["support_7d"].notna() &
        #         (out["low"].shift(1) <= out["support_7d"].shift(1) + level_buf.shift(1)) &
        #         (out["close"] > out["support_7d"] + level_buf)
        # )

        out["signal"] = 0
        long_trend = (out["ma_fast"] > out["ma_slow"]) & out["vol_ok"]
        short_trend = (out["ma_fast"] < out["ma_slow"]) & out["vol_ok"]

        # ✅ 只有趋势 + 关键位确认 同时满足，才允许给 signal

        out.loc[long_trend , "signal"] = 1
        out.loc[short_trend , "signal"] = -1

        # ===== 事件驱动：只在信号切换时交易 =====
        out["trade_signal"] = out["signal"].diff().fillna(0)
        out["trailing_active"] = False

        print("long_trend true:", long_trend.mean())
        print("short_trend true:", short_trend.mean())
        # print("long_level_ok true:", long_level_ok.mean())
        # print("short_level_ok true:", short_level_ok.mean())

        return out.dropna()
