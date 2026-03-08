import pandas as pd
import numpy as np

class BTCPerpTrendStrategy1H:

    def __init__(
        self,
        fast=5,
        slow=15,
        atr_period=14,
        atr_pct_threshold=0.0045,
        fast_4h=5,
        slow_4h=15,
        use_regime_filter=False,
        adx_period_4h=14,
        adx_threshold_4h=28,
        trend_strength_threshold_4h=0.006,
        slow_slope_lookback_4h=3,
    ):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_pct_threshold = atr_pct_threshold
        self.fast_4h = fast_4h
        self.slow_4h = slow_4h
        self.use_regime_filter = use_regime_filter
        self.adx_period_4h = adx_period_4h
        self.adx_threshold_4h = adx_threshold_4h
        self.trend_strength_threshold_4h = trend_strength_threshold_4h
        self.slow_slope_lookback_4h = slow_slope_lookback_4h
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

    @staticmethod
    def _adx(df, period=14):
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr

        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx

    def generate_signals(self, df):

        out = df.copy()

        out["ma_fast"] = out["close"].rolling(self.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.slow).mean()

        out["atr"] = self._atr(out, self.atr_period)

        # ===== 波动率（归一化）=====
        out["atr_pct"] = out["atr"] / out["close"]
        out["volatility"] = out["atr_pct"]

        # ===== 波动过滤 =====
        out["vol_ok"] = out["atr_pct"] > self.atr_pct_threshold

        # ===== 7天支撑/压力（1h = 168根）=====
        WINDOW = 7 * 24
        q = 0.975

        out["resistance_7d"] = out["high"].rolling(WINDOW, min_periods=WINDOW).quantile(q)
        out["support_7d"] = out["low"].rolling(WINDOW, min_periods=WINDOW).quantile(1 - q)

        # ===== 4h 趋势过滤 =====
        out_4h = out[["open", "high", "low", "close", "volume"]].resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        out_4h["ma_fast_4h"] = out_4h["close"].rolling(self.fast_4h).mean()
        out_4h["ma_slow_4h"] = out_4h["close"].rolling(self.slow_4h).mean()
        out_4h["trend_4h"] = 0
        out_4h.loc[out_4h["ma_fast_4h"] > out_4h["ma_slow_4h"], "trend_4h"] = 1
        out_4h.loc[out_4h["ma_fast_4h"] < out_4h["ma_slow_4h"], "trend_4h"] = -1

        out_4h["adx_4h"] = self._adx(out_4h, self.adx_period_4h)
        out_4h["trend_strength_4h"] = (out_4h["ma_fast_4h"] - out_4h["ma_slow_4h"]).abs() / out_4h["close"]
        out_4h["ma_slow_slope_4h"] = out_4h["ma_slow_4h"].pct_change(self.slow_slope_lookback_4h)

        out["ma_fast_4h"] = out_4h["ma_fast_4h"].reindex(out.index, method="ffill")
        out["ma_slow_4h"] = out_4h["ma_slow_4h"].reindex(out.index, method="ffill")
        out["trend_4h"] = out_4h["trend_4h"].reindex(out.index, method="ffill").fillna(0).astype(int)
        out["adx_4h"] = out_4h["adx_4h"].reindex(out.index, method="ffill")
        out["trend_strength_4h"] = out_4h["trend_strength_4h"].reindex(out.index, method="ffill")
        out["ma_slow_slope_4h"] = out_4h["ma_slow_slope_4h"].reindex(out.index, method="ffill")
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

        if self.use_regime_filter:
            regime_long = (
                (out["adx_4h"] >= self.adx_threshold_4h) &
                (out["trend_strength_4h"] >= self.trend_strength_threshold_4h) &
                (out["ma_slow_slope_4h"] > 0)
            )
            regime_short = (
                (out["adx_4h"] >= self.adx_threshold_4h) &
                (out["trend_strength_4h"] >= self.trend_strength_threshold_4h) &
                (out["ma_slow_slope_4h"] < 0)
            )
        else:
            regime_long = True
            regime_short = True

        long_cond = (
            (out["ma_fast"] > out["ma_slow"]) &
            (out["trend_4h"] == 1) &
            (out["atr_pct"] > self.atr_pct_threshold) &
            (out["close"] > out["resistance_7d"]) &
            regime_long
        )
        short_cond = (
            (out["ma_fast"] < out["ma_slow"]) &
            (out["trend_4h"] == -1) &
            (out["atr_pct"] > self.atr_pct_threshold) &
            (out["close"] < out["support_7d"]) &
            regime_short
        )

        out.loc[long_cond, "signal"] = 1
        out.loc[short_cond, "signal"] = -1

        # ===== 事件驱动：只在信号切换时交易 =====
        out["trade_signal"] = out["signal"].diff().fillna(0)
        out["trailing_active"] = False

        print("long_cond true:", long_cond.mean())
        print("short_cond true:", short_cond.mean())
        if self.use_regime_filter:
            print("regime_long true:", float((regime_long if isinstance(regime_long, pd.Series) else pd.Series([regime_long])).mean()))
            print("regime_short true:", float((regime_short if isinstance(regime_short, pd.Series) else pd.Series([regime_short])).mean()))
        # print("long_level_ok true:", long_level_ok.mean())
        # print("short_level_ok true:", short_level_ok.mean())

        return out.dropna()
