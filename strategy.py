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


class BTCPerpPullbackStrategy1H:
    """
    改进版：4h 趋势过滤 + 突破质量 + 首次回踩 + rejection 确认
    """

    def __init__(
        self,
        fast_4h=5,
        slow_4h=15,
        adx_period_4h=14,
        adx_threshold_4h=30,
        trend_strength_threshold_4h=0.006,
        breakout_window=24 * 7,
        breakout_confirm_atr=0.2,
        breakout_body_atr=0.35,
        pullback_bars=3,
        pullback_max_depth_atr=0.45,
        first_pullback_only=True,
        atr_period=14,
        atr_pct_low=0.0035,
        atr_pct_high=0.013,
        ema_entry=20,
    ):
        self.fast_4h = fast_4h
        self.slow_4h = slow_4h
        self.adx_period_4h = adx_period_4h
        self.adx_threshold_4h = adx_threshold_4h
        self.trend_strength_threshold_4h = trend_strength_threshold_4h
        self.breakout_window = breakout_window
        self.breakout_confirm_atr = breakout_confirm_atr
        self.breakout_body_atr = breakout_body_atr
        self.pullback_bars = pullback_bars
        self.pullback_max_depth_atr = pullback_max_depth_atr
        self.first_pullback_only = first_pullback_only
        self.atr_period = atr_period
        self.atr_pct_low = atr_pct_low
        self.atr_pct_high = atr_pct_high
        self.ema_entry = ema_entry

    @staticmethod
    def _atr(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _adx(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff(); down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["atr"] = self._atr(out, self.atr_period)
        out["atr_pct"] = out["atr"] / out["close"]
        out["volatility"] = out["atr_pct"]
        out["ema_entry"] = out["close"].ewm(span=self.ema_entry, adjust=False).mean()

        out_4h = out[["open", "high", "low", "close", "volume"]].resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        out_4h["ma_fast_4h"] = out_4h["close"].rolling(self.fast_4h).mean()
        out_4h["ma_slow_4h"] = out_4h["close"].rolling(self.slow_4h).mean()
        out_4h["adx_4h"] = self._adx(out_4h, self.adx_period_4h)
        out_4h["trend_strength_4h"] = (out_4h["ma_fast_4h"] - out_4h["ma_slow_4h"]).abs() / out_4h["close"]

        out["ma_fast_4h"] = out_4h["ma_fast_4h"].reindex(out.index, method="ffill")
        out["ma_slow_4h"] = out_4h["ma_slow_4h"].reindex(out.index, method="ffill")
        out["adx_4h"] = out_4h["adx_4h"].reindex(out.index, method="ffill")
        out["trend_strength_4h"] = out_4h["trend_strength_4h"].reindex(out.index, method="ffill")

        out["regime_ok"] = (
            (out["adx_4h"] >= self.adx_threshold_4h) &
            (out["trend_strength_4h"] >= self.trend_strength_threshold_4h) &
            (out["atr_pct"] >= self.atr_pct_low) &
            (out["atr_pct"] <= self.atr_pct_high)
        )

        out["state_signal"] = 0
        out.loc[(out["ma_fast_4h"] > out["ma_slow_4h"]) & out["regime_ok"], "state_signal"] = 1
        out.loc[(out["ma_fast_4h"] < out["ma_slow_4h"]) & out["regime_ok"], "state_signal"] = -1

        out["resistance_7d"] = out["high"].rolling(self.breakout_window, min_periods=self.breakout_window).quantile(0.975)
        out["support_7d"] = out["low"].rolling(self.breakout_window, min_periods=self.breakout_window).quantile(0.025)

        body = (out["close"] - out["open"]).abs()
        long_break = out["close"] > (out["resistance_7d"] + self.breakout_confirm_atr * out["atr"])
        short_break = out["close"] < (out["support_7d"] - self.breakout_confirm_atr * out["atr"])
        out["breakout_quality_long"] = long_break & (body >= self.breakout_body_atr * out["atr"]) & (out["close"] > out["open"])
        out["breakout_quality_short"] = short_break & (body >= self.breakout_body_atr * out["atr"]) & (out["close"] < out["open"])

        # 标记最近一次 breakout 价位
        long_level = out["resistance_7d"].where(out["breakout_quality_long"])
        short_level = out["support_7d"].where(out["breakout_quality_short"])
        out["last_long_break_level"] = long_level.ffill()
        out["last_short_break_level"] = short_level.ffill()
        out["bars_since_long_break"] = out["breakout_quality_long"][::-1].cumsum()[::-1]  # proxy
        out["bars_since_short_break"] = out["breakout_quality_short"][::-1].cumsum()[::-1]

        # 回踩深度（越小越好）
        out["pullback_depth_long"] = ((out["last_long_break_level"] - out["low"]) / out["atr"]).clip(lower=0)
        out["pullback_depth_short"] = ((out["high"] - out["last_short_break_level"]) / out["atr"]).clip(lower=0)
        out["pullback_depth_ok_long"] = out["pullback_depth_long"] <= self.pullback_max_depth_atr
        out["pullback_depth_ok_short"] = out["pullback_depth_short"] <= self.pullback_max_depth_atr

        near_ema_long = (out["low"] <= out["ema_entry"]) & (out["close"] >= out["ema_entry"])
        near_brk_long = (out["low"] <= out["last_long_break_level"]) & (out["low"] >= out["last_long_break_level"] - self.pullback_max_depth_atr * out["atr"])
        near_ema_short = (out["high"] >= out["ema_entry"]) & (out["close"] <= out["ema_entry"])
        near_brk_short = (out["high"] >= out["last_short_break_level"]) & (out["high"] <= out["last_short_break_level"] + self.pullback_max_depth_atr * out["atr"])

        # 更严格 rejection
        lower_wick = (out["open"].where(out["open"] < out["close"], out["close"]) - out["low"]).clip(lower=0)
        upper_wick = (out["high"] - out["open"].where(out["open"] > out["close"], out["close"])).clip(lower=0)
        out["reject_long"] = (out["close"] > out["open"]) & (lower_wick >= body * 0.8)
        out["reject_short"] = (out["close"] < out["open"]) & (upper_wick >= body * 0.8)

        pullback_touch_long = (near_ema_long | near_brk_long)
        pullback_touch_short = (near_ema_short | near_brk_short)
        out["pullback_touch_long"] = pullback_touch_long
        out["pullback_touch_short"] = pullback_touch_short

        # 首次回踩（突破后首次触发）
        touch_count_long = pullback_touch_long.astype(int).groupby((out["breakout_quality_long"]).cumsum()).cumsum()
        touch_count_short = pullback_touch_short.astype(int).groupby((out["breakout_quality_short"]).cumsum()).cumsum()
        out["first_pullback_ok_long"] = (touch_count_long == 1) | (not self.first_pullback_only)
        out["first_pullback_ok_short"] = (touch_count_short == 1) | (not self.first_pullback_only)

        long_pullback_ok = pullback_touch_long.rolling(self.pullback_bars).max().fillna(0).astype(bool)
        short_pullback_ok = pullback_touch_short.rolling(self.pullback_bars).max().fillna(0).astype(bool)

        out["entry_setup"] = 0
        long_setup = (
            (out["state_signal"] == 1) &
            out["breakout_quality_long"] &
            long_pullback_ok &
            out["reject_long"] &
            out["pullback_depth_ok_long"] &
            out["first_pullback_ok_long"]
        )
        short_setup = (
            (out["state_signal"] == -1) &
            out["breakout_quality_short"] &
            short_pullback_ok &
            out["reject_short"] &
            out["pullback_depth_ok_short"] &
            out["first_pullback_ok_short"]
        )

        out.loc[long_setup, "entry_setup"] = 1
        out.loc[short_setup, "entry_setup"] = -1

        out["signal"] = out["state_signal"]
        out["trade_signal"] = out["entry_setup"]

        print("state long true:", float((out["state_signal"] == 1).mean()))
        print("state short true:", float((out["state_signal"] == -1).mean()))
        print("entry long setup:", float((out["entry_setup"] == 1).mean()))
        print("entry short setup:", float((out["entry_setup"] == -1).mean()))

        return out.dropna()

