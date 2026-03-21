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

    @staticmethod
    def _atr(df, period):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

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
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    def generate_signals(self, df):
        out = df.copy()
        out["ma_fast"] = out["close"].rolling(self.fast).mean()
        out["ma_slow"] = out["close"].rolling(self.slow).mean()
        out["atr"] = self._atr(out, self.atr_period)
        out["atr_pct"] = out["atr"] / out["close"]
        out["volatility"] = out["atr_pct"]
        window = 7 * 24
        q = 0.975
        out["resistance_7d"] = out["high"].rolling(window, min_periods=window).quantile(q)
        out["support_7d"] = out["low"].rolling(window, min_periods=window).quantile(1 - q)
        out_4h = out[["open", "high", "low", "close", "volume"]].resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        out_4h["ma_fast_4h"] = out_4h["close"].rolling(self.fast_4h).mean()
        out_4h["ma_slow_4h"] = out_4h["close"].rolling(self.slow_4h).mean()
        out_4h["trend_4h"] = 0
        out_4h.loc[out_4h["ma_fast_4h"] > out_4h["ma_slow_4h"], "trend_4h"] = 1
        out_4h.loc[out_4h["ma_fast_4h"] < out_4h["ma_slow_4h"], "trend_4h"] = -1
        out_4h["adx_4h"] = self._adx(out_4h, self.adx_period_4h)
        out_4h["trend_strength_4h"] = (out_4h["ma_fast_4h"] - out_4h["ma_slow_4h"]).abs() / out_4h["close"]
        out_4h["ma_slow_slope_4h"] = out_4h["ma_slow_4h"].pct_change(self.slow_slope_lookback_4h)
        out["trend_4h"] = out_4h["trend_4h"].reindex(out.index, method="ffill").fillna(0).astype(int)
        out["adx_4h"] = out_4h["adx_4h"].reindex(out.index, method="ffill")
        out["trend_strength_4h"] = out_4h["trend_strength_4h"].reindex(out.index, method="ffill")
        out["ma_slow_slope_4h"] = out_4h["ma_slow_slope_4h"].reindex(out.index, method="ffill")
        out["signal"] = 0
        regime_long = True
        regime_short = True
        if self.use_regime_filter:
            regime_long = (out["adx_4h"] >= self.adx_threshold_4h) & (out["trend_strength_4h"] >= self.trend_strength_threshold_4h) & (out["ma_slow_slope_4h"] > 0)
            regime_short = (out["adx_4h"] >= self.adx_threshold_4h) & (out["trend_strength_4h"] >= self.trend_strength_threshold_4h) & (out["ma_slow_slope_4h"] < 0)
        long_cond = (out["ma_fast"] > out["ma_slow"]) & (out["trend_4h"] == 1) & (out["atr_pct"] > self.atr_pct_threshold) & (out["close"] > out["resistance_7d"]) & regime_long
        short_cond = (out["ma_fast"] < out["ma_slow"]) & (out["trend_4h"] == -1) & (out["atr_pct"] > self.atr_pct_threshold) & (out["close"] < out["support_7d"]) & regime_short
        out.loc[long_cond, "signal"] = 1
        out.loc[short_cond, "signal"] = -1
        out["trade_signal"] = out["signal"].diff().fillna(0)
        need_cols = ["atr", "resistance_7d", "support_7d", "trend_4h", "adx_4h", "trend_strength_4h"]
        return out.dropna(subset=[c for c in need_cols if c in out.columns])


class BTCPerpPullbackStrategy1H:
    """
    主链路：breakout -> pullback -> rejection -> entry
    新增第二套 long-only continuation setup（不恢复 short）：
    - 趋势已确立
    - 价格在 EMA20 上方整理后重新发力
    - 用于补充高质量 continuation 机会
    """

    def __init__(
        self,
        fast_4h=5,
        slow_4h=15,
        adx_period_4h=14,
        adx_threshold_4h=30,
        trend_strength_threshold_4h=0.006,
        breakout_window=24 * 7,
        breakout_confirm_atr=0.20,
        breakout_body_atr=0.35,
        breakout_valid_bars=8,
        pullback_bars=3,
        pullback_max_depth_atr=0.45,
        first_pullback_only=True,
        max_pullbacks_long=1,
        max_pullbacks_short=1,
        min_breakout_age_long=1,
        min_breakout_age_short=1,
        rejection_wick_ratio_long=0.8,
        rejection_wick_ratio_short=0.8,
        allow_short=True,
        allow_same_bar_entry=False,
        atr_period=14,
        atr_pct_low=0.0035,
        atr_pct_high=0.013,
        ema_entry=20,
        # 第二套 continuation setup
        enable_continuation_long=True,
        continuation_window=6,
        continuation_ema_buffer_atr=0.35,
        continuation_body_atr=0.25,
        continuation_cooldown_bars=3,
    ):
        self.fast_4h = fast_4h
        self.slow_4h = slow_4h
        self.adx_period_4h = adx_period_4h
        self.adx_threshold_4h = adx_threshold_4h
        self.trend_strength_threshold_4h = trend_strength_threshold_4h
        self.breakout_window = breakout_window
        self.breakout_confirm_atr = breakout_confirm_atr
        self.breakout_body_atr = breakout_body_atr
        self.breakout_valid_bars = breakout_valid_bars
        self.pullback_bars = pullback_bars
        self.pullback_max_depth_atr = pullback_max_depth_atr
        self.first_pullback_only = first_pullback_only
        self.max_pullbacks_long = max_pullbacks_long
        self.max_pullbacks_short = max_pullbacks_short
        self.min_breakout_age_long = min_breakout_age_long
        self.min_breakout_age_short = min_breakout_age_short
        self.rejection_wick_ratio_long = rejection_wick_ratio_long
        self.rejection_wick_ratio_short = rejection_wick_ratio_short
        self.allow_short = allow_short
        self.allow_same_bar_entry = allow_same_bar_entry
        self.atr_period = atr_period
        self.atr_pct_low = atr_pct_low
        self.atr_pct_high = atr_pct_high
        self.ema_entry = ema_entry
        self.enable_continuation_long = enable_continuation_long
        self.continuation_window = continuation_window
        self.continuation_ema_buffer_atr = continuation_ema_buffer_atr
        self.continuation_body_atr = continuation_body_atr
        self.continuation_cooldown_bars = continuation_cooldown_bars

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
        if self.allow_short:
            out.loc[(out["ma_fast_4h"] < out["ma_slow_4h"]) & out["regime_ok"], "state_signal"] = -1

        out["resistance_7d"] = out["high"].rolling(self.breakout_window, min_periods=self.breakout_window).quantile(0.975)
        out["support_7d"] = out["low"].rolling(self.breakout_window, min_periods=self.breakout_window).quantile(0.025)

        body = (out["close"] - out["open"]).abs()
        long_break_raw = out["close"] > (out["resistance_7d"] + self.breakout_confirm_atr * out["atr"])
        short_break_raw = out["close"] < (out["support_7d"] - self.breakout_confirm_atr * out["atr"])
        out["breakout_quality_long"] = long_break_raw & (body >= self.breakout_body_atr * out["atr"]) & (out["close"] > out["open"])
        out["breakout_quality_short"] = short_break_raw & (body >= self.breakout_body_atr * out["atr"]) & (out["close"] < out["open"])

        lower_wick = (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0)
        upper_wick = (out["high"] - out[["open", "close"]].max(axis=1)).clip(lower=0)
        out["reject_long"] = (out["close"] > out["open"]) & (lower_wick >= body * self.rejection_wick_ratio_long)
        out["reject_short"] = (out["close"] < out["open"]) & (upper_wick >= body * self.rejection_wick_ratio_short)
        out["rejection_type_long"] = np.where(out["reject_long"], "wick_reject_long", "none")
        out["rejection_type_short"] = np.where(out["reject_short"], "wick_reject_short", "none")

        idx = list(out.index)
        n = len(out)
        entry_setup = np.zeros(n, dtype=int)
        entry_reason = np.array(["none"] * n, dtype=object)

        b_event_l = np.zeros(n, dtype=bool)
        b_event_s = np.zeros(n, dtype=bool)
        b_level_l = np.full(n, np.nan)
        b_level_s = np.full(n, np.nan)
        b_age_l = np.full(n, np.nan)
        b_age_s = np.full(n, np.nan)
        p_touch_l = np.zeros(n, dtype=bool)
        p_touch_s = np.zeros(n, dtype=bool)
        p_depth_l = np.full(n, np.nan)
        p_depth_s = np.full(n, np.nan)
        p_depth_ok_l = np.zeros(n, dtype=bool)
        p_depth_ok_s = np.zeros(n, dtype=bool)
        first_pb_l = np.zeros(n, dtype=bool)
        first_pb_s = np.zeros(n, dtype=bool)
        b_time_l = [None] * n
        b_time_s = [None] * n

        last_l_idx = None; last_l_lvl = np.nan; l_touch_count = 0
        last_s_idx = None; last_s_lvl = np.nan; s_touch_count = 0
        min_age = 0 if self.allow_same_bar_entry else 1

        for i in range(n):
            if bool(out.iloc[i]["breakout_quality_long"]) and int(out.iloc[i]["state_signal"]) == 1:
                last_l_idx = i
                last_l_lvl = float(out.iloc[i]["resistance_7d"])
                l_touch_count = 0
                b_event_l[i] = True

            if bool(out.iloc[i]["breakout_quality_short"]) and int(out.iloc[i]["state_signal"]) == -1:
                last_s_idx = i
                last_s_lvl = float(out.iloc[i]["support_7d"])
                s_touch_count = 0
                b_event_s[i] = True

            if last_l_idx is not None and np.isfinite(last_l_lvl):
                age = i - last_l_idx
                b_age_l[i] = age
                b_level_l[i] = last_l_lvl
                b_time_l[i] = str(idx[last_l_idx])
                if age <= self.breakout_valid_bars:
                    atr = float(out.iloc[i]["atr"]) if pd.notna(out.iloc[i]["atr"]) else np.nan
                    low = float(out.iloc[i]["low"])
                    ema = float(out.iloc[i]["ema_entry"])
                    touch = (low <= last_l_lvl) or (low <= ema)
                    p_touch_l[i] = touch
                    if np.isfinite(atr) and atr > 0:
                        depth = max(0.0, (last_l_lvl - low) / atr)
                        p_depth_l[i] = depth
                        p_depth_ok_l[i] = depth <= self.pullback_max_depth_atr
                    if touch:
                        l_touch_count += 1
                        first_pb_l[i] = (l_touch_count == 1)
                    max_pb_long = 1 if self.first_pullback_only else self.max_pullbacks_long
                    first_ok = l_touch_count <= max_pb_long
                    reject_ok = bool(out.iloc[i]["reject_long"])
                    state_ok = int(out.iloc[i]["state_signal"]) == 1
                    if (age >= max(min_age, self.min_breakout_age_long)) and touch and p_depth_ok_l[i] and first_ok and reject_ok and state_ok:
                        entry_setup[i] = 1
                        entry_reason[i] = "breakout_pullback_rejection"

            if last_s_idx is not None and np.isfinite(last_s_lvl):
                age = i - last_s_idx
                b_age_s[i] = age
                b_level_s[i] = last_s_lvl
                b_time_s[i] = str(idx[last_s_idx])
                if age <= self.breakout_valid_bars:
                    atr = float(out.iloc[i]["atr"]) if pd.notna(out.iloc[i]["atr"]) else np.nan
                    high = float(out.iloc[i]["high"])
                    ema = float(out.iloc[i]["ema_entry"])
                    touch = (high >= last_s_lvl) or (high >= ema)
                    p_touch_s[i] = touch
                    if np.isfinite(atr) and atr > 0:
                        depth = max(0.0, (high - last_s_lvl) / atr)
                        p_depth_s[i] = depth
                        p_depth_ok_s[i] = depth <= self.pullback_max_depth_atr
                    if touch:
                        s_touch_count += 1
                        first_pb_s[i] = (s_touch_count == 1)
                    max_pb_short = 1 if self.first_pullback_only else self.max_pullbacks_short
                    first_ok = s_touch_count <= max_pb_short
                    reject_ok = bool(out.iloc[i]["reject_short"])
                    state_ok = int(out.iloc[i]["state_signal"]) == -1
                    if (age >= max(min_age, self.min_breakout_age_short)) and touch and p_depth_ok_s[i] and first_ok and reject_ok and state_ok:
                        entry_setup[i] = -1
                        entry_reason[i] = "breakout_pullback_rejection"

        out["breakout_event_long"] = b_event_l
        out["breakout_event_short"] = b_event_s
        out["breakout_level_long"] = b_level_l
        out["breakout_level_short"] = b_level_s
        out["bars_since_breakout_long"] = b_age_l
        out["bars_since_breakout_short"] = b_age_s
        out["breakout_bar_time_long"] = b_time_l
        out["breakout_bar_time_short"] = b_time_s
        out["pullback_touch_long"] = p_touch_l
        out["pullback_touch_short"] = p_touch_s
        out["pullback_depth_long"] = p_depth_l
        out["pullback_depth_short"] = p_depth_s
        out["pullback_depth_ok_long"] = p_depth_ok_l
        out["pullback_depth_ok_short"] = p_depth_ok_s
        out["first_pullback_ok_long"] = first_pb_l
        out["first_pullback_ok_short"] = first_pb_s

        # 第二套 breakout family setup：突破后的二次延续再突破
        # 不做 EMA reclaim，而是要求：
        # 1) 已发生有效 breakout
        # 2) 价格仍站在 breakout level/EMA 上方
        # 3) 完成一次健康浅回踩
        # 4) 再次突破近 continuation_window 根高点
        out["continuation_setup_long"] = False
        if self.enable_continuation_long:
            rolling_max_prev = out["high"].rolling(self.continuation_window).max().shift(1)
            ema_buf = self.continuation_ema_buffer_atr * out["atr"]
            healthy_hold = (out["low"] >= (out["ema_entry"] - ema_buf)) & (out["close"] >= out["ema_entry"])
            above_break_level = out["close"] >= (out["breakout_level_long"].ffill())
            continuation_break = out["close"] > rolling_max_prev
            continuation_body_ok = body >= self.continuation_body_atr * out["atr"]
            recent_shallow_pullback = out["pullback_touch_long"].shift(1).rolling(self.continuation_cooldown_bars).max().fillna(0).astype(bool) & (out["pullback_depth_long"].shift(1).fillna(np.inf) <= self.pullback_max_depth_atr)
            continuation_state_ok = (out["state_signal"] == 1) & out["regime_ok"]
            cont_setup = continuation_state_ok & healthy_hold & above_break_level & recent_shallow_pullback & continuation_break & continuation_body_ok
            out["continuation_setup_long"] = cont_setup
            fill_mask = (entry_setup == 0) & cont_setup.fillna(False).to_numpy()
            entry_setup[fill_mask] = 1
            entry_reason[fill_mask] = "breakout_rebreak_continuation"

        out["entry_setup"] = entry_setup
        out["entry_reason"] = entry_reason
        out["signal"] = out["state_signal"]
        out["trade_signal"] = out["entry_setup"]

        print("state long true:", float((out["state_signal"] == 1).mean()))
        print("state short true:", float((out["state_signal"] == -1).mean()))
        print("entry long setup:", float((out["entry_setup"] == 1).mean()))
        print("entry short setup:", float((out["entry_setup"] == -1).mean()))

        need_cols = ["atr", "resistance_7d", "support_7d", "ma_fast_4h", "ma_slow_4h", "adx_4h", "trend_strength_4h"]
        return out.dropna(subset=[c for c in need_cols if c in out.columns])


class SOLMeanReversionStrategy1H:
    """
    SOL mean reversion 主线：long-only，先验证 gross edge。
    方向：更早参与过度下杀后的回归，并支持 mean-target exit / time stop。
    """

    def __init__(
        self,
        atr_period=14,
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=30,
        reclaim_ema=20,
        reclaim_confirm_atr=0.05,
        adx_period_4h=14,
        adx_cap_4h=32,
        atr_pct_low=0.004,
        atr_pct_high=0.05,
        require_break_prev_high=True,
        require_bullish=True,
        reclaim_mode="ema",
        oversold_lookback=3,
        use_mean_targets=False,
        mean_target="bb_mid",
        second_target="bb_upper",
        time_stop_bars=0,
    ):
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.reclaim_ema = reclaim_ema
        self.reclaim_confirm_atr = reclaim_confirm_atr
        self.adx_period_4h = adx_period_4h
        self.adx_cap_4h = adx_cap_4h
        self.atr_pct_low = atr_pct_low
        self.atr_pct_high = atr_pct_high
        self.require_break_prev_high = require_break_prev_high
        self.require_bullish = require_bullish
        self.reclaim_mode = reclaim_mode
        self.oversold_lookback = oversold_lookback
        self.use_mean_targets = use_mean_targets
        self.mean_target = mean_target
        self.second_target = second_target
        self.time_stop_bars = time_stop_bars

    @staticmethod
    def _atr(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _rsi(close, period=14):
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

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

    def generate_signals(self, df):
        out = df.copy()
        out["atr"] = self._atr(out, self.atr_period)
        out["atr_pct"] = out["atr"] / out["close"]
        out["volatility"] = out["atr_pct"]
        out["ema20"] = out["close"].ewm(span=self.reclaim_ema, adjust=False).mean()
        out["rsi"] = self._rsi(out["close"], self.rsi_period)
        out["bb_mid"] = out["close"].rolling(self.bb_period).mean()
        out["bb_sigma"] = out["close"].rolling(self.bb_period).std()
        out["bb_lower"] = out["bb_mid"] - self.bb_std * out["bb_sigma"]
        out["bb_upper"] = out["bb_mid"] + self.bb_std * out["bb_sigma"]

        out_4h = out[["open", "high", "low", "close", "volume"]].resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        out_4h["adx_4h"] = self._adx(out_4h, self.adx_period_4h)
        out["adx_4h"] = out_4h["adx_4h"].reindex(out.index, method="ffill")

        regime_ok = (
            (out["adx_4h"] <= self.adx_cap_4h) &
            (out["atr_pct"] >= self.atr_pct_low) &
            (out["atr_pct"] <= self.atr_pct_high)
        )

        oversold = (out["close"] < out["bb_lower"]) & (out["rsi"] <= self.rsi_oversold)
        reclaim_ema = out["close"] >= (out["ema20"] + self.reclaim_confirm_atr * out["atr"])
        reclaim_bb = out["close"] >= out["bb_lower"]
        reclaim_prev_close = out["close"] > out["close"].shift(1)
        if self.reclaim_mode == "bb_in":
            reclaim = reclaim_bb
        elif self.reclaim_mode == "prev_close":
            reclaim = reclaim_prev_close
        elif self.reclaim_mode == "any":
            reclaim = reclaim_ema | reclaim_bb | reclaim_prev_close
        else:
            reclaim = reclaim_ema

        bullish = (out["close"] > out["open"]) if self.require_bullish else pd.Series(True, index=out.index)
        break_prev_high = (out["close"] > out["high"].shift(1)) if self.require_break_prev_high else pd.Series(True, index=out.index)
        recent_oversold = oversold.shift(1).rolling(self.oversold_lookback).max().fillna(0).astype(bool)

        out["state_signal"] = np.where(regime_ok, 1, 0)
        out["entry_setup"] = np.where(recent_oversold & reclaim & bullish & break_prev_high & regime_ok, 1, 0)
        out["signal"] = out["state_signal"]
        out["trade_signal"] = out["entry_setup"]
        out["entry_reason"] = np.where(out["entry_setup"] == 1, "sol_mean_reversion_reclaim", "none")
        out["resistance_7d"] = out["high"].rolling(24 * 7, min_periods=24 * 7).max()
        out["support_7d"] = out["low"].rolling(24 * 7, min_periods=24 * 7).min()
        out["zscore"] = (out["close"] - out["bb_mid"]) / out["bb_sigma"].replace(0, np.nan)

        if self.use_mean_targets:
            out["take_price_signal_long"] = np.where(self.mean_target == "ema20", out["ema20"], out["bb_mid"])
            if self.second_target == "zscore0":
                out["take_price_signal2_long"] = out["bb_mid"]
            elif self.second_target == "ema20":
                out["take_price_signal2_long"] = out["ema20"]
            else:
                out["take_price_signal2_long"] = out["bb_upper"]
        else:
            out["take_price_signal_long"] = np.nan
            out["take_price_signal2_long"] = np.nan

        need_cols = ["atr", "ema20", "bb_lower", "adx_4h"]
        return out.dropna(subset=[c for c in need_cols if c in out.columns])
