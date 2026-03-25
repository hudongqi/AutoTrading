from copy import deepcopy

COMMON_V6 = {
    "adx_threshold_4h": 28,
    "trend_strength_threshold_4h": 0.0055,
    "breakout_confirm_atr": 0.12,
    "breakout_body_atr": 0.20,
    "pullback_bars": 4,
    "first_pullback_only": False,
    "max_pullbacks_long": 3,
    "max_pullbacks_short": 1,
    "rejection_wick_ratio_short": 0.80,
    "allow_short": False,
    "allow_same_bar_entry": False,
    "breakout_valid_bars": 12,
    "atr_pct_low": 0.0030,
    "atr_pct_high": 0.016,
    "enable_continuation_long": False,
}

SOL_MEAN_REVERSION_PROFILES = {
    "sol_mean_rev_v1": {
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "reclaim_ema": 20,
        "reclaim_confirm_atr": 0.05,
        "adx_period_4h": 14,
        "adx_cap_4h": 32,
        "atr_pct_low": 0.004,
        "atr_pct_high": 0.05,
        "require_break_prev_high": True,
        "require_bullish": True,
        "reclaim_mode": "ema",
        "oversold_lookback": 3,
        "use_mean_targets": False,
        "mean_target": "bb_mid",
        "second_target": "bb_upper",
        "time_stop_bars": 0,
    },
    "sol_mr_v2_faster_entry": {
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 32,
        "reclaim_ema": 20,
        "reclaim_confirm_atr": 0.0,
        "adx_period_4h": 14,
        "adx_cap_4h": 34,
        "atr_pct_low": 0.004,
        "atr_pct_high": 0.05,
        "require_break_prev_high": False,
        "require_bullish": False,
        "reclaim_mode": "any",
        "oversold_lookback": 4,
        "use_mean_targets": False,
        "mean_target": "bb_mid",
        "second_target": "bb_upper",
        "time_stop_bars": 0,
    },
    "sol_mr_v3_mean_target_exit": {
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 32,
        "reclaim_ema": 20,
        "reclaim_confirm_atr": 0.0,
        "adx_period_4h": 14,
        "adx_cap_4h": 34,
        "atr_pct_low": 0.004,
        "atr_pct_high": 0.05,
        "require_break_prev_high": False,
        "require_bullish": False,
        "reclaim_mode": "any",
        "oversold_lookback": 4,
        "use_mean_targets": True,
        "mean_target": "bb_mid",
        "second_target": "bb_upper",
        "time_stop_bars": 10,
    },
    # ── V2：趋势对齐双向均值回归 ──────────────────────────────
    "sol_rev_v2": {
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "reclaim_ema": 20,
        "trend_ema": 200,
        "vol_period": 20,
        "vol_spike_mult": 1.2,
        "atr_pct_low": 0.003,
        "atr_pct_high": 0.10,
        "oversold_lookback": 4,
        "allow_short": True,
    },
}

STRATEGY_PROFILES = {
    "v6_relaxed": {
        **COMMON_V6,
        "pullback_max_depth_atr": 0.90,
        "min_breakout_age_long": 1,
        "rejection_wick_ratio_long": 0.40,
    },
    "v6_tuned": {
        **COMMON_V6,
        "pullback_max_depth_atr": 0.40,
        "min_breakout_age_long": 3,
        "rejection_wick_ratio_long": 0.65,
    },
    "v6_1_default": {
        **COMMON_V6,
        "pullback_max_depth_atr": 0.40,
        "min_breakout_age_long": 2,
        "rejection_wick_ratio_long": 0.65,
    },
    "v6_2_sample_up": {
        **COMMON_V6,
        "pullback_max_depth_atr": 0.40,
        "min_breakout_age_long": 1,
        "rejection_wick_ratio_long": 0.65,
    },
    "v7_dual_long": {
        **COMMON_V6,
        "pullback_max_depth_atr": 0.40,
        "min_breakout_age_long": 1,
        "rejection_wick_ratio_long": 0.65,
        "enable_continuation_long": True,
        "continuation_window": 6,
        "continuation_ema_buffer_atr": 0.35,
        "continuation_body_atr": 0.22,
        "continuation_cooldown_bars": 3,
    },
}

BACKTEST_COMMON = {
    "entry_is_maker": False,
    "leverage": 2.0,
    "max_pos": 0.8,
    "cooldown_bars": 3,
    "stop_atr": 1.4,
    "take_R": 2.6,
    "trail_start_R": 1.0,
    "trail_atr": 2.2,
    "risk_per_trade": 0.0075,
    "enable_risk_position_sizing": True,
    "allow_reentry": True,
    "partial_take_R": 0.0,
    "partial_take_frac": 0.0,
}

SOL_BACKTEST_PROFILES = {
    "sol_mean_rev_v1": {
        "leverage": 3.0,
        "max_pos": 1.2,
        "risk_per_trade": 0.015,
        "cooldown_bars": 2,
        "stop_atr": 1.2,
        "take_R": 1.8,
        "trail_start_R": 0.8,
        "trail_atr": 1.6,
        "partial_take_R": 1.0,
        "partial_take_frac": 0.5,
        "break_even_after_partial": True,
        "break_even_R": 0.8,
        "use_trailing": True,
        "use_signal_exit_targets": False,
        "max_hold_bars": 0,
    },
    "sol_mr_v2_faster_entry": {
        "leverage": 3.0,
        "max_pos": 1.2,
        "risk_per_trade": 0.015,
        "cooldown_bars": 1,
        "stop_atr": 1.2,
        "take_R": 1.6,
        "trail_start_R": 0.7,
        "trail_atr": 1.5,
        "partial_take_R": 0.9,
        "partial_take_frac": 0.4,
        "break_even_after_partial": True,
        "break_even_R": 0.7,
        "use_trailing": True,
        "use_signal_exit_targets": False,
        "max_hold_bars": 0,
    },
    "sol_mr_v3_mean_target_exit": {
        "leverage": 3.0,
        "max_pos": 1.2,
        "risk_per_trade": 0.015,
        "cooldown_bars": 1,
        "stop_atr": 1.2,
        "take_R": 4.0,
        "trail_start_R": 9.0,
        "trail_atr": 3.0,
        "partial_take_R": 0.0,
        "partial_take_frac": 0.0,
        "break_even_after_partial": False,
        "break_even_R": 0.0,
        "use_trailing": False,
        "use_signal_exit_targets": True,
        "max_hold_bars": 10,
    },
    # ── SOL V2：宽止损 + 纯 stop/take ───────────────────────
    "sol_rev_v2": {
        "leverage": 2.0,
        "max_pos": 500,
        "risk_per_trade": 0.040,  # 风险 4.0%（DD ≈ 25%，留有安全边际）
        "cooldown_bars": 3,
        "stop_atr": 5.0,          # 宽止损 5 ATR
        "take_R": 3.0,            # 固定止盈 3.0R（优化：2.5→3.0，+6%组合收益）
        "trail_start_R": 0.0,
        "trail_atr": 0.0,
        "use_trailing": False,
        "partial_take_R": 0.0,
        "partial_take_frac": 0.0,
        "break_even_after_partial": False,
        "break_even_R": 0.0,
        "use_signal_exit_targets": False,
        "max_hold_bars": 240,     # 最多 10 天（240H）强制平仓
        "entry_is_maker": False,
        "enable_risk_position_sizing": True,
        "allow_reentry": False,
    },
    # ── PEPE V2：高波动专属参数 ──────────────────────────────
    # 优化结论（参数扫描 2024-2026，两轮迭代）：
    #   vol_spike_mult=1.5（过滤弱量能假突破，胜率 46% → 49%）
    #   take_R=3.0（PEPE 动量持续性强，2.5R 离场太早）
    #   stop_atr=5.0（宽止损绝对最优，4.0/6.0 均差）
    #   cooldown_bars=2（第二轮扫描：减少冷却等待，Calmar 6.65→7.24）
    #   atr_pct_high=0.20（PEPE 波动远高于 SOL，需放宽上限）
    #   atr_pct_low=0.008（策略参数，过滤低波动期假信号，与 cd=2 叠加 Calmar→7.77）
    #   max_pos=5_000_000（单价 ~$0.01，需大量单位才有效仓位）
    # 注：atr_pct_low=0.008 是策略参数，需在调用方的 strat_params 中设置
    "pepe_rev_v2": {
        "leverage": 2.0,
        "max_pos": 5_000_000,     # 1000PEPE 单价极低，需大值
        "risk_per_trade": 0.040,
        "cooldown_bars": 3,       # 第三轮优化：3 比 2 更优（Calmar +0.60）
        "stop_atr": 5.0,
        "take_R": 4.0,            # 比 SOL 高：PEPE 能走更大（优化：3.0→4.0，+28%收益）
        "trail_start_R": 0.0,
        "trail_atr": 0.0,
        "use_trailing": False,
        "partial_take_R": 0.0,
        "partial_take_frac": 0.0,
        "break_even_after_partial": False,
        "break_even_R": 0.0,
        "use_signal_exit_targets": False,
        "max_hold_bars": 240,
        "entry_is_maker": False,
        "enable_risk_position_sizing": True,
        "allow_reentry": False,
    },
}


def get_strategy_profile(name: str):
    if name in STRATEGY_PROFILES:
        return deepcopy(STRATEGY_PROFILES[name])
    if name in SOL_MEAN_REVERSION_PROFILES:
        return deepcopy(SOL_MEAN_REVERSION_PROFILES[name])
    if name in SOL_BACKTEST_PROFILES:          # sol_rev_v2 策略参数也存在此字典
        return deepcopy(SOL_BACKTEST_PROFILES[name])
    raise KeyError(f"Unknown strategy profile: {name}")


def get_sol_backtest_profile(name: str):
    if name not in SOL_BACKTEST_PROFILES:
        raise KeyError(f"Unknown SOL backtest profile: {name}")
    return deepcopy(SOL_BACKTEST_PROFILES[name])


def list_profiles():
    return list(STRATEGY_PROFILES.keys()) + list(SOL_MEAN_REVERSION_PROFILES.keys())
