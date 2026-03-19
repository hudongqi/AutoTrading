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


def get_strategy_profile(name: str):
    if name not in STRATEGY_PROFILES:
        raise KeyError(f"Unknown strategy profile: {name}")
    return deepcopy(STRATEGY_PROFILES[name])


def list_profiles():
    return list(STRATEGY_PROFILES.keys())
