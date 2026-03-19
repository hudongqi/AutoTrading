from copy import deepcopy

EXIT_PROFILES = {
    "exit_baseline": {
        "stop_atr": 1.4,
        "take_R": 2.6,
        "trail_start_R": 1.0,
        "trail_atr": 2.2,
        "partial_take_R": 0.0,
        "partial_take_frac": 0.0,
        "break_even_after_partial": False,
        "break_even_R": 0.0,
    },
    "exit_fast_lock": {
        "stop_atr": 1.4,
        "take_R": 2.2,
        "trail_start_R": 0.8,
        "trail_atr": 1.6,
        "partial_take_R": 1.0,
        "partial_take_frac": 0.5,
        "break_even_after_partial": True,
        "break_even_R": 0.8,
    },
    "exit_loose_runner": {
        "stop_atr": 1.4,
        "take_R": 3.4,
        "trail_start_R": 1.6,
        "trail_atr": 2.8,
        "partial_take_R": 2.0,
        "partial_take_frac": 0.33,
        "break_even_after_partial": False,
        "break_even_R": 1.2,
    },
}


def get_exit_profile(name: str):
    if name not in EXIT_PROFILES:
        raise KeyError(f"Unknown exit profile: {name}")
    return deepcopy(EXIT_PROFILES[name])


def list_exit_profiles():
    return list(EXIT_PROFILES.keys())
