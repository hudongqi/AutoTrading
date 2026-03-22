#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np

from data import CCXTDataSource
from route1_sol_vol_breakout_research import SOLRoute1VolBreakout, run_variant, SYMBOL, START, END


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    body_frac_mins = [0.50, 0.55, 0.60]
    close_near_high_maxs = [0.20, 0.25, 0.30]
    breakout_buffer_atrs = [0.05, 0.10, 0.15]
    exit_profiles = [
        {
            'name': 'RUNNER',
            'take_R': 8.0,
            'partial_take_R': 0.0,
            'partial_take_frac': 0.0,
            'break_even_after_partial': False,
            'break_even_R': 0.0,
        },
        {
            'name': 'P25_1p5R',
            'take_R': 8.0,
            'partial_take_R': 1.5,
            'partial_take_frac': 0.25,
            'break_even_after_partial': True,
            'break_even_R': 0.8,
        },
        {
            'name': 'P33_2R',
            'take_R': 8.0,
            'partial_take_R': 2.0,
            'partial_take_frac': 0.33,
            'break_even_after_partial': True,
            'break_even_R': 1.0,
        },
    ]

    rows = []
    for body_frac_min, close_near_high_max, breakout_buffer_atr, exit_profile in itertools.product(
        body_frac_mins, close_near_high_maxs, breakout_buffer_atrs, exit_profiles
    ):
        name = (
            f"QX_B{body_frac_min:.2f}_C{close_near_high_max:.2f}_BUF{breakout_buffer_atr:.2f}_"
            f"{exit_profile['name']}"
        )
        strat = SOLRoute1VolBreakout(
            name=name,
            breakout_lookback=15,
            squeeze_bw_q=0.20,
            squeeze_atr_q=0.30,
            range_pos_threshold=0.65,
            breakout_buffer_atr=breakout_buffer_atr,
            body_frac_min=body_frac_min,
            close_near_high_max=close_near_high_max,
            range_expand_mult=1.2,
            volume_mult=1.40,
            require_volume=False,
            entry_mode='chase',
            squeeze_recent_bars=12,
        )
        cfg = dict(
            leverage=4.0,
            max_pos=2.0,
            risk_per_trade=0.03,
            cooldown_bars=1,
            stop_atr=2.0,
            take_R=exit_profile['take_R'],
            trail_start_R=1.2,
            trail_atr=2.4,
            partial_take_R=exit_profile['partial_take_R'],
            partial_take_frac=exit_profile['partial_take_frac'],
            break_even_after_partial=exit_profile['break_even_after_partial'],
            break_even_R=exit_profile['break_even_R'],
        )
        row = run_variant(name, strat, df, cfg)
        row.update({
            'body_frac_min': body_frac_min,
            'close_near_high_max': close_near_high_max,
            'breakout_buffer_atr': breakout_buffer_atr,
            'exit_profile': exit_profile['name'],
            'winner_loser_ratio': (row['avg_winner'] / abs(row['avg_loser'])) if pd.notna(row['avg_winner']) and pd.notna(row['avg_loser']) and row['avg_loser'] < 0 else np.nan,
            'return_over_dd': (row['return'] / abs(row['max_drawdown'])) if row['max_drawdown'] < 0 else np.nan,
        })
        rows.append(row)

    rep = pd.DataFrame(rows)
    rep = rep.sort_values(
        ['net_closed_pnl', 'profit_factor', 'remove_best_trade_net', 'max_drawdown'],
        ascending=[False, False, False, True],
    )

    top = rep.head(20).copy()
    print('=== TOP 20 ROUTE1_A QUALITY/EXIT SWEEP ===')
    print(top.to_string(index=False, formatters={
        'return': lambda x: f'{x:.2%}',
        'max_drawdown': lambda x: f'{x:.2%}',
        'gross_closed_pnl': lambda x: f'{x:.2f}',
        'net_closed_pnl': lambda x: f'{x:.2f}',
        'profit_factor': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'avg_winner': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'avg_loser': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'winner_loser_ratio': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'expectancy_per_trade': lambda x: f'{x:.2f}',
        'avg_R_realized': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'remove_best_trade_net': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'mfe_capture_ratio': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'return_over_dd': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
    }))

    csv_path = 'route1_a_quality_exit_sweep_results.csv'
    rep.to_csv(csv_path, index=False)
    print(f'\nSaved full results to {csv_path}')


if __name__ == '__main__':
    main()
