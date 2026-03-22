#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np

from data import CCXTDataSource
from route1_sol_vol_breakout_research import SOLRoute1VolBreakout, run_variant, SYMBOL, START, END


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    breakout_lookbacks = [15, 20, 25]
    range_expand_mults = [1.2, 1.3, 1.4]
    stop_atrs = [1.6, 1.8, 2.0]
    trail_atrs = [2.0, 2.4, 2.8]

    rows = []
    for breakout_lookback, range_expand_mult, stop_atr, trail_atr in itertools.product(
        breakout_lookbacks, range_expand_mults, stop_atrs, trail_atrs
    ):
        name = f"A_L{breakout_lookback}_RE{range_expand_mult:.1f}_S{stop_atr:.1f}_T{trail_atr:.1f}"
        strat = SOLRoute1VolBreakout(
            name=name,
            breakout_lookback=breakout_lookback,
            squeeze_bw_q=0.20,
            squeeze_atr_q=0.30,
            range_pos_threshold=0.65,
            breakout_buffer_atr=0.10,
            body_frac_min=0.55,
            close_near_high_max=0.25,
            range_expand_mult=range_expand_mult,
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
            stop_atr=stop_atr,
            take_R=8.0,
            trail_start_R=1.2,
            trail_atr=trail_atr,
            partial_take_R=0.0,
            partial_take_frac=0.0,
            break_even_after_partial=False,
            break_even_R=0.0,
        )
        row = run_variant(name, strat, df, cfg)
        row.update({
            'breakout_lookback': breakout_lookback,
            'range_expand_mult': range_expand_mult,
            'stop_atr': stop_atr,
            'trail_atr': trail_atr,
            'winner_loser_ratio': (row['avg_winner'] / abs(row['avg_loser'])) if pd.notna(row['avg_winner']) and pd.notna(row['avg_loser']) and row['avg_loser'] < 0 else np.nan,
            'return_over_dd': (row['return'] / abs(row['max_drawdown'])) if row['max_drawdown'] < 0 else np.nan,
        })
        rows.append(row)

    rep = pd.DataFrame(rows)
    rep = rep.sort_values(
        ['net_closed_pnl', 'profit_factor', 'remove_best_trade_net', 'max_drawdown'],
        ascending=[False, False, False, True],
    )

    top = rep.head(15).copy()
    print('=== TOP 15 ROUTE1_A_CHASE SWEEP ===')
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

    csv_path = 'route1_a_chase_sweep_results.csv'
    rep.to_csv(csv_path, index=False)
    print(f'\nSaved full results to {csv_path}')


if __name__ == '__main__':
    main()
