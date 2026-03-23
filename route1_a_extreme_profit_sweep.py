#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np

from data import CCXTDataSource
from route1_sol_vol_breakout_research import SOLRoute1VolBreakout, run_variant, SYMBOL, START, END


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    take_Rs = [18.0, 20.0, 24.0]
    breakout_lookbacks = [18, 21, 24]
    stop_atrs = [2.0, 2.2, 2.4]
    range_expand_mults = [1.2, 1.3]

    rows = []
    for take_R, breakout_lookback, stop_atr, range_expand_mult in itertools.product(
        take_Rs, breakout_lookbacks, stop_atrs, range_expand_mults
    ):
        name = f"XP_TK{take_R:.0f}_BL{breakout_lookback}_ST{stop_atr:.1f}_RE{range_expand_mult:.1f}"
        strat = SOLRoute1VolBreakout(
            name=name,
            breakout_lookback=breakout_lookback,
            squeeze_bw_q=0.20,
            squeeze_atr_q=0.30,
            range_pos_threshold=0.55,
            breakout_buffer_atr=0.10,
            body_frac_min=0.60,
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
            take_R=take_R,
            trail_start_R=1.2,
            trail_atr=2.4,
            partial_take_R=0.0,
            partial_take_frac=0.0,
            break_even_after_partial=False,
            break_even_R=0.0,
        )
        row = run_variant(name, strat, df, cfg)
        row.update({
            'take_R': take_R,
            'breakout_lookback': breakout_lookback,
            'stop_atr': stop_atr,
            'range_expand_mult': range_expand_mult,
            'winner_loser_ratio': (row['avg_winner'] / abs(row['avg_loser'])) if pd.notna(row['avg_winner']) and pd.notna(row['avg_loser']) and row['avg_loser'] < 0 else np.nan,
            'return_over_dd': (row['return'] / abs(row['max_drawdown'])) if row['max_drawdown'] < 0 else np.nan,
        })
        rows.append(row)

    rep = pd.DataFrame(rows)
    rep = rep.sort_values(
        ['net_closed_pnl', 'gross_closed_pnl', 'profit_factor', 'remove_best_trade_net'],
        ascending=[False, False, False, False],
    )

    top = rep.head(25).copy()
    print('=== TOP 25 ROUTE1_A EXTREME PROFIT SWEEP ===')
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

    csv_path = 'route1_a_extreme_profit_sweep_results.csv'
    rep.to_csv(csv_path, index=False)
    print(f'\nSaved full results to {csv_path}')


if __name__ == '__main__':
    main()
