#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np

from data import CCXTDataSource
from route1_sol_vol_breakout_research import SOLRoute1VolBreakout, run_variant, SYMBOL, START, END


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    squeeze_bw_qs = [0.20, 0.25, 0.30]
    squeeze_atr_qs = [0.30, 0.35, 0.40]
    range_pos_thresholds = [0.55, 0.60, 0.65]
    take_Rs = [8.0, 10.0, 12.0]
    trail_start_Rs = [1.2, 1.6, 2.0]

    rows = []
    for squeeze_bw_q, squeeze_atr_q, range_pos_threshold, take_R, trail_start_R in itertools.product(
        squeeze_bw_qs, squeeze_atr_qs, range_pos_thresholds, take_Rs, trail_start_Rs
    ):
        name = (
            f"PP_BW{squeeze_bw_q:.2f}_AQ{squeeze_atr_q:.2f}_RP{range_pos_threshold:.2f}_"
            f"TK{take_R:.0f}_TS{trail_start_R:.1f}"
        )
        strat = SOLRoute1VolBreakout(
            name=name,
            breakout_lookback=15,
            squeeze_bw_q=squeeze_bw_q,
            squeeze_atr_q=squeeze_atr_q,
            range_pos_threshold=range_pos_threshold,
            breakout_buffer_atr=0.10,
            body_frac_min=0.60,
            close_near_high_max=0.25,
            range_expand_mult=1.20,
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
            take_R=take_R,
            trail_start_R=trail_start_R,
            trail_atr=2.4,
            partial_take_R=0.0,
            partial_take_frac=0.0,
            break_even_after_partial=False,
            break_even_R=0.0,
        )
        row = run_variant(name, strat, df, cfg)
        row.update({
            'squeeze_bw_q': squeeze_bw_q,
            'squeeze_atr_q': squeeze_atr_q,
            'range_pos_threshold': range_pos_threshold,
            'take_R': take_R,
            'trail_start_R': trail_start_R,
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
    print('=== TOP 25 ROUTE1_A PROFIT PUSH SWEEP ===')
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

    csv_path = 'route1_a_profit_push_sweep_results.csv'
    rep.to_csv(csv_path, index=False)
    print(f'\nSaved full results to {csv_path}')


if __name__ == '__main__':
    main()
