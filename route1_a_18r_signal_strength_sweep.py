#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np

from data import CCXTDataSource
from route1_sol_vol_breakout_research import SOLRoute1VolBreakout, run_variant, SYMBOL, START, END


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    squeeze_bw_qs = [0.15, 0.20, 0.25]
    range_expand_mults = [1.10, 1.20, 1.30]
    require_volume_opts = [False, True]
    volume_mults = [1.20, 1.40, 1.60]
    volume_combos = []
    for req in require_volume_opts:
        if req:
            for vm in volume_mults:
                volume_combos.append((req, vm))
        else:
            volume_combos.append((req, 1.40))

    rows = []
    for squeeze_bw_q, range_expand_mult, (require_volume, volume_mult) in itertools.product(
        squeeze_bw_qs, range_expand_mults, volume_combos
    ):
        name = f"SS_BW{squeeze_bw_q:.2f}_RE{range_expand_mult:.2f}_{'VOL' if require_volume else 'NOVOL'}_{volume_mult:.1f}"
        strat = SOLRoute1VolBreakout(
            name=name,
            breakout_lookback=18,
            squeeze_bw_q=squeeze_bw_q,
            squeeze_atr_q=0.30,
            range_pos_threshold=0.55,
            breakout_buffer_atr=0.10,
            body_frac_min=0.60,
            close_near_high_max=0.25,
            range_expand_mult=range_expand_mult,
            volume_mult=volume_mult,
            require_volume=require_volume,
            entry_mode='chase',
            squeeze_recent_bars=12,
        )
        cfg = dict(
            leverage=4.0,
            max_pos=2.0,
            risk_per_trade=0.03,
            cooldown_bars=1,
            stop_atr=2.0,
            take_R=18.0,
            trail_start_R=1.2,
            trail_atr=2.4,
            partial_take_R=0.0,
            partial_take_frac=0.0,
            break_even_after_partial=False,
            break_even_R=0.0,
        )
        row = run_variant(name, strat, df, cfg)
        row.update({
            'squeeze_bw_q': squeeze_bw_q,
            'range_expand_mult': range_expand_mult,
            'require_volume': require_volume,
            'volume_mult': volume_mult,
            'winner_loser_ratio': (row['avg_winner'] / abs(row['avg_loser'])) if pd.notna(row['avg_winner']) and pd.notna(row['avg_loser']) and row['avg_loser'] < 0 else np.nan,
            'return_over_dd': (row['return'] / abs(row['max_drawdown'])) if row['max_drawdown'] < 0 else np.nan,
        })
        rows.append(row)

    rep = pd.DataFrame(rows)
    rep = rep.sort_values(
        ['net_closed_pnl', 'profit_factor', 'remove_best_trade_net', 'trade_count', 'max_drawdown'],
        ascending=[False, False, False, False, True],
    )

    top = rep.head(25).copy()
    print('=== TOP 25 ROUTE1_A 18R SIGNAL STRENGTH SWEEP ===')
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

    csv_path = 'route1_a_18r_signal_strength_sweep_results.csv'
    rep.to_csv(csv_path, index=False)
    print(f'\nSaved full results to {csv_path}')


if __name__ == '__main__':
    main()
