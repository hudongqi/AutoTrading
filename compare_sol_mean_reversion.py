#!/usr/bin/env python3
import pandas as pd
import numpy as np

from data import CCXTDataSource
from strategy import SOLMeanReversionStrategy1H
from strategy_profiles import get_strategy_profile, get_sol_backtest_profile
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester
from config import INITIAL_CASH, TAKER_FEE_RATE, MAKER_FEE_RATE, FUNDING_RATE_PER_8H, SLIPPAGE_BPS

SYMBOL = 'SOL/USDT:USDT'
START = '2025-01-01'
END = '2026-03-19'
VERSIONS = [
    ('SOL_MEAN_REV_V1', 'sol_mean_rev_v1'),
    ('SOL_MR_V2_FASTER_ENTRY', 'sol_mr_v2_faster_entry'),
    ('SOL_MR_V3_MEAN_TARGET_EXIT', 'sol_mr_v3_mean_target_exit'),
]


def run_variant(label: str, profile_name: str):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    strat = SOLMeanReversionStrategy1H(**get_strategy_profile(profile_name))
    bt_cfg = get_sol_backtest_profile(profile_name)
    sig = strat.generate_signals(df)
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=PerpPortfolio(INITIAL_CASH, leverage=bt_cfg['leverage'], taker_fee_rate=TAKER_FEE_RATE, maker_fee_rate=MAKER_FEE_RATE, maint_margin_rate=0.005),
        strategy=strat,
        max_pos=bt_cfg['max_pos'],
        cooldown_bars=bt_cfg['cooldown_bars'],
        stop_atr=bt_cfg['stop_atr'],
        take_R=bt_cfg['take_R'],
        trail_start_R=bt_cfg['trail_start_R'],
        trail_atr=bt_cfg['trail_atr'],
        use_trailing=bt_cfg.get('use_trailing', True),
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=bt_cfg['risk_per_trade'],
        enable_risk_position_sizing=True,
        allow_reentry=True,
        partial_take_R=bt_cfg.get('partial_take_R', 0.0),
        partial_take_frac=bt_cfg.get('partial_take_frac', 0.0),
        break_even_after_partial=bt_cfg.get('break_even_after_partial', False),
        break_even_R=bt_cfg.get('break_even_R', 0.0),
        use_signal_exit_targets=bt_cfg.get('use_signal_exit_targets', False),
        max_hold_bars=bt_cfg.get('max_hold_bars', 0),
    )
    out = bt.run(sig)
    st = out.attrs.get('stats', {})
    trades = pd.DataFrame(out.attrs.get('closed_trades', []))
    eq = out['equity']
    dd = (eq / eq.cummax() - 1).min()
    remove_best = None
    avg_net = 0.0
    if not trades.empty and 'realized_net' in trades.columns:
        trades = trades.copy()
        trades['realized_net'] = trades['realized_net'].astype(float)
        avg_net = float(trades['realized_net'].mean())
        if len(trades) > 1:
            remove_best = float(trades.drop(trades['realized_net'].idxmax())['realized_net'].sum())
    fees = float(st.get('total_fees', 0.0))
    gross = float(st.get('gross_closed_pnl', 0.0))
    return {
        'variant': label,
        'return': float(eq.iloc[-1] / INITIAL_CASH - 1),
        'max_drawdown': float(dd),
        'trade_count': int(st.get('closed_trade_count', 0)),
        'fees': fees,
        'gross_closed_pnl': gross,
        'net_closed_pnl': float(st.get('net_closed_pnl', 0.0)),
        'gross_fee_ratio': (gross / fees) if fees > 0 else np.nan,
        'profit_factor': st.get('profit_factor', float('nan')),
        'expectancy_per_trade': float(st.get('expectancy_per_trade', 0.0)),
        'avg_net_pnl_per_trade': avg_net,
        'remove_best_trade_net': remove_best,
        'exit_reason_split': st.get('exit_reason_split', {}),
    }


def main():
    rows = [run_variant(label, profile) for label, profile in VERSIONS]
    rep = pd.DataFrame(rows).sort_values('net_closed_pnl', ascending=False)
    print('\n=== SOL MEAN REVERSION COMPARISON ===')
    print(rep.drop(columns=['exit_reason_split']).to_string(index=False, formatters={
        'return': lambda x: f'{x:.2%}',
        'max_drawdown': lambda x: f'{x:.2%}',
        'fees': lambda x: f'{x:.2f}',
        'gross_closed_pnl': lambda x: f'{x:.2f}',
        'net_closed_pnl': lambda x: f'{x:.2f}',
        'gross_fee_ratio': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'profit_factor': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'expectancy_per_trade': lambda x: f'{x:.2f}',
        'avg_net_pnl_per_trade': lambda x: f'{x:.2f}',
        'remove_best_trade_net': lambda x: f'{x:.2f}' if x is not None and pd.notna(x) else 'N/A',
    }))
    print('\n=== EXIT REASONS ===')
    for row in rows:
        print(row['variant'], row['exit_reason_split'])


if __name__ == '__main__':
    main()
