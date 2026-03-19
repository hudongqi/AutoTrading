#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import numpy as np

from data import CCXTDataSource
from strategy import BTCPerpPullbackStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester
from config import INITIAL_CASH, TAKER_FEE_RATE, MAKER_FEE_RATE, FUNDING_RATE_PER_8H, SLIPPAGE_BPS

SYMBOL = 'SOL/USDT:USDT'
START = '2025-01-01'
END = '2026-03-19'


def load_overlay(event_file='event_signals.json'):
    defaults = {'block_new_entries': False, 'reduce_risk': False, 'risk_mode': 'NORMAL', 'market_bias': 'NEUTRAL', 'leverage_mult': 1.0, 'max_pos_mult': 1.0}
    p = Path(event_file)
    if not p.exists():
        return defaults
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return defaults
    macro = data.get('macro', {})
    geo = data.get('geopolitics', {})
    risk_mode = str(data.get('risk_mode', 'NORMAL')).upper()
    reduce_risk = bool(macro.get('reduce_risk', False) or geo.get('reduce_risk', False) or risk_mode == 'REDUCE_RISK')
    return {
        'block_new_entries': bool(macro.get('block', False) or geo.get('block_new_entries', False)),
        'reduce_risk': reduce_risk,
        'risk_mode': risk_mode,
        'market_bias': str(data.get('market_bias', 'NEUTRAL')).upper(),
        'leverage_mult': 0.6 if reduce_risk else 1.0,
        'max_pos_mult': 0.5 if reduce_risk else 1.0,
    }


def run_variant(name, strat_kwargs, backtest_cfg):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    overlay = load_overlay()
    strat = BTCPerpPullbackStrategy1H(**strat_kwargs)
    sig = strat.generate_signals(df)

    leverage = backtest_cfg['leverage'] * overlay['leverage_mult']
    max_pos = backtest_cfg['max_pos'] * overlay['max_pos_mult']
    if overlay['block_new_entries']:
        max_pos = 0.0

    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=PerpPortfolio(INITIAL_CASH, leverage=leverage, taker_fee_rate=TAKER_FEE_RATE, maker_fee_rate=MAKER_FEE_RATE, maint_margin_rate=0.005),
        strategy=strat,
        max_pos=max_pos,
        cooldown_bars=backtest_cfg['cooldown_bars'],
        stop_atr=backtest_cfg['stop_atr'],
        take_R=backtest_cfg['take_R'],
        trail_start_R=backtest_cfg['trail_start_R'],
        trail_atr=backtest_cfg['trail_atr'],
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=backtest_cfg['risk_per_trade'],
        enable_risk_position_sizing=True,
        allow_reentry=True,
        partial_take_R=backtest_cfg.get('partial_take_R', 0.0),
        partial_take_frac=backtest_cfg.get('partial_take_frac', 0.0),
        break_even_after_partial=backtest_cfg.get('break_even_after_partial', False),
        break_even_R=backtest_cfg.get('break_even_R', 0.0),
    )
    out = bt.run(sig)
    st = out.attrs.get('stats', {})
    trades = pd.DataFrame(out.attrs.get('closed_trades', []))
    eq = out['equity']
    dd = (eq / eq.cummax() - 1).min()
    remove_best = None
    if not trades.empty and 'realized_net' in trades.columns and len(trades) > 1:
        trades = trades.copy()
        trades['realized_net'] = trades['realized_net'].astype(float)
        remove_best = float(trades.drop(trades['realized_net'].idxmax())['realized_net'].sum())
    return {
        'variant': name,
        'return': float(eq.iloc[-1] / INITIAL_CASH - 1),
        'max_drawdown': float(dd),
        'trade_count': int(st.get('closed_trade_count', 0)),
        'fees': float(st.get('total_fees', 0.0)),
        'gross_closed_pnl': float(st.get('gross_closed_pnl', 0.0)),
        'net_closed_pnl': float(st.get('net_closed_pnl', 0.0)),
        'profit_factor': st.get('profit_factor', float('nan')),
        'expectancy_per_trade': float(st.get('expectancy_per_trade', 0.0)),
        'remove_best_trade_net': remove_best,
    }


def main():
    # baseline: 当前 BTC 参数直接迁移到 SOL（对照）
    baseline_strat = {
        'adx_threshold_4h': 28,
        'trend_strength_threshold_4h': 0.0055,
        'breakout_confirm_atr': 0.12,
        'breakout_body_atr': 0.20,
        'pullback_bars': 4,
        'pullback_max_depth_atr': 0.40,
        'first_pullback_only': False,
        'max_pullbacks_long': 3,
        'max_pullbacks_short': 1,
        'min_breakout_age_long': 1,
        'rejection_wick_ratio_long': 0.65,
        'rejection_wick_ratio_short': 0.80,
        'allow_short': False,
        'allow_same_bar_entry': False,
        'breakout_valid_bars': 12,
        'atr_pct_low': 0.0030,
        'atr_pct_high': 0.016,
        'enable_continuation_long': False,
    }
    baseline_bt = {
        'leverage': 3.0,
        'max_pos': 1.2,
        'risk_per_trade': 0.015,
        'cooldown_bars': 2,
        'stop_atr': 1.4,
        'take_R': 3.4,
        'trail_start_R': 1.6,
        'trail_atr': 2.8,
        'partial_take_R': 2.0,
        'partial_take_frac': 0.33,
        'break_even_after_partial': False,
        'break_even_R': 1.2,
    }

    # SOL 专属：更快节奏、更高波动容忍、更快兑现
    sol_strat = {
        'adx_threshold_4h': 24,
        'trend_strength_threshold_4h': 0.0045,
        'breakout_confirm_atr': 0.08,
        'breakout_body_atr': 0.15,
        'pullback_bars': 3,
        'pullback_max_depth_atr': 0.55,
        'first_pullback_only': False,
        'max_pullbacks_long': 4,
        'max_pullbacks_short': 1,
        'min_breakout_age_long': 1,
        'rejection_wick_ratio_long': 0.45,
        'rejection_wick_ratio_short': 0.80,
        'allow_short': False,
        'allow_same_bar_entry': False,
        'breakout_valid_bars': 10,
        'atr_pct_low': 0.0040,
        'atr_pct_high': 0.030,
        'enable_continuation_long': False,
    }
    sol_bt = {
        'leverage': 4.0,
        'max_pos': 1.5,
        'risk_per_trade': 0.020,
        'cooldown_bars': 1,
        'stop_atr': 1.2,
        'take_R': 2.2,
        'trail_start_R': 0.9,
        'trail_atr': 1.8,
        'partial_take_R': 1.2,
        'partial_take_frac': 0.5,
        'break_even_after_partial': True,
        'break_even_R': 0.8,
    }

    # 更激进 SOL 版
    sol_push_bt = {
        'leverage': 5.0,
        'max_pos': 1.8,
        'risk_per_trade': 0.025,
        'cooldown_bars': 1,
        'stop_atr': 1.2,
        'take_R': 2.6,
        'trail_start_R': 1.0,
        'trail_atr': 2.0,
        'partial_take_R': 1.4,
        'partial_take_frac': 0.4,
        'break_even_after_partial': True,
        'break_even_R': 0.9,
    }

    rows = [
        run_variant('SOL_BTC_PORT', baseline_strat, baseline_bt),
        run_variant('SOL_NATIVE_V1', sol_strat, sol_bt),
        run_variant('SOL_NATIVE_V1_PUSH', sol_strat, sol_push_bt),
    ]

    rep = pd.DataFrame(rows).sort_values('net_closed_pnl', ascending=False)
    print('\n=== SOL STRATEGY TEST ===')
    print(rep.to_string(index=False, formatters={
        'return': lambda x: f'{x:.2%}',
        'max_drawdown': lambda x: f'{x:.2%}',
        'fees': lambda x: f'{x:.2f}',
        'gross_closed_pnl': lambda x: f'{x:.2f}',
        'net_closed_pnl': lambda x: f'{x:.2f}',
        'profit_factor': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'expectancy_per_trade': lambda x: f'{x:.2f}',
        'remove_best_trade_net': lambda x: f'{x:.2f}' if x is not None and pd.notna(x) else 'N/A',
    }))


if __name__ == '__main__':
    main()
