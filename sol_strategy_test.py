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


class SOLTrendReclaimStrategy1H:
    """
    更适合高波动 alt（SOL）的原型：
    4h 强势趋势过滤 + 1h 深回撤后恢复确认

    逻辑：
    1) 4h 处于上升趋势 + 趋势强度达标
    2) 1h 出现较深回撤（跌到 EMA20 下方、RSI 降温）
    3) 出现恢复确认：重新站回 EMA20 且收阳，并突破前一根高点
    4) 仅 long-only
    """

    def __init__(
        self,
        fast_4h=5,
        slow_4h=15,
        adx_period_4h=14,
        adx_threshold_4h=20,
        trend_strength_threshold_4h=0.0035,
        atr_period=14,
        atr_pct_low=0.004,
        atr_pct_high=0.05,
        ema_fast=20,
        ema_slow=50,
        reclaim_buffer_atr=0.10,
        pullback_depth_atr=0.6,
        rsi_period=14,
        rsi_pullback_max=48,
        volume_lookback=20,
    ):
        self.fast_4h = fast_4h
        self.slow_4h = slow_4h
        self.adx_period_4h = adx_period_4h
        self.adx_threshold_4h = adx_threshold_4h
        self.trend_strength_threshold_4h = trend_strength_threshold_4h
        self.atr_period = atr_period
        self.atr_pct_low = atr_pct_low
        self.atr_pct_high = atr_pct_high
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.reclaim_buffer_atr = reclaim_buffer_atr
        self.pullback_depth_atr = pullback_depth_atr
        self.rsi_period = rsi_period
        self.rsi_pullback_max = rsi_pullback_max
        self.volume_lookback = volume_lookback

    @staticmethod
    def _atr(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _adx(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        up_move = high.diff(); down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _rsi(close, period=14):
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def generate_signals(self, df):
        out = df.copy()
        out['atr'] = self._atr(out, self.atr_period)
        out['atr_pct'] = out['atr'] / out['close']
        out['volatility'] = out['atr_pct']
        out['ema20'] = out['close'].ewm(span=self.ema_fast, adjust=False).mean()
        out['ema50'] = out['close'].ewm(span=self.ema_slow, adjust=False).mean()
        out['rsi'] = self._rsi(out['close'], self.rsi_period)
        out['vol_median'] = out['volume'].rolling(self.volume_lookback).median()

        out_4h = out[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        out_4h['ma_fast_4h'] = out_4h['close'].rolling(self.fast_4h).mean()
        out_4h['ma_slow_4h'] = out_4h['close'].rolling(self.slow_4h).mean()
        out_4h['adx_4h'] = self._adx(out_4h, self.adx_period_4h)
        out_4h['trend_strength_4h'] = (out_4h['ma_fast_4h'] - out_4h['ma_slow_4h']).abs() / out_4h['close']

        out['ma_fast_4h'] = out_4h['ma_fast_4h'].reindex(out.index, method='ffill')
        out['ma_slow_4h'] = out_4h['ma_slow_4h'].reindex(out.index, method='ffill')
        out['adx_4h'] = out_4h['adx_4h'].reindex(out.index, method='ffill')
        out['trend_strength_4h'] = out_4h['trend_strength_4h'].reindex(out.index, method='ffill')

        regime_ok = (
            (out['ma_fast_4h'] > out['ma_slow_4h']) &
            (out['adx_4h'] >= self.adx_threshold_4h) &
            (out['trend_strength_4h'] >= self.trend_strength_threshold_4h) &
            (out['atr_pct'] >= self.atr_pct_low) &
            (out['atr_pct'] <= self.atr_pct_high)
        )

        # 回撤特征：低点跌到 EMA20 下方一定 ATR，且 RSI 回落
        out['pullback_depth'] = ((out['ema20'] - out['low']) / out['atr']).clip(lower=0)
        pullback_ready = (
            regime_ok &
            (out['pullback_depth'] >= self.pullback_depth_atr) &
            (out['rsi'] <= self.rsi_pullback_max)
        )

        # 恢复确认：重新站回 EMA20，并突破前一根高点，收阳，量能不弱
        reclaim = out['close'] >= (out['ema20'] + self.reclaim_buffer_atr * out['atr'])
        bullish = out['close'] > out['open']
        break_prev_high = out['close'] > out['high'].shift(1)
        volume_ok = out['volume'] >= out['vol_median']

        out['state_signal'] = np.where(regime_ok, 1, 0)
        out['entry_setup'] = np.where(pullback_ready.shift(1).rolling(3).max().fillna(0).astype(bool) & reclaim & bullish & break_prev_high & volume_ok, 1, 0)
        out['signal'] = out['state_signal']
        out['trade_signal'] = out['entry_setup']
        out['entry_reason'] = np.where(out['entry_setup'] == 1, 'sol_trend_reclaim', 'none')
        out['resistance_7d'] = out['high'].rolling(24 * 7, min_periods=24 * 7).max()
        out['support_7d'] = out['low'].rolling(24 * 7, min_periods=24 * 7).min()

        print('state long true:', float((out['state_signal'] == 1).mean()))
        print('entry long setup:', float((out['entry_setup'] == 1).mean()))
        return out.dropna(subset=['atr', 'ema20', 'ema50', 'adx_4h', 'trend_strength_4h'])


def run_variant(name, strat, backtest_cfg):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)
    sig = strat.generate_signals(df)

    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=PerpPortfolio(INITIAL_CASH, leverage=backtest_cfg['leverage'], taker_fee_rate=TAKER_FEE_RATE, maker_fee_rate=MAKER_FEE_RATE, maint_margin_rate=0.005),
        strategy=strat,
        max_pos=backtest_cfg['max_pos'],
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
    btc_port = BTCPerpPullbackStrategy1H(
        adx_threshold_4h=28,
        trend_strength_threshold_4h=0.0055,
        breakout_confirm_atr=0.12,
        breakout_body_atr=0.20,
        pullback_bars=4,
        pullback_max_depth_atr=0.40,
        first_pullback_only=False,
        max_pullbacks_long=3,
        max_pullbacks_short=1,
        min_breakout_age_long=1,
        rejection_wick_ratio_long=0.65,
        rejection_wick_ratio_short=0.80,
        allow_short=False,
        allow_same_bar_entry=False,
        breakout_valid_bars=12,
        atr_pct_low=0.0030,
        atr_pct_high=0.016,
        enable_continuation_long=False,
    )
    sol_native = SOLTrendReclaimStrategy1H()
    sol_native_push = SOLTrendReclaimStrategy1H(
        adx_threshold_4h=18,
        trend_strength_threshold_4h=0.003,
        reclaim_buffer_atr=0.05,
        pullback_depth_atr=0.45,
        rsi_pullback_max=52,
    )

    conservative_bt = {
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
    native_bt = {
        'leverage': 4.0,
        'max_pos': 1.5,
        'risk_per_trade': 0.020,
        'cooldown_bars': 1,
        'stop_atr': 1.2,
        'take_R': 2.4,
        'trail_start_R': 1.0,
        'trail_atr': 1.8,
        'partial_take_R': 1.2,
        'partial_take_frac': 0.5,
        'break_even_after_partial': True,
        'break_even_R': 0.8,
    }
    native_push_bt = {
        'leverage': 5.0,
        'max_pos': 1.8,
        'risk_per_trade': 0.025,
        'cooldown_bars': 1,
        'stop_atr': 1.1,
        'take_R': 2.8,
        'trail_start_R': 1.2,
        'trail_atr': 2.0,
        'partial_take_R': 1.4,
        'partial_take_frac': 0.4,
        'break_even_after_partial': True,
        'break_even_R': 0.9,
    }

    rows = [
        run_variant('SOL_BTC_PORT', btc_port, conservative_bt),
        run_variant('SOL_TREND_RECLAIM_V1', sol_native, native_bt),
        run_variant('SOL_TREND_RECLAIM_V1_PUSH', sol_native_push, native_push_bt),
    ]

    rep = pd.DataFrame(rows).sort_values('net_closed_pnl', ascending=False)
    print('\n=== SOL NEW STRATEGY TEST ===')
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
