#!/usr/bin/env python3
import pandas as pd
import numpy as np

from data import CCXTDataSource
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester
from config import INITIAL_CASH, TAKER_FEE_RATE, MAKER_FEE_RATE, FUNDING_RATE_PER_8H, SLIPPAGE_BPS

SYMBOL = 'SOL/USDT:USDT'
START = '2025-01-01'
END = '2026-03-19'


class SOLRoute1VolBreakout:
    """
    Route 1 research strategy: SOL volatility compression -> expansion breakout.

    Supports two entry modes:
    - chase: confirmed breakout close, enter next bar
    - pullback: breakout confirmed first, then wait a few bars for shallow retest and bullish re-acceptance
    """

    def __init__(
        self,
        name='ROUTE1',
        atr_period=14,
        bb_period=20,
        bb_std=2.0,
        squeeze_lookback=120,
        squeeze_bw_q=0.20,
        squeeze_atr_q=0.30,
        trend_ema_fast=20,
        trend_ema_mid=50,
        trend_ema_slow=100,
        trend_slope_bars=8,
        range_lookback=24,
        range_pos_threshold=0.65,
        breakout_lookback=20,
        breakout_buffer_atr=0.10,
        body_frac_min=0.55,
        close_near_high_max=0.25,
        range_expand_mult=1.30,
        volume_lookback=20,
        volume_mult=1.50,
        require_volume=True,
        entry_mode='chase',
        pullback_window=3,
        pullback_max_atr=0.60,
        reclaim_buffer_atr=0.05,
        squeeze_recent_bars=12,
    ):
        vars(self).update(locals())
        del self.self

    @staticmethod
    def _atr(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def generate_signals(self, df):
        out = df.copy()
        out['atr'] = self._atr(out, self.atr_period)
        out['volatility'] = out['atr'] / out['close']
        out['atr_pct'] = out['volatility']

        out['ema_fast'] = out['close'].ewm(span=self.trend_ema_fast, adjust=False).mean()
        out['ema_mid'] = out['close'].ewm(span=self.trend_ema_mid, adjust=False).mean()
        out['ema_slow'] = out['close'].ewm(span=self.trend_ema_slow, adjust=False).mean()
        out['ema_mid_slope'] = out['ema_mid'] - out['ema_mid'].shift(self.trend_slope_bars)

        out['bb_mid'] = out['close'].rolling(self.bb_period).mean()
        out['bb_sigma'] = out['close'].rolling(self.bb_period).std()
        out['bb_upper'] = out['bb_mid'] + self.bb_std * out['bb_sigma']
        out['bb_lower'] = out['bb_mid'] - self.bb_std * out['bb_sigma']
        out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / out['bb_mid'].replace(0, np.nan)
        out['bw_q'] = out['bb_width'].rolling(self.squeeze_lookback).quantile(self.squeeze_bw_q)
        out['atr_q'] = out['atr_pct'].rolling(self.squeeze_lookback).quantile(self.squeeze_atr_q)

        out['range_high'] = out['high'].rolling(self.range_lookback).max().shift(1)
        out['range_low'] = out['low'].rolling(self.range_lookback).min().shift(1)
        out['range_size'] = out['range_high'] - out['range_low']
        out['range_pos'] = (out['close'] - out['range_low']) / out['range_size'].replace(0, np.nan)

        out['recent_high'] = out['high'].rolling(self.breakout_lookback).max().shift(1)
        out['tr'] = pd.concat([
            out['high'] - out['low'],
            (out['high'] - out['close'].shift(1)).abs(),
            (out['low'] - out['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        out['tr_ma'] = out['tr'].rolling(20).mean()
        out['vol_ma'] = out['volume'].rolling(self.volume_lookback).mean()

        squeeze = (out['bb_width'] <= out['bw_q']) & (out['atr_pct'] <= out['atr_q'])
        squeeze_recent = squeeze.shift(1).rolling(self.squeeze_recent_bars).max().fillna(0).astype(bool)
        trend_ok = (
            (out['close'] > out['ema_slow']) &
            (out['ema_fast'] > out['ema_mid']) &
            (out['ema_mid'] > out['ema_slow']) &
            (out['ema_mid_slope'] > 0)
        )
        structure_ok = out['range_pos'] >= self.range_pos_threshold

        body = (out['close'] - out['open']).abs()
        body_frac = body / out['tr'].replace(0, np.nan)
        close_near_high = (out['high'] - out['close']) / out['tr'].replace(0, np.nan)
        range_expand_ok = out['tr'] >= self.range_expand_mult * out['tr_ma']
        volume_ok = out['volume'] >= self.volume_mult * out['vol_ma']
        confirm_ok = (
            (body_frac >= self.body_frac_min) &
            (close_near_high <= self.close_near_high_max) &
            range_expand_ok &
            (volume_ok if self.require_volume else True)
        )
        breakout_level = out['recent_high'] + self.breakout_buffer_atr * out['atr']
        breakout_bar = squeeze_recent & trend_ok & structure_ok & (out['close'] > breakout_level) & confirm_ok

        out['breakout_bar_long'] = breakout_bar.astype(int)
        out['breakout_level_long'] = np.where(breakout_bar, out['recent_high'], np.nan)
        out['breakout_quality_long'] = breakout_bar
        out['bars_since_breakout_long'] = np.nan
        out['pullback_depth_long'] = np.nan
        out['continuation_setup_long'] = False
        out['first_pullback_ok_long'] = False
        out['reject_long'] = False
        out['regime_ok'] = trend_ok & structure_ok

        entry_setup = pd.Series(0, index=out.index, dtype=int)
        breakout_age = None
        active_breakout_level = np.nan
        breakout_bar_low = np.nan
        breakout_atr = np.nan

        for i, idx in enumerate(out.index):
            if bool(breakout_bar.iloc[i]):
                breakout_age = 0
                active_breakout_level = float(out['recent_high'].iloc[i])
                breakout_bar_low = float(out['low'].iloc[i])
                breakout_atr = float(out['atr'].iloc[i]) if pd.notna(out['atr'].iloc[i]) else np.nan
                out.at[idx, 'bars_since_breakout_long'] = 0
                if self.entry_mode == 'chase':
                    entry_setup.iloc[i] = 1
                    out.at[idx, 'entry_reason'] = f'{self.name}:chase_breakout'
                continue

            if breakout_age is None:
                out.at[idx, 'entry_reason'] = 'none'
                continue

            breakout_age += 1
            out.at[idx, 'bars_since_breakout_long'] = breakout_age
            out.at[idx, 'breakout_level_long'] = active_breakout_level

            if breakout_age > self.pullback_window:
                breakout_age = None
                active_breakout_level = np.nan
                breakout_bar_low = np.nan
                breakout_atr = np.nan
                out.at[idx, 'entry_reason'] = 'none'
                continue

            if self.entry_mode != 'pullback':
                out.at[idx, 'entry_reason'] = 'none'
                continue

            atr_now = float(out['atr'].iloc[i]) if pd.notna(out['atr'].iloc[i]) else np.nan
            if not np.isfinite(active_breakout_level) or not np.isfinite(atr_now):
                out.at[idx, 'entry_reason'] = 'none'
                continue

            low_i = float(out['low'].iloc[i])
            close_i = float(out['close'].iloc[i])
            open_i = float(out['open'].iloc[i])
            tr_i = float(out['tr'].iloc[i]) if pd.notna(out['tr'].iloc[i]) else np.nan
            body_frac_i = float(body.iloc[i] / tr_i) if np.isfinite(tr_i) and tr_i > 0 else np.nan
            pullback_depth = max(0.0, active_breakout_level - low_i)
            shallow_pullback = pullback_depth <= self.pullback_max_atr * atr_now
            reaccept = close_i >= active_breakout_level + self.reclaim_buffer_atr * atr_now
            bullish_reject = close_i > open_i and (body_frac_i >= 0.45 if pd.notna(body_frac_i) else False)
            hold_breakout_bar_low = (not np.isfinite(breakout_bar_low)) or (low_i >= breakout_bar_low - 0.25 * atr_now)
            cont_ok = bool(out['regime_ok'].iloc[i]) and shallow_pullback and reaccept and bullish_reject and hold_breakout_bar_low

            out.at[idx, 'pullback_depth_long'] = pullback_depth / atr_now if atr_now > 0 else np.nan
            out.at[idx, 'first_pullback_ok_long'] = shallow_pullback
            out.at[idx, 'reject_long'] = bullish_reject and reaccept
            out.at[idx, 'continuation_setup_long'] = cont_ok

            if cont_ok:
                entry_setup.iloc[i] = 1
                out.at[idx, 'entry_reason'] = f'{self.name}:pullback_reaccept'
                breakout_age = None
                active_breakout_level = np.nan
                breakout_bar_low = np.nan
                breakout_atr = np.nan
            else:
                out.at[idx, 'entry_reason'] = 'none'

        out['state_signal'] = np.where(out['regime_ok'], 1, 0)
        out['entry_setup'] = entry_setup
        out['trade_signal'] = entry_setup
        out['signal'] = out['state_signal']
        out['resistance_7d'] = out['high'].rolling(24 * 7, min_periods=24 * 7).max()
        out['support_7d'] = out['low'].rolling(24 * 7, min_periods=24 * 7).min()
        return out.dropna(subset=['atr', 'bw_q', 'atr_q', 'recent_high', 'ema_slow', 'tr_ma', 'vol_ma'])


def run_variant(name, strat, df, cfg):
    sig = strat.generate_signals(df)
    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=PerpPortfolio(
            INITIAL_CASH,
            leverage=cfg['leverage'],
            taker_fee_rate=TAKER_FEE_RATE,
            maker_fee_rate=MAKER_FEE_RATE,
            maint_margin_rate=0.005,
        ),
        strategy=strat,
        max_pos=cfg['max_pos'],
        cooldown_bars=cfg['cooldown_bars'],
        stop_atr=cfg['stop_atr'],
        take_R=cfg['take_R'],
        trail_start_R=cfg['trail_start_R'],
        trail_atr=cfg['trail_atr'],
        use_trailing=True,
        funding_rate_per_8h=FUNDING_RATE_PER_8H,
        risk_per_trade=cfg['risk_per_trade'],
        enable_risk_position_sizing=True,
        allow_reentry=True,
        partial_take_R=cfg['partial_take_R'],
        partial_take_frac=cfg['partial_take_frac'],
        break_even_after_partial=cfg['break_even_after_partial'],
        break_even_R=cfg['break_even_R'],
        use_signal_exit_targets=False,
        max_hold_bars=cfg.get('max_hold_bars', 0),
    )
    out = bt.run(sig)
    st = out.attrs.get('stats', {})
    trades = pd.DataFrame(out.attrs.get('closed_trades', []))
    eq = out['equity']
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0

    avg_winner = avg_loser = remove_best = np.nan
    if not trades.empty and 'realized_net' in trades.columns:
        trades = trades.copy()
        trades['realized_net'] = trades['realized_net'].astype(float)
        wins = trades[trades['realized_net'] > 0]
        losses = trades[trades['realized_net'] < 0]
        avg_winner = float(wins['realized_net'].mean()) if len(wins) else np.nan
        avg_loser = float(losses['realized_net'].mean()) if len(losses) else np.nan
        if len(trades) > 1:
            remove_best = float(trades.drop(trades['realized_net'].idxmax())['realized_net'].sum())

    return {
        'variant': name,
        'entry_mode': strat.entry_mode,
        'return': float(eq.iloc[-1] / INITIAL_CASH - 1),
        'max_drawdown': dd,
        'trade_count': int(st.get('closed_trade_count', 0)),
        'gross_closed_pnl': float(st.get('gross_closed_pnl', 0.0)),
        'net_closed_pnl': float(st.get('net_closed_pnl', 0.0)),
        'profit_factor': float(st.get('profit_factor', np.nan)),
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'expectancy_per_trade': float(st.get('expectancy_per_trade', 0.0)),
        'avg_R_realized': float(st.get('avg_R_realized', np.nan)),
        'remove_best_trade_net': remove_best,
        'mfe_capture_ratio': float(st.get('mfe_capture_ratio', np.nan)),
    }


def main():
    ds = CCXTDataSource()
    df = ds.load_ohlcv(SYMBOL, START, END)

    variants = [
        (
            'ROUTE1_A_CHASE',
            SOLRoute1VolBreakout(
                name='ROUTE1_A',
                breakout_lookback=20,
                squeeze_bw_q=0.20,
                squeeze_atr_q=0.30,
                range_pos_threshold=0.65,
                breakout_buffer_atr=0.10,
                body_frac_min=0.55,
                close_near_high_max=0.25,
                range_expand_mult=1.30,
                volume_mult=1.40,
                require_volume=False,
                entry_mode='chase',
                squeeze_recent_bars=12,
            ),
            dict(
                leverage=4.0, max_pos=2.0, risk_per_trade=0.03, cooldown_bars=1,
                stop_atr=1.8, take_R=8.0, trail_start_R=1.2, trail_atr=2.4,
                partial_take_R=0.0, partial_take_frac=0.0,
                break_even_after_partial=False, break_even_R=0.0,
            ),
        ),
        (
            'ROUTE1_B_PULLBACK',
            SOLRoute1VolBreakout(
                name='ROUTE1_B',
                breakout_lookback=20,
                squeeze_bw_q=0.20,
                squeeze_atr_q=0.30,
                range_pos_threshold=0.65,
                breakout_buffer_atr=0.10,
                body_frac_min=0.55,
                close_near_high_max=0.25,
                range_expand_mult=1.30,
                volume_mult=1.50,
                require_volume=True,
                entry_mode='pullback',
                pullback_window=3,
                pullback_max_atr=0.60,
                reclaim_buffer_atr=0.05,
                squeeze_recent_bars=12,
            ),
            dict(
                leverage=4.0, max_pos=2.0, risk_per_trade=0.03, cooldown_bars=1,
                stop_atr=1.8, take_R=8.0, trail_start_R=1.2, trail_atr=2.2,
                partial_take_R=1.5, partial_take_frac=0.50,
                break_even_after_partial=True, break_even_R=0.8,
            ),
        ),
        (
            'ROUTE1_C_STRICT',
            SOLRoute1VolBreakout(
                name='ROUTE1_C',
                breakout_lookback=30,
                squeeze_bw_q=0.10,
                squeeze_atr_q=0.20,
                range_pos_threshold=0.72,
                breakout_buffer_atr=0.15,
                body_frac_min=0.60,
                close_near_high_max=0.20,
                range_expand_mult=1.45,
                volume_mult=1.70,
                require_volume=True,
                entry_mode='chase',
                squeeze_recent_bars=8,
            ),
            dict(
                leverage=5.0, max_pos=2.5, risk_per_trade=0.04, cooldown_bars=1,
                stop_atr=2.0, take_R=10.0, trail_start_R=1.5, trail_atr=2.6,
                partial_take_R=2.0, partial_take_frac=0.50,
                break_even_after_partial=True, break_even_R=1.0,
            ),
        ),
    ]

    rows = [run_variant(name, strat, df, cfg) for name, strat, cfg in variants]
    rep = pd.DataFrame(rows).sort_values(['net_closed_pnl', 'gross_closed_pnl'], ascending=False)
    print(rep.to_string(index=False, formatters={
        'return': lambda x: f'{x:.2%}',
        'max_drawdown': lambda x: f'{x:.2%}',
        'gross_closed_pnl': lambda x: f'{x:.2f}',
        'net_closed_pnl': lambda x: f'{x:.2f}',
        'profit_factor': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'avg_winner': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'avg_loser': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'expectancy_per_trade': lambda x: f'{x:.2f}',
        'avg_R_realized': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'remove_best_trade_net': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'mfe_capture_ratio': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
    }))


if __name__ == '__main__':
    main()
