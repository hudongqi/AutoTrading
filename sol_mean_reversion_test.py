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


class SOLMeanReversionStrategy1H:
    """
    SOL 原型：过度下杀后的反抽 / 均值回归
    仅做 long，先验证 gross edge。
    """

    def __init__(
        self,
        atr_period=14,
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_oversold=30,
        reclaim_ema=20,
        reclaim_confirm_atr=0.05,
        adx_period_4h=14,
        adx_cap_4h=32,
        atr_pct_low=0.004,
        atr_pct_high=0.05,
    ):
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.reclaim_ema = reclaim_ema
        self.reclaim_confirm_atr = reclaim_confirm_atr
        self.adx_period_4h = adx_period_4h
        self.adx_cap_4h = adx_cap_4h
        self.atr_pct_low = atr_pct_low
        self.atr_pct_high = atr_pct_high

    @staticmethod
    def _atr(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _rsi(close, period=14):
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

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

    def generate_signals(self, df):
        out = df.copy()
        out['atr'] = self._atr(out, self.atr_period)
        out['atr_pct'] = out['atr'] / out['close']
        out['volatility'] = out['atr_pct']
        out['ema20'] = out['close'].ewm(span=self.reclaim_ema, adjust=False).mean()
        out['rsi'] = self._rsi(out['close'], self.rsi_period)
        out['bb_mid'] = out['close'].rolling(self.bb_period).mean()
        out['bb_std'] = out['close'].rolling(self.bb_period).std()
        out['bb_lower'] = out['bb_mid'] - self.bb_std * out['bb_std']
        out['bb_upper'] = out['bb_mid'] + self.bb_std * out['bb_std']

        out_4h = out[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        out_4h['adx_4h'] = self._adx(out_4h, self.adx_period_4h)
        out['adx_4h'] = out_4h['adx_4h'].reindex(out.index, method='ffill')

        out['state_signal'] = 0
        # 反转策略：避免最强趋势中硬抄底，因此要求 ADX 不过高
        regime_ok = (
            (out['adx_4h'] <= self.adx_cap_4h) &
            (out['atr_pct'] >= self.atr_pct_low) &
            (out['atr_pct'] <= self.atr_pct_high)
        )

        oversold = (out['close'] < out['bb_lower']) & (out['rsi'] <= self.rsi_oversold)
        reclaim = out['close'] >= (out['ema20'] + self.reclaim_confirm_atr * out['atr'])
        bullish = out['close'] > out['open']
        break_prev_high = out['close'] > out['high'].shift(1)

        out['entry_setup'] = np.where(oversold.shift(1).rolling(3).max().fillna(0).astype(bool) & reclaim & bullish & break_prev_high & regime_ok, 1, 0)
        out['signal'] = np.where(regime_ok, 1, 0)
        out['trade_signal'] = out['entry_setup']
        out['entry_reason'] = np.where(out['entry_setup'] == 1, 'sol_mean_reversion_reclaim', 'none')
        out['resistance_7d'] = out['high'].rolling(24 * 7, min_periods=24 * 7).max()
        out['support_7d'] = out['low'].rolling(24 * 7, min_periods=24 * 7).min()
        print('state long true:', float((out['signal'] == 1).mean()))
        print('entry long setup:', float((out['entry_setup'] == 1).mean()))
        return out.dropna(subset=['atr', 'ema20', 'bb_lower', 'adx_4h'])


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
    conservative = {
        'leverage': 3.0,
        'max_pos': 1.2,
        'risk_per_trade': 0.015,
        'cooldown_bars': 2,
        'stop_atr': 1.2,
        'take_R': 1.8,
        'trail_start_R': 0.8,
        'trail_atr': 1.6,
        'partial_take_R': 1.0,
        'partial_take_frac': 0.5,
        'break_even_after_partial': True,
        'break_even_R': 0.8,
    }
    push = {
        'leverage': 4.0,
        'max_pos': 1.5,
        'risk_per_trade': 0.020,
        'cooldown_bars': 1,
        'stop_atr': 1.1,
        'take_R': 2.2,
        'trail_start_R': 0.9,
        'trail_atr': 1.8,
        'partial_take_R': 1.2,
        'partial_take_frac': 0.4,
        'break_even_after_partial': True,
        'break_even_R': 0.8,
    }

    rows = [
        run_variant('SOL_MEAN_REV_V1', SOLMeanReversionStrategy1H(), conservative),
        run_variant('SOL_MEAN_REV_V1_PUSH', SOLMeanReversionStrategy1H(adx_cap_4h=36, rsi_oversold=34), push),
    ]
    rep = pd.DataFrame(rows).sort_values('net_closed_pnl', ascending=False)
    print('\n=== SOL MEAN REVERSION TEST ===')
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
