"""
SOL RevV2 多时间区间对比
测试相同参数在三段不同历史上的表现，验证策略是否存在过拟合
"""

import pandas as pd
import numpy as np

from data import CCXTDataSource
from strategy import SOLReversionV2Strategy1H
from strategy_profiles import get_sol_backtest_profile
from backtest import Backtester
from portfolio import PerpPortfolio
from broker import SimBroker
import config

# ─────────────────────────────────────────────────────────────
# 固定策略参数（不随区间变化）
# ─────────────────────────────────────────────────────────────
SYMBOL = "SOL/USDT:USDT"
CASH   = 10_000

STRAT_PARAMS = {
    "atr_period":      14,
    "bb_period":       20,
    "bb_std":          2.0,
    "rsi_period":      14,
    "rsi_oversold":    40,
    "rsi_overbought":  60,
    "reclaim_ema":     20,
    "trend_ema":       200,
    "vol_period":      20,
    "vol_spike_mult":  1.2,
    "atr_pct_low":     0.003,
    "atr_pct_high":    0.10,
    "oversold_lookback": 3,
    "allow_short":     True,
}

PERIODS = [
    ("2023-01-01", "2024-01-01", "2023全年（牛市）"),
    ("2024-01-01", "2025-01-01", "2024全年（牛转熊）"),
    ("2025-01-01", "2026-03-24", "2025-26（开发区间）"),
]


def run_period(start, end, label):
    print(f"\n{'─'*60}")
    print(f"  {label}  [{start} → {end}]")
    print(f"{'─'*60}")

    ds  = CCXTDataSource()
    df  = ds.load_ohlcv(symbol=SYMBOL, start=start, end=end, timeframe="1h")
    print(f"  K线数量: {len(df)}  首: {df.index[0]}  末: {df.index[-1]}")

    strat    = SOLReversionV2Strategy1H(**STRAT_PARAMS)
    df_sig   = strat.generate_signals(df)
    n_long   = int((df_sig["entry_setup"] ==  1).sum())
    n_short  = int((df_sig["entry_setup"] == -1).sum())
    print(f"  信号: 做多 {n_long} / 做空 {n_short}")

    bt_params = get_sol_backtest_profile("sol_rev_v2")

    portfolio = PerpPortfolio(
        initial_cash      = CASH,
        leverage          = bt_params["leverage"],
        taker_fee_rate    = config.TAKER_FEE_RATE,
        maker_fee_rate    = config.MAKER_FEE_RATE,
        maint_margin_rate = 0.005,
    )
    broker = SimBroker(slippage_bps=config.SLIPPAGE_BPS)

    bt = Backtester(
        broker                      = broker,
        portfolio                   = portfolio,
        strategy                    = strat,
        max_pos                     = bt_params["max_pos"],
        cooldown_bars               = bt_params["cooldown_bars"],
        stop_atr                    = bt_params["stop_atr"],
        take_R                      = bt_params["take_R"],
        trail_start_R               = bt_params["trail_start_R"],
        trail_atr                   = bt_params["trail_atr"],
        use_trailing                = bt_params["use_trailing"],
        check_liq                   = True,
        entry_is_maker              = bt_params["entry_is_maker"],
        funding_rate_per_8h         = config.FUNDING_RATE_PER_8H,
        risk_per_trade              = bt_params["risk_per_trade"],
        enable_risk_position_sizing = bt_params["enable_risk_position_sizing"],
        allow_reentry               = bt_params["allow_reentry"],
        partial_take_R              = bt_params["partial_take_R"],
        partial_take_frac           = bt_params["partial_take_frac"],
        break_even_after_partial    = bt_params["break_even_after_partial"],
        break_even_R                = bt_params["break_even_R"],
        use_signal_exit_targets     = bt_params["use_signal_exit_targets"],
        max_hold_bars               = bt_params["max_hold_bars"],
    )

    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])

    final_equity = result["equity"].iloc[-1]
    peak_equity  = result["equity"].cummax()
    drawdown     = (result["equity"] - peak_equity) / peak_equity
    max_dd       = drawdown.min()
    total_return = (final_equity - CASH) / CASH * 100

    # SOL 同期涨跌幅（基准对比）
    sol_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    print(f"\n  总收益率:     {total_return:+.2f}%   (SOL 同期: {sol_return:+.2f}%)")
    print(f"  最大回撤:     {max_dd*100:.2f}%")
    print(f"  已结算交易:   {stats.get('closed_trade_count', len(closed))}")
    print(f"  胜率:         {stats.get('win_rate', 0)*100:.1f}%")
    print(f"  盈亏比:       {stats.get('pnl_ratio', 0):.2f}")
    print(f"  利润因子:     {stats.get('profit_factor', 0):.2f}")
    print(f"  期望值/笔:    {stats.get('expectancy_per_trade', 0):.2f} USDT")
    print(f"  累计手续费:   {stats.get('total_fees', 0):.2f} USDT")

    exit_split = stats.get("exit_reason_split", {})
    if exit_split:
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(exit_split.items()))
        print(f"  出场分布:     {parts}")

    return {
        "label":          label,
        "start":          start,
        "end":            end,
        "return_pct":     total_return,
        "sol_return_pct": sol_return,
        "max_dd_pct":     max_dd * 100,
        "closed_trades":  stats.get("closed_trade_count", len(closed)),
        "win_rate":       stats.get("win_rate", 0) * 100,
        "pnl_ratio":      stats.get("pnl_ratio", 0),
        "profit_factor":  stats.get("profit_factor", 0),
        "expectancy":     stats.get("expectancy_per_trade", 0),
        "fees":           stats.get("total_fees", 0),
    }


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SOL RevV2 多区间对比（参数完全固定，仅改时间窗口）")
print("=" * 60)

rows = []
for start, end, label in PERIODS:
    rows.append(run_period(start, end, label))

# ─────────────────────────────────────────────────────────────
# 汇总表
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  汇总对比")
print("=" * 60)
print(f"{'区间':<18} {'收益率':>8} {'SOL涨跌':>8} {'最大回撤':>9} {'成交':>5} {'胜率':>7} {'盈亏比':>7} {'利润因子':>8}")
print("-" * 75)
for r in rows:
    print(
        f"{r['label']:<18} "
        f"{r['return_pct']:>+7.1f}%  "
        f"{r['sol_return_pct']:>+7.1f}%  "
        f"{r['max_dd_pct']:>8.1f}%  "
        f"{r['closed_trades']:>4}  "
        f"{r['win_rate']:>6.1f}%  "
        f"{r['pnl_ratio']:>6.2f}  "
        f"{r['profit_factor']:>7.2f}"
    )
print("=" * 60)
print("\n注：策略参数完全相同，仅时间窗口不同。")
print("    2025-26 为开发区间（in-sample），其余两段为未见过的数据（out-of-sample）。")
