"""
SOL Reversion V2 回测脚本
策略：SOLReversionV2Strategy1H
周期：2025-01-01 → 2026-03-24
资金：10,000 USDT
"""

import sys
import pandas as pd
import numpy as np

from data import CCXTDataSource
from strategy import SOLReversionV2Strategy1H
from strategy_profiles import get_sol_backtest_profile, SOL_MEAN_REVERSION_PROFILES
from backtest import Backtester
from portfolio import PerpPortfolio
from broker import SimBroker
import config


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────
SYMBOL   = "SOL/USDT:USDT"
START    = "2024-01-01"
END      = "2025-01-01"
CASH     = 10_000

STRAT_PROFILE  = "sol_rev_v2"   # strategy_profiles.SOL_BACKTEST_PROFILES["sol_rev_v2"] 的策略参数
BT_PROFILE     = "sol_rev_v2"   # 同一条目里也含回测参数


# ─────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────
print(f"[1/4] 拉取 {SYMBOL} 1H 数据 {START} → {END} ...")
ds = CCXTDataSource()
df = ds.load_ohlcv(symbol=SYMBOL, start=START, end=END, timeframe="1h")
print(f"      共 {len(df)} 根 K 线，首根：{df.index[0]}，末根：{df.index[-1]}")


# ─────────────────────────────────────────────────────────────
# 策略参数（从 sol_rev_v2 profile 取）
# ─────────────────────────────────────────────────────────────
strat_params = {
    "atr_period":       14,
    "bb_period":        20,
    "bb_std":           2.0,
    "rsi_period":       14,
    "rsi_oversold":     38,    # 参数扫描最优：38/62（比 40/60 Calmar +15%）
    "rsi_overbought":   62,
    "reclaim_ema":      20,
    "trend_ema":        200,
    "vol_period":       20,
    "vol_spike_mult":   1.2,
    "atr_pct_low":      0.003,
    "atr_pct_high":     0.10,
    "oversold_lookback": 3,    # 3 根 bar 内有 BB 触碰 + RSI 极值
    "allow_short":      True,
}

print("[2/4] 生成策略信号 ...")
strat = SOLReversionV2Strategy1H(**strat_params)
df_sig = strat.generate_signals(df)
n_long  = int((df_sig["entry_setup"] == 1).sum())
n_short = int((df_sig["entry_setup"] == -1).sum())
print(f"      信号 bar 数：{len(df_sig)}，做多信号：{n_long}，做空信号：{n_short}")


# ─────────────────────────────────────────────────────────────
# 回测参数
# ─────────────────────────────────────────────────────────────
bt_params = get_sol_backtest_profile(BT_PROFILE)

portfolio = PerpPortfolio(
    initial_cash      = CASH,
    leverage          = bt_params["leverage"],
    taker_fee_rate    = config.TAKER_FEE_RATE,
    maker_fee_rate    = config.MAKER_FEE_RATE,
    maint_margin_rate = 0.005,
)
broker = SimBroker(slippage_bps=config.SLIPPAGE_BPS)

bt = Backtester(
    broker                    = broker,
    portfolio                 = portfolio,
    strategy                  = strat,
    max_pos                   = bt_params["max_pos"],
    cooldown_bars             = bt_params["cooldown_bars"],
    stop_atr                  = bt_params["stop_atr"],
    take_R                    = bt_params["take_R"],
    trail_start_R             = bt_params["trail_start_R"],
    trail_atr                 = bt_params["trail_atr"],
    use_trailing              = bt_params["use_trailing"],
    check_liq                 = True,
    entry_is_maker            = bt_params["entry_is_maker"],
    funding_rate_per_8h       = config.FUNDING_RATE_PER_8H,
    risk_per_trade            = bt_params["risk_per_trade"],
    enable_risk_position_sizing = bt_params["enable_risk_position_sizing"],
    allow_reentry             = bt_params["allow_reentry"],
    partial_take_R            = bt_params["partial_take_R"],
    partial_take_frac         = bt_params["partial_take_frac"],
    break_even_after_partial  = bt_params["break_even_after_partial"],
    break_even_R              = bt_params["break_even_R"],
    use_signal_exit_targets   = bt_params["use_signal_exit_targets"],
    max_hold_bars             = bt_params["max_hold_bars"],
)

print("[3/4] 运行回测 ...")
result = bt.run(df_sig)


# ─────────────────────────────────────────────────────────────
# 结果输出
# ─────────────────────────────────────────────────────────────
print("\n[4/4] 回测结果\n" + "=" * 56)

stats = result.attrs.get("stats", {})
closed = result.attrs.get("closed_trades", [])

final_equity = result["equity"].iloc[-1]
peak_equity  = result["equity"].cummax()
drawdown     = (result["equity"] - peak_equity) / peak_equity
max_dd       = drawdown.min()

# 权益曲线收益
total_return = (final_equity - CASH) / CASH * 100

print(f"  初始资金：      {CASH:,.0f} USDT")
print(f"  最终资金：      {final_equity:,.2f} USDT")
print(f"  总收益率：      {total_return:+.2f}%")
print(f"  最大回撤：      {max_dd*100:.2f}%")
print()

print(f"  总交易次数：    {stats.get('trade_count', 0)}")
print(f"  已结算交易：    {stats.get('closed_trade_count', len(closed))}")
print(f"  胜率：          {stats.get('win_rate', 0)*100:.1f}%")
print(f"  盈亏比：        {stats.get('pnl_ratio', 0):.2f}")
print(f"  期望值/笔：     {stats.get('expectancy_per_trade', 0):.2f} USDT")
print(f"  利润因子：      {stats.get('profit_factor', 0):.2f}")
print(f"  平均持仓 bar：  {stats.get('avg_holding_bars', 0):.1f}")
print(f"  在市时间：      {stats.get('time_in_market', 0)*100:.1f}%")
print(f"  平均 R 实现：   {stats.get('avg_R_realized', 0):.2f}")
print(f"  累计手续费：    {stats.get('total_fees', 0):.2f} USDT")
print()

# 出场原因分布
exit_split = stats.get("exit_reason_split", {})
if exit_split:
    print("  出场原因分布：")
    for reason, cnt in sorted(exit_split.items(), key=lambda x: -x[1]):
        print(f"    {reason:<22} {cnt} 次")
print()

# 近 5 笔交易
if closed:
    print("  最近 5 笔交易：")
    for t in closed[-5:]:
        pnl = t.get("realized_net", 0)
        sign = "+" if pnl >= 0 else ""
        print(
            f"    {str(t.get('entry_time',''))[:16]}  "
            f"入场 {t.get('entry_price',0):.2f}  "
            f"出场 {t.get('exit_price',0):.2f}  "
            f"原因 {t.get('exit_reason','?'):<14}  "
            f"净损益 {sign}{pnl:.2f} USDT"
        )

print("\n" + "=" * 56)
