"""
SOL RevV2 季度拆分回测
将 2024-2026 按季度拆成 9 段逐个测试，验证策略在短周期是否仍有稳定表现
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

SYMBOL = "SOL/USDT:USDT"
CASH   = 10_000

STRAT_PARAMS = {
    "atr_period":       14,
    "bb_period":        20,
    "bb_std":           2.0,
    "rsi_period":       14,
    "rsi_oversold":     40,
    "rsi_overbought":   60,
    "reclaim_ema":      20,
    "trend_ema":        200,
    "vol_period":       20,
    "vol_spike_mult":   1.2,
    "atr_pct_low":      0.003,
    "atr_pct_high":     0.10,
    "oversold_lookback": 3,
    "allow_short":      True,
}

QUARTERS = [
    ("2024-01-01", "2024-04-01", "2024 Q1"),
    ("2024-04-01", "2024-07-01", "2024 Q2"),
    ("2024-07-01", "2024-10-01", "2024 Q3"),
    ("2024-10-01", "2025-01-01", "2024 Q4"),
    ("2025-01-01", "2025-04-01", "2025 Q1"),
    ("2025-04-01", "2025-07-01", "2025 Q2"),
    ("2025-07-01", "2025-10-01", "2025 Q3"),
    ("2025-10-01", "2026-01-01", "2025 Q4"),
    ("2026-01-01", "2026-03-24", "2026 Q1（不完整）"),
]

# 提前拉取全量数据，避免每季度重复请求
print("拉取全量数据 2024-01-01 → 2026-03-24 ...")
ds  = CCXTDataSource()
df_all = ds.load_ohlcv(symbol=SYMBOL, start="2024-01-01", end="2026-03-24", timeframe="1h")
print(f"共 {len(df_all)} 根 K 线\n")


def run_quarter(start, end, label, df_full):
    # 从全量数据中切片（保留 ema_90d 预热所需的历史）
    # 策略内部会重新计算指标，因此直接截取当前区间即可
    df = df_full[(df_full.index >= pd.Timestamp(start, tz="UTC"))
               & (df_full.index <  pd.Timestamp(end,   tz="UTC"))].copy()

    if len(df) < 200:
        return None  # 数据太少跳过

    strat    = SOLReversionV2Strategy1H(**STRAT_PARAMS)
    df_sig   = strat.generate_signals(df)
    n_long   = int((df_sig["entry_setup"] ==  1).sum())
    n_short  = int((df_sig["entry_setup"] == -1).sum())

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
    peak         = result["equity"].cummax()
    max_dd       = ((result["equity"] - peak) / peak).min()
    total_return = (final_equity - CASH) / CASH * 100
    sol_return   = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    exit_split = stats.get("exit_reason_split", {})
    exits_str  = "  ".join(f"{k}:{v}" for k, v in sorted(exit_split.items()))

    print(f"  {label:<18}  "
          f"收益:{total_return:>+7.1f}%  "
          f"SOL:{sol_return:>+7.1f}%  "
          f"回撤:{max_dd*100:>6.1f}%  "
          f"成交:{stats.get('closed_trade_count', len(closed)):>3}  "
          f"胜率:{stats.get('win_rate',0)*100:>5.1f}%  "
          f"PF:{stats.get('profit_factor',0):>5.2f}  "
          f"信号[多{n_long}/空{n_short}]  "
          f"出场[{exits_str}]")

    return {
        "label":         label,
        "return_pct":    total_return,
        "sol_pct":       sol_return,
        "max_dd_pct":    max_dd * 100,
        "closed":        stats.get("closed_trade_count", len(closed)),
        "win_rate":      stats.get("win_rate", 0) * 100,
        "profit_factor": stats.get("profit_factor", 0),
        "n_long":        n_long,
        "n_short":       n_short,
    }


print("=" * 110)
print(f"  {'季度':<18}  {'策略收益':>8}  {'SOL涨跌':>8}  {'最大回撤':>7}  {'成交':>4}  {'胜率':>6}  {'PF':>5}  信号            出场分布")
print("=" * 110)

rows = []
for start, end, label in QUARTERS:
    r = run_quarter(start, end, label, df_all)
    if r:
        rows.append(r)

# ── 汇总统计 ──────────────────────────────────────────────
print("=" * 110)
if rows:
    positive  = [r for r in rows if r["return_pct"] > 0]
    negative  = [r for r in rows if r["return_pct"] <= 0]
    avg_ret   = np.mean([r["return_pct"] for r in rows])
    avg_dd    = np.mean([r["max_dd_pct"] for r in rows])
    avg_wr    = np.mean([r["win_rate"]   for r in rows])
    avg_pf    = np.mean([r["profit_factor"] for r in rows])
    total_closed = sum(r["closed"] for r in rows)

    print(f"\n  季度数量:    {len(rows)}")
    print(f"  盈利季度:    {len(positive)} / {len(rows)}")
    print(f"  亏损季度:    {len(negative)} / {len(rows)}")
    print(f"  平均季度收益: {avg_ret:+.2f}%")
    print(f"  平均最大回撤: {avg_dd:.2f}%")
    print(f"  平均胜率:    {avg_wr:.1f}%")
    print(f"  平均利润因子: {avg_pf:.2f}")
    print(f"  总成交笔数:  {total_closed}")

    if negative:
        print(f"\n  亏损季度明细:")
        for r in negative:
            print(f"    {r['label']:<18}  收益:{r['return_pct']:>+7.1f}%  "
                  f"回撤:{r['max_dd_pct']:>6.1f}%  PF:{r['profit_factor']:>.2f}")
