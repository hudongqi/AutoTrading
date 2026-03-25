"""
ETH/USDT 策略验证
用与 SOL 完全相同的参数，分别跑：
  1. 年度对比（2023 / 2024 / 2025-26）
  2. 2024-26 季度拆分（9 段）
目的：验证策略逻辑是否对 ETH 同样适用（out-of-sample 品种验证）
"""

import pandas as pd
import numpy as np

from data import CCXTDataSource
from strategy import SOLReversionV2Strategy1H   # 策略逻辑与品种无关，直接复用
from strategy_profiles import get_sol_backtest_profile
from backtest import Backtester
from portfolio import PerpPortfolio
from broker import SimBroker
import config

SYMBOL = "ETH/USDT:USDT"
CASH   = 10_000

# ── 与 SOL 完全相同的策略参数，一字不改 ────────────────────
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

ANNUAL_PERIODS = [
    ("2023-01-01", "2024-01-01", "2023全年（牛市）"),
    ("2024-01-01", "2025-01-01", "2024全年（牛转熊）"),
    ("2025-01-01", "2026-03-24", "2025-26（对比区间）"),
]

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


def run_single(df, label, eth_ref_start=None, eth_ref_end=None):
    """对给定 DataFrame 运行一次回测，返回结果字典"""
    if len(df) < 300:
        print(f"  {label:<22}  数据不足（{len(df)} 根），跳过")
        return None

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
        broker=broker, portfolio=portfolio, strategy=strat,
        max_pos=bt_params["max_pos"], cooldown_bars=bt_params["cooldown_bars"],
        stop_atr=bt_params["stop_atr"], take_R=bt_params["take_R"],
        trail_start_R=bt_params["trail_start_R"], trail_atr=bt_params["trail_atr"],
        use_trailing=bt_params["use_trailing"], check_liq=True,
        entry_is_maker=bt_params["entry_is_maker"],
        funding_rate_per_8h=config.FUNDING_RATE_PER_8H,
        risk_per_trade=bt_params["risk_per_trade"],
        enable_risk_position_sizing=bt_params["enable_risk_position_sizing"],
        allow_reentry=bt_params["allow_reentry"],
        partial_take_R=bt_params["partial_take_R"],
        partial_take_frac=bt_params["partial_take_frac"],
        break_even_after_partial=bt_params["break_even_after_partial"],
        break_even_R=bt_params["break_even_R"],
        use_signal_exit_targets=bt_params["use_signal_exit_targets"],
        max_hold_bars=bt_params["max_hold_bars"],
    )

    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])

    final_equity = result["equity"].iloc[-1]
    peak         = result["equity"].cummax()
    max_dd       = ((result["equity"] - peak) / peak).min()
    total_return = (final_equity - CASH) / CASH * 100
    eth_return   = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    exit_split   = stats.get("exit_reason_split", {})

    return {
        "label":         label,
        "return_pct":    total_return,
        "eth_pct":       eth_return,
        "max_dd_pct":    max_dd * 100,
        "closed":        stats.get("closed_trade_count", len(closed)),
        "win_rate":      stats.get("win_rate", 0) * 100,
        "pnl_ratio":     stats.get("pnl_ratio", 0),
        "profit_factor": stats.get("profit_factor", 0),
        "expectancy":    stats.get("expectancy_per_trade", 0),
        "fees":          stats.get("total_fees", 0),
        "n_long":        n_long,
        "n_short":       n_short,
        "exit_split":    exit_split,
    }


def print_annual(r):
    exits = "  ".join(f"{k}:{v}" for k, v in sorted(r["exit_split"].items()))
    print(f"\n  {r['label']}")
    print(f"    收益: {r['return_pct']:>+7.2f}%   ETH同期: {r['eth_pct']:>+8.2f}%   最大回撤: {r['max_dd_pct']:>7.2f}%")
    print(f"    成交: {r['closed']}  胜率: {r['win_rate']:.1f}%  盈亏比: {r['pnl_ratio']:.2f}  利润因子: {r['profit_factor']:.2f}")
    print(f"    期望/笔: {r['expectancy']:.2f} USDT  手续费: {r['fees']:.2f} USDT  信号[多{r['n_long']}/空{r['n_short']}]")
    print(f"    出场: {exits}")


def print_quarterly(r):
    exits = "  ".join(f"{k}:{v}" for k, v in sorted(r["exit_split"].items()))
    sign  = "✓" if r["return_pct"] > 0 else "✗"
    print(f"  {sign} {r['label']:<18}  "
          f"收益:{r['return_pct']:>+7.1f}%  "
          f"ETH:{r['eth_pct']:>+7.1f}%  "
          f"回撤:{r['max_dd_pct']:>6.1f}%  "
          f"成交:{r['closed']:>3}  "
          f"胜率:{r['win_rate']:>5.1f}%  "
          f"PF:{r['profit_factor']:>5.2f}  "
          f"[多{r['n_long']}/空{r['n_short']}]  [{exits}]")


# ══════════════════════════════════════════════════════════════
#  第一部分：年度对比
# ══════════════════════════════════════════════════════════════
ds = CCXTDataSource()

print("=" * 70)
print("  ETH/USDT 策略验证（参数与 SOL 完全相同）")
print("=" * 70)
print("\n【第一部分】年度对比\n")

annual_rows = []
for start, end, label in ANNUAL_PERIODS:
    print(f"  拉取 {label} 数据...")
    df = ds.load_ohlcv(symbol=SYMBOL, start=start, end=end, timeframe="1h")
    r  = run_single(df, label)
    if r:
        print_annual(r)
        annual_rows.append(r)

print("\n  ── 年度汇总 " + "─" * 48)
print(f"  {'区间':<22} {'收益':>8} {'ETH涨跌':>9} {'最大回撤':>9} {'成交':>5} {'胜率':>7} {'利润因子':>8}")
print("  " + "-" * 68)
for r in annual_rows:
    pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] != float("inf") else "∞"
    print(f"  {r['label']:<22} {r['return_pct']:>+7.1f}%  {r['eth_pct']:>+8.1f}%  "
          f"{r['max_dd_pct']:>8.1f}%  {r['closed']:>4}  {r['win_rate']:>6.1f}%  {pf_str:>8}")

# ══════════════════════════════════════════════════════════════
#  第二部分：季度拆分
# ══════════════════════════════════════════════════════════════
print("\n\n【第二部分】2024-26 季度拆分\n")
print("  预先拉取 2024-01-01 → 2026-03-24 全量数据...")
df_all = ds.load_ohlcv(symbol=SYMBOL, start="2024-01-01", end="2026-03-24", timeframe="1h")
print(f"  共 {len(df_all)} 根 K 线\n")

print("  " + "=" * 100)
print(f"  {'':2} {'季度':<18}  {'策略收益':>8}  {'ETH涨跌':>8}  {'最大回撤':>7}  {'成交':>4}  {'胜率':>6}  {'PF':>5}  信号 & 出场")
print("  " + "=" * 100)

q_rows = []
for start, end, label in QUARTERS:
    df = df_all[
        (df_all.index >= pd.Timestamp(start, tz="UTC")) &
        (df_all.index <  pd.Timestamp(end,   tz="UTC"))
    ].copy()
    r = run_single(df, label)
    if r:
        print_quarterly(r)
        q_rows.append(r)

print("  " + "=" * 100)

# 季度汇总
if q_rows:
    valid = [r for r in q_rows if r["closed"] > 0]
    pos   = [r for r in valid  if r["return_pct"] > 0]
    neg   = [r for r in valid  if r["return_pct"] <= 0]
    pfs   = [r["profit_factor"] for r in valid if r["profit_factor"] not in (float("inf"), float("nan"))]

    print(f"\n  季度盈利: {len(pos)}/{len(valid)}  亏损: {len(neg)}/{len(valid)}")
    print(f"  平均季度收益: {np.mean([r['return_pct'] for r in valid]):+.2f}%")
    print(f"  平均最大回撤: {np.mean([r['max_dd_pct'] for r in valid]):.2f}%")
    print(f"  平均胜率:    {np.mean([r['win_rate'] for r in valid]):.1f}%")
    if pfs:
        print(f"  平均利润因子: {np.mean(pfs):.2f}")
    print(f"  总成交笔数:  {sum(r['closed'] for r in valid)}")

# ══════════════════════════════════════════════════════════════
#  第三部分：SOL vs ETH 横向对比摘要
# ══════════════════════════════════════════════════════════════
SOL_ANNUAL = {
    "2023全年（牛市）":       (70.6, -20.5, 2.45),
    "2024全年（牛转熊）":     ( 8.1, -17.0, 1.14),
    "2025-26（对比区间）":    (54.3, -17.5, 1.88),
}

print("\n\n【第三部分】SOL vs ETH 横向对比（相同参数）\n")
print(f"  {'区间':<22} {'SOL收益':>8} {'SOL回撤':>8} {'SOL PF':>7}  │  {'ETH收益':>8} {'ETH回撤':>8} {'ETH PF':>7}")
print("  " + "-" * 80)
for r in annual_rows:
    sol = SOL_ANNUAL.get(r["label"], (0, 0, 0))
    pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] not in (float("inf"),) else "∞"
    print(f"  {r['label']:<22} {sol[0]:>+7.1f}%  {sol[1]:>7.1f}%  {sol[2]:>6.2f}  │  "
          f"{r['return_pct']:>+7.1f}%  {r['max_dd_pct']:>7.1f}%  {pf_str:>7}")
