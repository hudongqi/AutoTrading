"""
SOL + PEPE 组合策略季度拆分回测
使用最终优化参数，2024-01 → 2026-03 按季度观察组合表现
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

TOTAL_CASH = 10_000
HALF_CASH  = 5_000

# ── 最终优化参数 ──────────────────────────────────────────
SOL_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 38, "rsi_overbought": 62,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.2,
    "atr_pct_low": 0.003, "atr_pct_high": 0.10,
    "oversold_lookback": 3, "allow_short": True,
}
PEPE_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.2,
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
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


def run_coin(df, strat_params, bt_profile, max_pos, cash):
    if len(df) < 200:
        return None
    strat  = SOLReversionV2Strategy1H(**strat_params)
    df_sig = strat.generate_signals(df)
    btp    = get_sol_backtest_profile(bt_profile)
    port   = PerpPortfolio(initial_cash=cash, leverage=btp["leverage"],
                taker_fee_rate=config.TAKER_FEE_RATE, maker_fee_rate=config.MAKER_FEE_RATE,
                maint_margin_rate=0.005)
    bt = Backtester(
        broker=SimBroker(slippage_bps=config.SLIPPAGE_BPS),
        portfolio=port, strategy=strat,
        max_pos=max_pos, cooldown_bars=btp["cooldown_bars"],
        stop_atr=btp["stop_atr"], take_R=btp["take_R"],
        trail_start_R=0.0, trail_atr=0.0, use_trailing=False,
        check_liq=True, entry_is_maker=False,
        funding_rate_per_8h=config.FUNDING_RATE_PER_8H,
        risk_per_trade=btp["risk_per_trade"],
        enable_risk_position_sizing=True,
        allow_reentry=False, partial_take_R=0.0, partial_take_frac=0.0,
        break_even_after_partial=False, break_even_R=0.0,
        use_signal_exit_targets=False, max_hold_bars=btp["max_hold_bars"],
    )
    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])
    eq  = result["equity"]
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - cash) / cash * 100
    coin_ret = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    return dict(
        ret=ret, dd=dd, coin_ret=coin_ret, eq=eq,
        wr=stats.get("win_rate", 0) * 100,
        pf=stats.get("profit_factor", 0),
        nc=stats.get("closed_trade_count", len(closed)),
        nl=int((df_sig["entry_setup"] == 1).sum()),
        ns=int((df_sig["entry_setup"] == -1).sum()),
        es=stats.get("exit_reason_split", {}),
    )


# ── 预拉全量数据 ─────────────────────────────────────────
print("拉取全量数据...")
ds = CCXTDataSource()
df_sol_all  = ds.load_ohlcv("SOL/USDT:USDT",      start="2024-01-01", end="2026-03-24", timeframe="1h")
df_pepe_all = ds.load_ohlcv("1000PEPE/USDT:USDT", start="2024-01-01", end="2026-03-24", timeframe="1h")
print(f"SOL {len(df_sol_all)} 根  PEPE {len(df_pepe_all)} 根\n")

# ── 全程汇总（2024-01 → 2026-03-24 连续运行）──────────────
print("=" * 80)
print("  全程回测（2024-01-01 → 2026-03-24，连续运行，各 $5,000）")
print("=" * 80)
rs_all = run_coin(df_sol_all,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH)
rp_all = run_coin(df_pepe_all, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH)
eq_s = rs_all["eq"].reindex(rs_all["eq"].index.union(rp_all["eq"].index)).ffill()
eq_p = rp_all["eq"].reindex(rs_all["eq"].index.union(rp_all["eq"].index)).ffill()
eq_c = eq_s + eq_p
dd_all  = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
ret_all = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
print(f"  SOL  总收益: {rs_all['ret']:>+.2f}%   最大回撤: {rs_all['dd']:.2f}%   "
      f"胜率: {rs_all['wr']:.0f}%   成交: {rs_all['nc']}笔")
print(f"  PEPE 总收益: {rp_all['ret']:>+.2f}%   最大回撤: {rp_all['dd']:.2f}%   "
      f"胜率: {rp_all['wr']:.0f}%   成交: {rp_all['nc']}笔")
print(f"  组合 总收益: {ret_all:>+.2f}%   最大回撤: {dd_all:.2f}%   "
      f"总成交: {rs_all['nc']+rp_all['nc']}笔")
print()

# ── 表头 ────────────────────────────────────────────────
W = 120
print("=" * W)
print(f"  SOL + PEPE 组合策略季度拆分  |  总资金 $10,000（各 $5,000）")
print("=" * W)
hdr = (f"  {'季度':<12} │ {'SOL':>7} {'PEPE':>7} {'组合':>7} │ "
       f"{'SOL回撤':>7} {'PEPE回撤':>8} {'组合回撤':>8} │ "
       f"{'SOL PF':>6} {'PEPEPF':>7} │ "
       f"{'SOL成交':>7} {'PEPE成交':>8} │ "
       f"{'SOL标的':>8} {'PEPE标的':>9}")
print(hdr)
print("  " + "─" * (W - 2))

rows = []
cumulative_sol  = HALF_CASH
cumulative_pepe = HALF_CASH

for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_sol  = df_sol_all [(df_sol_all.index  >= ts) & (df_sol_all.index  < te)].copy()
    df_pepe = df_pepe_all[(df_pepe_all.index >= ts) & (df_pepe_all.index < te)].copy()

    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH)

    if rs is None and rp is None:
        continue

    # 合并权益曲线
    if rs and rp:
        eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_c = eq_s + eq_p
        dd_c = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
        ret_c = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    elif rs:
        ret_c, dd_c = rs["ret"] / 2, rs["dd"]
    else:
        ret_c, dd_c = rp["ret"] / 2, rp["dd"]

    sol_ret  = rs["ret"]  if rs else 0
    pepe_ret = rp["ret"]  if rp else 0
    sol_dd   = rs["dd"]   if rs else 0
    pepe_dd  = rp["dd"]   if rp else 0
    sol_pf   = rs["pf"]   if rs else 0
    pepe_pf  = rp["pf"]   if rp else 0
    sol_nc   = rs["nc"]   if rs else 0
    pepe_nc  = rp["nc"]   if rp else 0
    sol_coin = rs["coin_ret"]  if rs else 0
    pepe_coin= rp["coin_ret"] if rp else 0

    sign = "✓" if ret_c > 0 else "✗"
    spf  = f"{sol_pf:.2f}"  if sol_pf  != float("inf") else "∞"
    ppf  = f"{pepe_pf:.2f}" if pepe_pf != float("inf") else "∞"

    print(f"  {sign} {label:<10} │ "
          f"{sol_ret:>+6.1f}% {pepe_ret:>+6.1f}% {ret_c:>+6.1f}% │ "
          f"{sol_dd:>6.1f}%  {pepe_dd:>7.1f}%  {dd_c:>7.1f}% │ "
          f"{spf:>6}  {ppf:>6} │ "
          f"{sol_nc:>6}笔   {pepe_nc:>6}笔 │ "
          f"{sol_coin:>+7.1f}%  {pepe_coin:>+8.1f}%")

    rows.append(dict(label=label, sol=sol_ret, pepe=pepe_ret, comb=ret_c,
                     dd_s=sol_dd, dd_p=pepe_dd, dd_c=dd_c,
                     pf_s=sol_pf, pf_p=pepe_pf, nc_s=sol_nc, nc_p=pepe_nc))

print("  " + "─" * (W - 2))

# ── 汇总统计 ─────────────────────────────────────────────
valid = [r for r in rows]
pos_c = [r for r in valid if r["comb"] > 0]
neg_c = [r for r in valid if r["comb"] <= 0]

print(f"\n  季度总数:   {len(valid)}")
print(f"  盈利季度:   {len(pos_c)} / {len(valid)}  ({len(pos_c)/len(valid)*100:.0f}%)")
print(f"  亏损季度:   {len(neg_c)} / {len(valid)}")
print()
print(f"  平均组合季度收益:  {np.mean([r['comb'] for r in valid]):>+.2f}%")
print(f"  最佳季度:         {max(valid, key=lambda r: r['comb'])['label']}  "
      f"{max(r['comb'] for r in valid):>+.1f}%")
print(f"  最差季度:         {min(valid, key=lambda r: r['comb'])['label']}  "
      f"{min(r['comb'] for r in valid):>+.1f}%")
print()
print(f"  平均组合最大回撤:  {np.mean([r['dd_c'] for r in valid]):>.2f}%")
print(f"  最大单季回撤:      {min(r['dd_c'] for r in valid):>.2f}%")
print()
print(f"  总成交（SOL）:  {sum(r['nc_s'] for r in valid)} 笔")
print(f"  总成交（PEPE）: {sum(r['nc_p'] for r in valid)} 笔")
print(f"  合计:           {sum(r['nc_s']+r['nc_p'] for r in valid)} 笔")

if neg_c:
    print(f"\n  亏损季度明细:")
    for r in neg_c:
        print(f"    {r['label']:<14}  组合:{r['comb']:>+6.1f}%  "
              f"SOL:{r['sol']:>+6.1f}%  PEPE:{r['pepe']:>+6.1f}%  "
              f"组合回撤:{r['dd_c']:>6.1f}%")

print("\n" + "=" * W)
