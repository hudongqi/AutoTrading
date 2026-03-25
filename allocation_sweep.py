"""
SOL/PEPE 资金分配比例扫描
从 100/0 到 0/100，步长 10%
全程 2024-01-01 → 2026-03-24，新参数 SOL=3.0R / PEPE=4.0R
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
    "vol_period": 20, "vol_spike_mult": 1.5,
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
    if cash <= 0 or len(df) < 200:
        return None
    strat  = SOLReversionV2Strategy1H(**strat_params)
    df_sig = strat.generate_signals(df)
    btp    = get_sol_backtest_profile(bt_profile)
    port   = PerpPortfolio(
        initial_cash=cash, leverage=btp["leverage"],
        taker_fee_rate=config.TAKER_FEE_RATE, maker_fee_rate=config.MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
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
        dd_stop_pct=0.0, dd_cooldown_bars=0,
    )
    result = bt.run(df_sig)
    eq  = result["equity"]
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - cash) / cash * 100
    return dict(ret=ret, dd=dd, eq=eq, final=eq.iloc[-1])


def run_alloc(df_sol, df_pepe, sol_pct):
    cash_sol  = TOTAL_CASH * sol_pct
    cash_pepe = TOTAL_CASH * (1 - sol_pct)
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       cash_sol)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, cash_pepe)

    # 合并权益
    if rs and rp:
        eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_c = eq_s + eq_p
    elif rs:
        eq_c = rs["eq"]
    else:
        eq_c = rp["eq"]

    dd_c   = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
    ret_c  = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    calmar = ret_c / abs(dd_c) if dd_c != 0 else 0.0
    return dict(
        ret_c=ret_c, dd_c=dd_c, calmar=calmar,
        ret_s=rs["ret"] if rs else 0.0,
        ret_p=rp["ret"] if rp else 0.0,
        dd_s=rs["dd"]  if rs else 0.0,
        dd_p=rp["dd"]  if rp else 0.0,
    )


def run_alloc_quarterly(df_sol, df_pepe, sol_pct):
    cash_sol  = TOTAL_CASH * sol_pct
    cash_pepe = TOTAL_CASH * (1 - sol_pct)
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       cash_sol)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, cash_pepe)

    if rs is None and rp is None:
        return None
    if rs and rp:
        eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_c = eq_s + eq_p
    elif rs:
        eq_c = rs["eq"]
    else:
        eq_c = rp["eq"]

    dd_c  = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
    ret_c = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    return dict(ret=ret_c, dd=dd_c)


# ── 拉数据 ───────────────────────────────────────────────
print("拉取数据...")
ds = CCXTDataSource()
df_sol_all  = ds.load_ohlcv("SOL/USDT:USDT",      "2024-01-01", "2026-03-24", "1h")
df_pepe_all = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"SOL {len(df_sol_all)} 根  PEPE {len(df_pepe_all)} 根\n")

# ── 全程扫描 ─────────────────────────────────────────────
allocs = [i / 10 for i in range(0, 11)]   # 0.0 ~ 1.0
full_results = []

print("全程扫描...")
for sol_pct in allocs:
    label = f"SOL {sol_pct:.0%} / PEPE {1-sol_pct:.0%}"
    print(f"  {label}...", end="\r", flush=True)
    r = run_alloc(df_sol_all, df_pepe_all, sol_pct)
    full_results.append((sol_pct, r))

W = 95
print(f"\n\n{'='*W}")
print("  资金分配扫描  |  全程 2024-01-01 → 2026-03-24  |  SOL=3.0R / PEPE=4.0R")
print(f"{'='*W}")
print(f"  {'SOL%':>5} {'PEPE%':>6} │ {'组合收益':>8} {'组合DD':>7} {'Calmar':>7} │ "
      f"{'SOL收益':>8} {'SOL DD':>7} │ {'PEPE收益':>9} {'PEPE DD':>8}")
print("  " + "─" * (W - 2))

base_r = next(r for sp, r in full_results if sp == 0.5)
best_r = max(full_results, key=lambda x: x[1]["ret_c"])

for sol_pct, r in full_results:
    is_base = abs(sol_pct - 0.5) < 0.01
    is_best = sol_pct == best_r[0]
    mark = " ★基线" if is_base else (" ◀最优" if is_best else "")
    diff = f" ({r['ret_c']-base_r['ret_c']:+.1f}%)" if not is_base else ""
    print(f"  {sol_pct:>4.0%}  {1-sol_pct:>5.0%} │ "
          f"{r['ret_c']:>+7.1f}%{diff:<9} {r['dd_c']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['ret_s']:>+7.1f}% {r['dd_s']:>6.1f}% │ "
          f"{r['ret_p']:>+8.1f}% {r['dd_p']:>7.1f}%{mark}")

# ── 季度拆分：基线 vs 最优分配 ──────────────────────────
best_pct = best_r[0]
print(f"\n\n{'='*W}")
print(f"  季度拆分对比：基线 50/50 vs 最优 SOL {best_pct:.0%}/PEPE {1-best_pct:.0%}")
print(f"{'='*W}")
print(f"  {'季度':<14} {'50/50收益':>9} {'最优收益':>9} {'差值':>7} │ "
      f"{'50/50 DD':>9} {'最优DD':>8}")
print("  " + "─" * (W - 2))

rows_base, rows_best = [], []
for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_s = df_sol_all [(df_sol_all.index  >= ts) & (df_sol_all.index  < te)].copy()
    df_p = df_pepe_all[(df_pepe_all.index >= ts) & (df_pepe_all.index < te)].copy()

    rb = run_alloc_quarterly(df_s, df_p, 0.5)
    ro = run_alloc_quarterly(df_s, df_p, best_pct)

    _z = dict(ret=0, dd=0)
    rb = rb or _z
    ro = ro or _z

    diff = ro["ret"] - rb["ret"]
    mark = "↑" if diff > 0.3 else ("↓" if diff < -0.3 else "─")
    print(f"  {mark} {label:<12} {rb['ret']:>+8.2f}% {ro['ret']:>+8.2f}% {diff:>+6.2f}% │ "
          f"{rb['dd']:>8.2f}% {ro['dd']:>7.2f}%")
    rows_base.append(rb)
    rows_best.append(ro)

print("  " + "─" * (W - 2))

def summary(rows):
    rets = [r["ret"] for r in rows]
    dds  = [r["dd"]  for r in rows]
    return dict(avg_ret=np.mean(rets), win_q=sum(1 for r in rets if r > 0),
                avg_dd=np.mean(dds), worst_dd=min(dds))

sb = summary(rows_base)
so = summary(rows_best)
print(f"\n  {'指标':<16} {'50/50':>10} {f'SOL{best_pct:.0%}/PEPE{1-best_pct:.0%}':>12} │ {'变化':>10}")
print(f"  {'─'*55}")
for label, vb, vo, d, higher_good, kind in [
    ("盈利季度",     f"{sb['win_q']}/9",         f"{so['win_q']}/9",
     so['win_q']-sb['win_q'],       True,  "q"),
    ("平均季度收益", f"{sb['avg_ret']:>+.2f}%",  f"{so['avg_ret']:>+.2f}%",
     so['avg_ret']-sb['avg_ret'],   True,  "pct"),
    ("平均季度回撤", f"{sb['avg_dd']:.2f}%",     f"{so['avg_dd']:.2f}%",
     so['avg_dd']-sb['avg_dd'],     False, "pct"),
    ("最大单季回撤", f"{sb['worst_dd']:.2f}%",   f"{so['worst_dd']:.2f}%",
     so['worst_dd']-sb['worst_dd'], False, "pct"),
]:
    if kind == "pct":
        d_str = ("  ─" if abs(d) < 0.01 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+.2f}%").rjust(12)
    else:
        d_str = ("  ─" if d == 0 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+d}").rjust(12)
    print(f"  {label:<16} {vb:>10} {vo:>12} │  {d_str}")

print(f"\n{'='*W}")
