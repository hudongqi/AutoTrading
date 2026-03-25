"""
take_R 参数扫描：SOL × PEPE 组合
全程 2024-01-01 → 2026-03-24
基线：SOL=2.5R / PEPE=3.0R
"""

import pandas as pd
import numpy as np
from itertools import product

from data import CCXTDataSource
from strategy import SOLReversionV2Strategy1H
from strategy_profiles import get_sol_backtest_profile
from backtest import Backtester
from portfolio import PerpPortfolio
from broker import SimBroker
import config

TOTAL_CASH = 10_000
HALF_CASH  = 5_000

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

SOL_TAKE_RS  = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
PEPE_TAKE_RS = [2.5, 3.0, 3.5, 4.0]


def run_coin(df, strat_params, bt_profile, max_pos, cash, take_r):
    if len(df) < 200:
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
        stop_atr=btp["stop_atr"], take_R=take_r,
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
    stats  = result.attrs.get("stats", {})
    eq     = result["equity"]
    dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret    = (eq.iloc[-1] - cash) / cash * 100
    wr     = stats.get("win_rate", 0) * 100
    nc     = len(bt.closed_trades)
    return dict(ret=ret, dd=dd, wr=wr, nc=nc, eq=eq)


def run_combo(df_sol, df_pepe, take_sol, take_pepe):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH, take_sol)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH, take_pepe)
    eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_c = eq_s + eq_p
    dd_c  = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
    ret_c = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    calmar = ret_c / abs(dd_c) if dd_c != 0 else 0.0
    return dict(
        ret_c=ret_c, dd_c=dd_c, calmar=calmar,
        ret_s=rs["ret"], wr_s=rs["wr"], nc_s=rs["nc"], dd_s=rs["dd"],
        ret_p=rp["ret"], wr_p=rp["wr"], nc_p=rp["nc"], dd_p=rp["dd"],
    )


# ── 拉数据 ───────────────────────────────────────────────
print("拉取数据...")
ds = CCXTDataSource()
df_sol  = ds.load_ohlcv("SOL/USDT:USDT",      "2024-01-01", "2026-03-24", "1h")
df_pepe = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"SOL {len(df_sol)} 根  PEPE {len(df_pepe)} 根\n")

# ── 扫描 ─────────────────────────────────────────────────
combos = list(product(SOL_TAKE_RS, PEPE_TAKE_RS))
total  = len(combos)
results = []

for idx, (ts, tp) in enumerate(combos):
    print(f"  [{idx+1:>2}/{total}] SOL take={ts}R  PEPE take={tp}R ...", end="\r", flush=True)
    r = run_combo(df_sol, df_pepe, ts, tp)
    results.append((ts, tp, r))

print(f"\n  扫描完成，共 {total} 组\n")

results.sort(key=lambda x: -x[2]["ret_c"])

BASE_SOL, BASE_PEPE = 2.5, 3.0
base = next(r for ts, tp, r in results if ts == BASE_SOL and tp == BASE_PEPE)

W = 115
print("=" * W)
print("  take_R 扫描  |  全程 2024-01-01 → 2026-03-24  |  SOL+PEPE 组合")
print("=" * W)
print(f"  {'SOL':>5} {'PEPE':>5} │ {'组合收益':>8} {'组合DD':>7} {'Calmar':>7} │ "
      f"{'SOL收益':>8} {'SOL胜率':>7} {'SOLDD':>6} │ "
      f"{'PEPE收益':>9} {'PEPE胜率':>8} {'PEPEDD':>7} │ {'vs基线':>7}")
print("  " + "─" * (W - 2))

for ts, tp, r in results:
    is_base = (ts == BASE_SOL and tp == BASE_PEPE)
    diff    = r["ret_c"] - base["ret_c"]
    mark    = " ★基线" if is_base else (" ◀最优" if r is results[0][2] else "")
    print(f"  {ts:>5.1f} {tp:>5.1f} │ "
          f"{r['ret_c']:>+7.1f}% {r['dd_c']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['ret_s']:>+7.1f}% {r['wr_s']:>6.0f}%  {r['dd_s']:>5.1f}% │ "
          f"{r['ret_p']:>+8.1f}% {r['wr_p']:>7.0f}%  {r['dd_p']:>6.1f}% │ "
          f"{diff:>+6.1f}%{mark}")

print("\n  " + "─" * (W - 2))

# ── SOL 单独分析（固定 PEPE=3.0）───────────────────────
print(f"\n  SOL take_R 单独影响（PEPE 固定 3.0R）：")
print(f"  {'SOL take_R':>12} │ {'SOL收益':>8} {'SOL胜率':>7} {'SOL DD':>7} │ {'组合收益':>8} {'vs基线':>7}")
print("  " + "─" * 65)
for ts, tp, r in sorted([(ts, tp, r) for ts, tp, r in results if tp == BASE_PEPE],
                         key=lambda x: x[0]):
    is_base = (ts == BASE_SOL)
    diff = r["ret_c"] - base["ret_c"]
    mark = " ★" if is_base else ""
    print(f"  {ts:>12.1f} │ {r['ret_s']:>+7.1f}% {r['wr_s']:>6.0f}%  {r['dd_s']:>6.1f}% │ "
          f"{r['ret_c']:>+7.1f}% {diff:>+6.1f}%{mark}")

# ── PEPE 单独分析（固定 SOL=2.5）───────────────────────
print(f"\n  PEPE take_R 单独影响（SOL 固定 2.5R）：")
print(f"  {'PEPE take_R':>12} │ {'PEPE收益':>9} {'PEPE胜率':>8} {'PEPEDD':>7} │ {'组合收益':>8} {'vs基线':>7}")
print("  " + "─" * 65)
for ts, tp, r in sorted([(ts, tp, r) for ts, tp, r in results if ts == BASE_SOL],
                         key=lambda x: x[1]):
    is_base = (tp == BASE_PEPE)
    diff = r["ret_c"] - base["ret_c"]
    mark = " ★" if is_base else ""
    print(f"  {tp:>12.1f} │ {r['ret_p']:>+8.1f}% {r['wr_p']:>7.0f}%  {r['dd_p']:>6.1f}% │ "
          f"{r['ret_c']:>+7.1f}% {diff:>+6.1f}%{mark}")

print(f"\n  最优组合：SOL={results[0][0]}R / PEPE={results[0][1]}R  →  "
      f"组合 {results[0][2]['ret_c']:+.1f}%  DD {results[0][2]['dd_c']:.1f}%  "
      f"Calmar {results[0][2]['calmar']:.2f}")
print(f"  基线对比：SOL={BASE_SOL}R / PEPE={BASE_PEPE}R  →  "
      f"组合 {base['ret_c']:+.1f}%  DD {base['dd_c']:.1f}%  "
      f"Calmar {base['calmar']:.2f}")
print("=" * W)
