"""
分批止盈参数扫描
partial_take_R × partial_take_frac × break_even_after_partial
全程 2024-01-01 → 2026-03-24，SOL+PEPE 组合，新参数 SOL=3.0R / PEPE=4.0R
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

PARTIAL_RS    = [1.0, 1.5, 2.0]
PARTIAL_FRACS = [0.3, 0.5, 0.7]
BREAK_EVENS   = [False, True]


def run_coin(df, strat_params, bt_profile, max_pos, cash,
             partial_r, partial_frac, break_even):
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
        stop_atr=btp["stop_atr"], take_R=btp["take_R"],
        trail_start_R=0.0, trail_atr=0.0, use_trailing=False,
        check_liq=True, entry_is_maker=False,
        funding_rate_per_8h=config.FUNDING_RATE_PER_8H,
        risk_per_trade=btp["risk_per_trade"],
        enable_risk_position_sizing=True,
        allow_reentry=False,
        partial_take_R=partial_r,
        partial_take_frac=partial_frac,
        break_even_after_partial=break_even,
        break_even_R=0.0,
        use_signal_exit_targets=False,
        max_hold_bars=btp["max_hold_bars"],
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


def run_combo(df_sol, df_pepe, partial_r, partial_frac, break_even):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH,
                  partial_r, partial_frac, break_even)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH,
                  partial_r, partial_frac, break_even)
    eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_c = eq_s + eq_p
    dd_c   = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
    ret_c  = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    calmar = ret_c / abs(dd_c) if dd_c != 0 else 0.0
    return dict(
        ret_c=ret_c, dd_c=dd_c, calmar=calmar,
        ret_s=rs["ret"], dd_s=rs["dd"], wr_s=rs["wr"],
        ret_p=rp["ret"], dd_p=rp["dd"], wr_p=rp["wr"],
        nc=rs["nc"] + rp["nc"],
    )


# ── 拉数据 ───────────────────────────────────────────────
print("拉取数据...")
ds = CCXTDataSource()
df_sol  = ds.load_ohlcv("SOL/USDT:USDT",      "2024-01-01", "2026-03-24", "1h")
df_pepe = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"SOL {len(df_sol)} 根  PEPE {len(df_pepe)} 根\n")

# ── 基线 ─────────────────────────────────────────────────
print("运行基线...")
base = run_combo(df_sol, df_pepe, partial_r=0.0, partial_frac=0.0, break_even=False)

# ── 扫描 ─────────────────────────────────────────────────
combos = list(product(PARTIAL_RS, PARTIAL_FRACS, BREAK_EVENS))
results = []
for idx, (pr, pf, be) in enumerate(combos):
    print(f"  [{idx+1:>2}/{len(combos)}] partial_R={pr}  frac={pf}  break_even={be}...",
          end="\r", flush=True)
    r = run_combo(df_sol, df_pepe, pr, pf, be)
    results.append((pr, pf, be, r))

print(f"\n  扫描完成，共 {len(combos)} 组\n")
results.sort(key=lambda x: -x[3]["ret_c"])

W = 115
print("=" * W)
print("  分批止盈扫描  |  全程 2024-01-01 → 2026-03-24  |  SOL=3.0R / PEPE=4.0R")
print("=" * W)
print(f"  {'partial_R':>9} {'frac':>5} {'BE':>5} │ {'组合收益':>8} {'组合DD':>7} {'Calmar':>7} │ "
      f"{'SOL':>7} {'SOL DD':>7} │ {'PEPE':>8} {'PEPE DD':>8} │ {'笔数':>5} {'vs基线':>7}")
print("  " + "─" * (W - 2))

# 基线行
print(f"  {'★ 基线（无分批）':>30}  │ "
      f"{base['ret_c']:>+7.1f}% {base['dd_c']:>6.1f}% {base['calmar']:>7.2f} │ "
      f"{base['ret_s']:>+6.1f}% {base['dd_s']:>6.1f}% │ "
      f"{base['ret_p']:>+7.1f}% {base['dd_p']:>7.1f}% │ "
      f"{base['nc']:>4}笔")
print("  " + "─" * (W - 2))

for pr, pf, be, r in results:
    diff   = r["ret_c"] - base["ret_c"]
    be_str = "✓" if be else "✗"
    marker = " ◀" if r is results[0][3] else ""
    print(f"  {pr:>9.1f} {pf:>5.0%} {be_str:>5} │ "
          f"{r['ret_c']:>+7.1f}% {r['dd_c']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['ret_s']:>+6.1f}% {r['dd_s']:>6.1f}% │ "
          f"{r['ret_p']:>+7.1f}% {r['dd_p']:>7.1f}% │ "
          f"{r['nc']:>4}笔 {diff:>+6.1f}%{marker}")

print("\n  " + "─" * (W - 2))

# ── TOP 3 详细 ───────────────────────────────────────────
print(f"\n  TOP 3（按组合收益）vs 基线 {base['ret_c']:+.1f}%  DD {base['dd_c']:.1f}%  Calmar {base['calmar']:.2f}：")
for i, (pr, pf, be, r) in enumerate(results[:3]):
    diff_ret = r["ret_c"] - base["ret_c"]
    diff_dd  = r["dd_c"]  - base["dd_c"]
    print(f"\n  [{i+1}] partial_R={pr}R  出仓={pf:.0%}  止损移成本={be}")
    print(f"      组合收益: {r['ret_c']:>+.1f}%  (vs 基线 {diff_ret:>+.1f}%)")
    print(f"      最大回撤: {r['dd_c']:>.1f}%  (vs 基线 {diff_dd:>+.1f}%)")
    print(f"      Calmar:  {r['calmar']:.2f}  (vs 基线 {base['calmar']:.2f})")
    print(f"      SOL: {r['ret_s']:>+.1f}% / DD {r['dd_s']:.1f}%   "
          f"PEPE: {r['ret_p']:>+.1f}% / DD {r['dd_p']:.1f}%")
    print(f"      成交: {r['nc']}笔")

print(f"\n{'='*W}")
