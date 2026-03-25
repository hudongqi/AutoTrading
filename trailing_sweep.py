"""
追踪止损参数扫描
对比：无追踪（基线）vs trail_start_R × trail_atr × take_R 组合
全程 2024-01-01 → 2026-03-24，SOL+PEPE 组合
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

# 扫描空间
TRAIL_STARTS = [1.0, 1.5, 2.0]
TRAIL_ATRS   = [1.5, 2.0, 2.5, 3.0]
# take_R: (SOL, PEPE) 两档
TAKE_R_SETS  = [
    (2.5, 3.0, "当前"),    # 基线 take_R
    (5.0, 6.0, "放宽"),    # 让追踪主导退出
]


def run_coin(df, strat_params, bt_profile, max_pos, cash,
             take_r, trail_start, trail_atr_mult, use_trailing):
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
        trail_start_R=trail_start, trail_atr=trail_atr_mult,
        use_trailing=use_trailing,
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
    nc     = len(bt.closed_trades)
    wr     = stats.get("win_rate", 0) * 100
    return dict(ret=ret, dd=dd, nc=nc, wr=wr, eq=eq)


def run_combo(df_sol, df_pepe, take_sol, take_pepe,
              trail_start, trail_atr_mult, use_trailing):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH,
                  take_sol, trail_start, trail_atr_mult, use_trailing)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH,
                  take_pepe, trail_start, trail_atr_mult, use_trailing)
    eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_c = eq_s + eq_p
    dd_c  = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
    ret_c = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    calmar = ret_c / abs(dd_c) if dd_c != 0 else 0.0
    return dict(
        ret_c=ret_c, dd_c=dd_c, calmar=calmar,
        ret_s=rs["ret"], ret_p=rp["ret"],
        dd_s=rs["dd"],   dd_p=rp["dd"],
        wr_s=rs["wr"],   wr_p=rp["wr"],
        nc_s=rs["nc"],   nc_p=rp["nc"],
    )


# ── 拉数据 ───────────────────────────────────────────────
print("拉取数据...")
ds = CCXTDataSource()
df_sol  = ds.load_ohlcv("SOL/USDT:USDT",      "2024-01-01", "2026-03-24", "1h")
df_pepe = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"SOL {len(df_sol)} 根  PEPE {len(df_pepe)} 根\n")

# ── 基线（无追踪）────────────────────────────────────────
print("运行基线（无追踪止损）...")
base = run_combo(df_sol, df_pepe, 2.5, 3.0, 0.0, 0.0, False)

W = 120
print(f"\n{'='*W}")
print(f"  追踪止损参数扫描  |  全程 2024-01-01 → 2026-03-24  |  SOL+PEPE 组合")
print(f"{'='*W}")
print(f"  {'配置':<30} {'take_R':>8} │ {'组合收益':>8} {'组合DD':>7} {'Calmar':>7} │ "
      f"{'SOL':>7} {'PEPE':>7} │ {'SOL DD':>7} {'PEPEDD':>7} │ {'笔数':>5}")
print(f"  {'─'*(W-2)}")

# 基线行
print(f"  {'★ 基线（无追踪）':<30} {'2.5/3.0':>8} │ "
      f"{base['ret_c']:>+7.1f}% {base['dd_c']:>6.1f}% {base['calmar']:>7.2f} │ "
      f"{base['ret_s']:>+6.1f}% {base['ret_p']:>+6.1f}% │ "
      f"{base['dd_s']:>6.1f}% {base['dd_p']:>6.1f}% │ "
      f"{base['nc_s']+base['nc_p']:>4}笔")
print(f"  {'─'*(W-2)}")

# ── 扫描 ─────────────────────────────────────────────────
all_results = []

total = len(TRAIL_STARTS) * len(TRAIL_ATRS) * len(TAKE_R_SETS)
done  = 0
for ts_r, ta_r, (tk_s, tk_p, tk_label) in product(TRAIL_STARTS, TRAIL_ATRS, TAKE_R_SETS):
    done += 1
    print(f"  [{done:>2}/{total}] trail_start={ts_r} trail_atr={ta_r} take={tk_label}...",
          end="\r", flush=True)
    r = run_combo(df_sol, df_pepe, tk_s, tk_p, ts_r, ta_r, True)
    all_results.append((ts_r, ta_r, tk_s, tk_p, tk_label, r))

print(f"\n  扫描完成，共 {total} 组配置\n")

# 按组合收益降序排列
all_results.sort(key=lambda x: -x[5]["ret_c"])

for ts_r, ta_r, tk_s, tk_p, tk_label, r in all_results:
    diff = r["ret_c"] - base["ret_c"]
    marker = " ◀" if r == all_results[0][5] else ""
    label  = f"start={ts_r}R trail={ta_r}ATR"
    print(f"  {label:<30} {f'{tk_s}/{tk_p}':>8} │ "
          f"{r['ret_c']:>+7.1f}% {r['dd_c']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['ret_s']:>+6.1f}% {r['ret_p']:>+6.1f}% │ "
          f"{r['dd_s']:>6.1f}% {r['dd_p']:>6.1f}% │ "
          f"{r['nc_s']+r['nc_p']:>4}笔  {diff:>+5.1f}%{marker}")

print(f"\n  {'─'*(W-2)}")

# ── TOP 5 详细 ───────────────────────────────────────────
print(f"\n  TOP 5 配置（按组合收益）vs 基线 {base['ret_c']:+.1f}%：")
for i, (ts_r, ta_r, tk_s, tk_p, tk_label, r) in enumerate(all_results[:5]):
    diff_ret = r["ret_c"] - base["ret_c"]
    diff_dd  = r["dd_c"]  - base["dd_c"]
    print(f"\n  [{i+1}] trail_start={ts_r}R  trail_atr={ta_r}ATR  take={tk_s}/{tk_p}R ({tk_label})")
    print(f"      组合收益: {r['ret_c']:>+.1f}%  (vs 基线 {diff_ret:>+.1f}%)")
    print(f"      最大回撤: {r['dd_c']:>.1f}%  (vs 基线 {diff_dd:>+.1f}%)")
    print(f"      Calmar:  {r['calmar']:.2f}  (vs 基线 {base['calmar']:.2f})")
    print(f"      SOL: {r['ret_s']:>+.1f}% / {r['dd_s']:.1f}%  PEPE: {r['ret_p']:>+.1f}% / {r['dd_p']:.1f}%")
    print(f"      成交: {r['nc_s']+r['nc_p']}笔")

print(f"\n{'='*W}")
