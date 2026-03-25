"""
B: 参数敏感性分析
对 vol_spike_mult / take_R / stop_atr 各自在最优值附近扫描
同时做 vol_spike × take_R 二维热力图
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
BASE = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.2,
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
}
OPT_VOL   = 1.2
OPT_TAKE  = 4.0
OPT_STOP  = 5.0
OPT_CD    = 3

VOL_RANGE  = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
TAKE_RANGE = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
STOP_RANGE = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def run(df, params, cd, stop_atr, take_r):
    strat  = SOLReversionV2Strategy1H(**params)
    df_sig = strat.generate_signals(df)
    btp    = get_sol_backtest_profile("pepe_rev_v2")
    port   = PerpPortfolio(
        initial_cash=TOTAL_CASH, leverage=btp["leverage"],
        taker_fee_rate=config.TAKER_FEE_RATE, maker_fee_rate=config.MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=config.SLIPPAGE_BPS),
        portfolio=port, strategy=strat,
        max_pos=5_000_000, cooldown_bars=cd,
        stop_atr=stop_atr, take_R=take_r,
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
    eq = result["equity"]
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    nc  = len(bt.closed_trades)
    calmar = ret / abs(dd) if dd != 0 else 0
    return dict(ret=ret, dd=dd, calmar=calmar, nc=nc)

print("拉取 PEPE 数据...")
ds = CCXTDataSource()
df = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"PEPE {len(df)} 根\n")

base_r = run(df, BASE, OPT_CD, OPT_STOP, OPT_TAKE)

W = 80
# ── 1. vol_spike_mult 单独敏感性 ──────────────────────────
print("=" * W)
print("  [1/4] vol_spike_mult 敏感性（take_R=4.0 / stop_atr=5.0 固定）")
print(f"  最优值：{OPT_VOL}  基线收益：{base_r['ret']:+.1f}%  Calmar {base_r['calmar']:.2f}")
print("=" * W)
print(f"  {'vol_mult':>9} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'成交':>5} │ {'vs最优':>7}  {'稳健性'}")
print("  " + "─" * (W - 2))
for v in VOL_RANGE:
    p = {**BASE, "vol_spike_mult": v}
    r = run(df, p, OPT_CD, OPT_STOP, OPT_TAKE)
    diff = r["ret"] - base_r["ret"]
    mark = " ◀最优" if abs(v - OPT_VOL) < 0.01 else ""
    # 稳健性：收益在最优的80%以上为稳健
    robust = "✓ 稳健" if r["ret"] >= base_r["ret"] * 0.8 else ("△ 尚可" if r["ret"] >= base_r["ret"] * 0.5 else "✗ 敏感")
    print(f"  {v:>9.1f} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['nc']:>4}笔 │ {diff:>+6.1f}%  {robust}{mark}")

# ── 2. take_R 单独敏感性 ──────────────────────────────────
print(f"\n{'='*W}")
print("  [2/4] take_R 敏感性（vol_spike=1.2 / stop_atr=5.0 固定）")
print(f"  最优值：{OPT_TAKE}R  基线收益：{base_r['ret']:+.1f}%  Calmar {base_r['calmar']:.2f}")
print("=" * W)
print(f"  {'take_R':>8} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'成交':>5} │ {'vs最优':>7}  {'稳健性'}")
print("  " + "─" * (W - 2))
for t in TAKE_RANGE:
    r = run(df, BASE, OPT_CD, OPT_STOP, t)
    diff = r["ret"] - base_r["ret"]
    mark = " ◀最优" if abs(t - OPT_TAKE) < 0.01 else ""
    robust = "✓ 稳健" if r["ret"] >= base_r["ret"] * 0.8 else ("△ 尚可" if r["ret"] >= base_r["ret"] * 0.5 else "✗ 敏感")
    print(f"  {t:>8.1f} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['nc']:>4}笔 │ {diff:>+6.1f}%  {robust}{mark}")

# ── 3. stop_atr 单独敏感性 ────────────────────────────────
print(f"\n{'='*W}")
print("  [3/4] stop_atr 敏感性（vol_spike=1.2 / take_R=4.0 固定）")
print(f"  最优值：{OPT_STOP}  基线收益：{base_r['ret']:+.1f}%  Calmar {base_r['calmar']:.2f}")
print("=" * W)
print(f"  {'stop_atr':>9} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'成交':>5} │ {'vs最优':>7}  {'稳健性'}")
print("  " + "─" * (W - 2))
for s in STOP_RANGE:
    r = run(df, BASE, OPT_CD, s, OPT_TAKE)
    diff = r["ret"] - base_r["ret"]
    mark = " ◀最优" if abs(s - OPT_STOP) < 0.01 else ""
    robust = "✓ 稳健" if r["ret"] >= base_r["ret"] * 0.8 else ("△ 尚可" if r["ret"] >= base_r["ret"] * 0.5 else "✗ 敏感")
    print(f"  {s:>9.1f} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['nc']:>4}笔 │ {diff:>+6.1f}%  {robust}{mark}")

# ── 4. vol_spike × take_R 二维热力图 ─────────────────────
print(f"\n{'='*W}")
print("  [4/4] vol_spike × take_R 二维热力图（Calmar，stop_atr=5.0 固定）")
print(f"{'='*W}")
VOL_2D  = [0.8, 1.0, 1.2, 1.5, 2.0]
TAKE_2D = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

grid = {}
total = len(VOL_2D) * len(TAKE_2D)
done  = 0
for v, t in product(VOL_2D, TAKE_2D):
    done += 1
    print(f"  [{done:>2}/{total}] vol={v} take={t}...", end="\r", flush=True)
    p = {**BASE, "vol_spike_mult": v}
    r = run(df, p, OPT_CD, OPT_STOP, t)
    grid[(v, t)] = r["calmar"]

print(f"\n")
# 表头
header = f"  {'vol↓ take→':>12}" + "".join(f"  {t:>5.1f}R" for t in TAKE_2D)
print(header)
print("  " + "─" * (len(header) - 2))
for v in VOL_2D:
    row = f"  vol={v:>4.1f}     "
    for t in TAKE_2D:
        cal = grid[(v, t)]
        mark = "★" if (abs(v-OPT_VOL)<0.01 and abs(t-OPT_TAKE)<0.01) else " "
        row += f"  {cal:>4.1f}{mark}"
    print(row)

print(f"\n  ★ = 当前最优点（vol={OPT_VOL}, take={OPT_TAKE}R）")
print(f"  Calmar 越高越好，相邻格差异小 = 稳健，差异大 = 脆弱")

# 稳健性总结
print(f"\n{'='*W}")
print("  稳健性总结")
print("=" * W)
vol_robustness  = sum(1 for v in VOL_RANGE  if run(df, {**BASE, "vol_spike_mult": v}, OPT_CD, OPT_STOP, OPT_TAKE)["ret"] >= base_r["ret"]*0.8)
take_robustness = sum(1 for t in TAKE_RANGE if run(df, BASE, OPT_CD, OPT_STOP, t)["ret"] >= base_r["ret"]*0.8)
stop_robustness = sum(1 for s in STOP_RANGE if run(df, BASE, OPT_CD, s, OPT_TAKE)["ret"] >= base_r["ret"]*0.8)

print(f"  vol_spike_mult：{vol_robustness}/{len(VOL_RANGE)} 个值收益在最优的 80% 以上")
print(f"  take_R        ：{take_robustness}/{len(TAKE_RANGE)} 个值收益在最优的 80% 以上")
print(f"  stop_atr      ：{stop_robustness}/{len(STOP_RANGE)} 个值收益在最优的 80% 以上")
print(f"\n  判断标准：≥60% 稳健 ✓  40-60% 尚可 △  <40% 过拟合风险 ✗")
print("=" * W)
