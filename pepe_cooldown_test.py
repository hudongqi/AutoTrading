"""
J: PEPE cooldown_bars 扫描（0 ~ 4）
全程 + 季度拆分，PEPE 单币 $10,000
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
CDS = [0, 1, 2, 3, 4]


def run(df, cd):
    if len(df) < 200:
        return None
    strat  = SOLReversionV2Strategy1H(**PEPE_PARAMS)
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
    stats  = result.attrs.get("stats", {})
    eq     = result["equity"]
    dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret    = (eq.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    wr     = stats.get("win_rate", 0) * 100
    nc     = len(bt.closed_trades)
    calmar = ret / abs(dd) if dd != 0 else 0
    return dict(ret=ret, dd=dd, calmar=calmar, wr=wr, nc=nc, eq=eq)


print("拉取 PEPE 数据...")
ds = CCXTDataSource()
df_all = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"PEPE {len(df_all)} 根\n")

# ── 全程扫描 ─────────────────────────────────────────────
full = {}
for cd in CDS:
    full[cd] = run(df_all, cd)

W = 80
print("=" * W)
print("  cooldown_bars 扫描  |  PEPE 单币  |  全程 2024-01 → 2026-03")
print("=" * W)
print(f"  {'cooldown':>9} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'胜率':>5} {'成交':>5} │ {'vs基线':>7}")
print("  " + "─" * (W - 2))

base_r = full[2]
for cd in CDS:
    r = full[cd]
    is_base = cd == 2
    diff    = r["ret"] - base_r["ret"]
    best_cal = max(full[c]["calmar"] for c in CDS)
    mark    = " ★" if is_base else (" ◀" if abs(r["calmar"] - best_cal) < 0.01 else "")
    print(f"  {cd:>9} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} "
          f"{r['wr']:>4.0f}% {r['nc']:>4}笔 │ {diff:>+6.1f}%{mark}")

# ── 季度拆分：0 / 1 / 2（当前）───────────────────────────
print(f"\n{'='*W}")
print("  季度拆分：cooldown = 0 / 1 / 2（当前）")
print("=" * W)
print(f"  {'季度':<14} │ {'cd=0':>8} {'cd=1':>8} {'cd=2':>8} │ {'cd=0 DD':>8} {'cd=1 DD':>8} {'cd=2 DD':>8}")
print("  " + "─" * (W - 2))

rows = {cd: [] for cd in [0, 1, 2]}
for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_q = df_all[(df_all.index >= ts) & (df_all.index < te)].copy()
    rs = {cd: (run(df_q, cd) or dict(ret=0, dd=0, nc=0)) for cd in [0, 1, 2]}
    best = max(rs[cd]["ret"] for cd in [0, 1, 2])
    marks = {cd: " ◀" if abs(rs[cd]["ret"] - best) < 0.01 else "  " for cd in [0, 1, 2]}
    print(f"  {label:<14} │ "
          f"{rs[0]['ret']:>+7.2f}%{marks[0]} {rs[1]['ret']:>+7.2f}%{marks[1]} "
          f"{rs[2]['ret']:>+7.2f}%{marks[2]} │ "
          f"{rs[0]['dd']:>7.2f}% {rs[1]['dd']:>7.2f}% {rs[2]['dd']:>7.2f}%")
    for cd in [0, 1, 2]:
        rows[cd].append(rs[cd])

print("  " + "─" * (W - 2))

def smry(rs):
    rets = [r["ret"] for r in rs]
    dds  = [r["dd"]  for r in rs]
    return dict(avg=np.mean(rets), wins=sum(1 for r in rets if r > 0),
                worst=min(rets), avg_dd=np.mean(dds), worst_dd=min(dds))

sums = {cd: smry(rows[cd]) for cd in [0, 1, 2]}
print(f"\n  {'指标':<14} {'cd=0':>10} {'cd=1':>10} {'cd=2':>10}")
print(f"  {'─'*48}")
for lbl, key, fmt in [
    ("盈利季度",     "wins",     lambda v: f"{v}/9"),
    ("平均季度收益", "avg",      lambda v: f"{v:>+.2f}%"),
    ("平均回撤",     "avg_dd",   lambda v: f"{v:.2f}%"),
    ("最大单季回撤", "worst_dd", lambda v: f"{v:.2f}%"),
    ("最差季度",     "worst",    lambda v: f"{v:>+.2f}%"),
]:
    vals = [fmt(sums[cd][key]) for cd in [0, 1, 2]]
    print(f"  {lbl:<14} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

print("=" * W)
