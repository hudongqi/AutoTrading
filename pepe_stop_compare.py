"""
stop_atr=3.5 vs 5.0 季度拆分对比
PEPE 单币 $10,000
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
STOPS = [3.5, 5.0]


def run(df, stop_atr):
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
        max_pos=5_000_000, cooldown_bars=btp["cooldown_bars"],
        stop_atr=stop_atr, take_R=btp["take_R"],
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
    return dict(ret=ret, dd=dd, calmar=calmar, wr=wr, nc=nc)


print("拉取 PEPE 数据...")
ds = CCXTDataSource()
df_all = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"PEPE {len(df_all)} 根\n")

# ── 全程对比 ──────────────────────────────────────────────
W = 82
print("=" * W)
print("  全程对比（2024-01 → 2026-03）  stop_atr=3.5 vs 5.0")
print("=" * W)
print(f"  {'stop_atr':>9} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'胜率':>5} {'成交':>5}")
print("  " + "─" * (W - 2))
full = {}
for s in STOPS:
    r = run(df_all, s)
    full[s] = r
    mark = " ◀当前" if s == 5.0 else ""
    print(f"  {s:>9.1f} │ {r['ret']:>+7.1f}%  {r['dd']:>6.1f}%  {r['calmar']:>6.2f}  "
          f"{r['wr']:>4.0f}%  {r['nc']:>4}笔{mark}")

# ── 季度拆分 ──────────────────────────────────────────────
print(f"\n{'='*W}")
print("  季度拆分：stop_atr = 3.5 vs 5.0（当前）")
print("=" * W)
print(f"  {'季度':<16} │ {'3.5收益':>8} {'5.0收益':>8} │ {'3.5 DD':>7} {'5.0 DD':>7} │ {'3.5 Cal':>7} {'5.0 Cal':>7} │ {'3.5笔':>5} {'5.0笔':>5}")
print("  " + "─" * (W - 2))

rows = {s: [] for s in STOPS}
for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_q = df_all[(df_all.index >= ts) & (df_all.index < te)].copy()
    rs = {}
    for s in STOPS:
        r = run(df_q, s)
        rs[s] = r if r else dict(ret=0, dd=0, calmar=0, wr=0, nc=0)
    better = "↑3.5" if rs[3.5]["ret"] > rs[5.0]["ret"] else ("↑5.0" if rs[5.0]["ret"] > rs[3.5]["ret"] else "  ==")
    sign = "✓" if rs[3.5]["ret"] > 0 and rs[5.0]["ret"] > 0 else ("△" if rs[3.5]["ret"] > 0 or rs[5.0]["ret"] > 0 else "✗")
    print(f"  {sign} {label:<14} │ "
          f"{rs[3.5]['ret']:>+7.2f}% {rs[5.0]['ret']:>+7.2f}% │ "
          f"{rs[3.5]['dd']:>6.2f}% {rs[5.0]['dd']:>6.2f}% │ "
          f"{rs[3.5]['calmar']:>7.2f} {rs[5.0]['calmar']:>7.2f} │ "
          f"{rs[3.5]['nc']:>4}笔  {rs[5.0]['nc']:>4}笔  {better}")
    for s in STOPS:
        rows[s].append(rs[s])

print("  " + "─" * (W - 2))

# 汇总
print(f"\n  {'指标':<16} {'stop=3.5':>12} {'stop=5.0':>12}")
print(f"  {'─'*42}")
for lbl, fn in [
    ("盈利季度",     lambda rs: f"{sum(1 for r in rs if r['ret']>0)}/9"),
    ("平均季度收益", lambda rs: f"{np.mean([r['ret'] for r in rs]):>+.2f}%"),
    ("平均 Calmar",  lambda rs: f"{np.mean([r['calmar'] for r in rs]):.2f}"),
    ("平均回撤",     lambda rs: f"{np.mean([r['dd'] for r in rs]):.2f}%"),
    ("最大单季回撤", lambda rs: f"{min(r['dd'] for r in rs):.2f}%"),
    ("最差季度收益", lambda rs: f"{min(r['ret'] for r in rs):>+.2f}%"),
    ("平均成交/季",  lambda rs: f"{np.mean([r['nc'] for r in rs]):.1f}笔"),
]:
    print(f"  {lbl:<16} {fn(rows[3.5]):>12} {fn(rows[5.0]):>12}")

print("=" * W)
