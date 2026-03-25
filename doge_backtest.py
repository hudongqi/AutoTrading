"""
DOGE 回测 — 使用当前 PEPE 策略参数直接测试
对比：PEPE vs DOGE 季度拆分 + 全程
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

# 当前 PEPE 策略参数（直接迁移）
PARAMS = {
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


def run(df, max_pos=5_000_000):
    if len(df) < 200:
        return None
    strat  = SOLReversionV2Strategy1H(**PARAMS)
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
    stats  = result.attrs.get("stats", {})
    eq     = result["equity"]
    dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret    = (eq.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    wr     = stats.get("win_rate", 0) * 100
    pf     = stats.get("profit_factor", 0)
    nc     = len(bt.closed_trades)
    calmar = ret / abs(dd) if dd != 0 else 0
    coin_ret = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    nl = int((df_sig["entry_setup"] == 1).sum())
    ns = int((df_sig["entry_setup"] == -1).sum())
    return dict(ret=ret, dd=dd, calmar=calmar, wr=wr, pf=pf, nc=nc,
                coin_ret=coin_ret, nl=nl, ns=ns, eq=eq)


# ── 拉取数据 ─────────────────────────────────────────────
print("拉取数据...")
ds = CCXTDataSource()
df_doge = ds.load_ohlcv("DOGE/USDT:USDT",      "2024-01-01", "2026-03-24", "1h")
df_pepe = ds.load_ohlcv("1000PEPE/USDT:USDT",  "2024-01-01", "2026-03-24", "1h")
print(f"DOGE {len(df_doge)} 根  |  PEPE {len(df_pepe)} 根\n")

# ── 全程对比 ──────────────────────────────────────────────
W = 96
print("=" * W)
print("  全程回测（2024-01 → 2026-03）  相同策略参数，$10,000")
print("=" * W)
r_doge_all = run(df_doge, max_pos=500_000)
r_pepe_all = run(df_pepe, max_pos=5_000_000)

print(f"  {'标的':<8} │ {'收益':>8} {'DD':>7} {'Calmar':>7} {'胜率':>5} {'成交':>5} │ "
      f"{'多头信号':>7} {'空头信号':>7} │ {'标的涨幅':>8}")
print("  " + "─" * (W - 2))
for label, r in [("DOGE", r_doge_all), ("PEPE", r_pepe_all)]:
    pf_s = f"{r['pf']:.2f}" if r["pf"] not in (float("inf"), 0) else "∞"
    print(f"  {label:<8} │ {r['ret']:>+7.1f}%  {r['dd']:>6.1f}%  {r['calmar']:>6.2f}  "
          f"{r['wr']:>4.0f}%  {r['nc']:>4}笔 │ "
          f"{r['nl']:>6}个   {r['ns']:>6}个 │ {r['coin_ret']:>+7.1f}%")

# ── 季度拆分对比 ──────────────────────────────────────────
print(f"\n{'='*W}")
print("  季度拆分对比  |  相同参数  |  各 $10,000")
print("=" * W)
print(f"  {'季度':<16} │ {'DOGE收益':>8} {'PEPE收益':>9} │ "
      f"{'DOGE DD':>7} {'PEPE DD':>8} │ "
      f"{'DOGE Cal':>8} {'PEPE Cal':>9} │ "
      f"{'DOGE笔':>6} {'PEPE笔':>7} │ "
      f"{'DOGE标的':>8} {'PEPE标的':>9}")
print("  " + "─" * (W - 2))

doge_rows, pepe_rows = [], []
for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_d = df_doge[(df_doge.index >= ts) & (df_doge.index < te)].copy()
    df_p = df_pepe[(df_pepe.index >= ts) & (df_pepe.index < te)].copy()
    rd = run(df_d, max_pos=500_000)
    rp = run(df_p, max_pos=5_000_000)
    if rd is None and rp is None:
        continue
    rd = rd or dict(ret=0, dd=0, calmar=0, wr=0, nc=0, coin_ret=0)
    rp = rp or dict(ret=0, dd=0, calmar=0, wr=0, nc=0, coin_ret=0)

    d_sign = "✓" if rd["ret"] > 0 else "✗"
    p_sign = "✓" if rp["ret"] > 0 else "✗"
    better = "D↑" if rd["ret"] > rp["ret"] else ("P↑" if rp["ret"] > rd["ret"] else "==")

    print(f"  {d_sign}{p_sign} {label:<13} │ "
          f"{rd['ret']:>+7.2f}%  {rp['ret']:>+7.2f}% │ "
          f"{rd['dd']:>6.2f}%  {rp['dd']:>7.2f}% │ "
          f"{rd['calmar']:>8.2f}  {rp['calmar']:>8.2f} │ "
          f"{rd['nc']:>4}笔   {rp['nc']:>5}笔 │ "
          f"{rd['coin_ret']:>+7.1f}%  {rp['coin_ret']:>+8.1f}%  {better}")
    doge_rows.append(rd)
    pepe_rows.append(rp)

print("  " + "─" * (W - 2))

# ── 汇总 ──────────────────────────────────────────────────
print(f"\n  {'指标':<18} {'DOGE':>12} {'PEPE':>12}")
print(f"  {'─'*44}")
for lbl, fn in [
    ("盈利季度",     lambda rs: f"{sum(1 for r in rs if r['ret']>0)}/{len(rs)}"),
    ("平均季度收益", lambda rs: f"{np.mean([r['ret'] for r in rs]):>+.2f}%"),
    ("平均 Calmar",  lambda rs: f"{np.mean([r['calmar'] for r in rs]):.2f}"),
    ("平均回撤",     lambda rs: f"{np.mean([r['dd'] for r in rs]):.2f}%"),
    ("最大单季回撤", lambda rs: f"{min(r['dd'] for r in rs):.2f}%"),
    ("最差季度收益", lambda rs: f"{min(r['ret'] for r in rs):>+.2f}%"),
    ("平均胜率",     lambda rs: f"{np.mean([r['wr'] for r in rs]):.1f}%"),
    ("平均成交/季",  lambda rs: f"{np.mean([r['nc'] for r in rs]):.1f}笔"),
]:
    print(f"  {lbl:<18} {fn(doge_rows):>12} {fn(pepe_rows):>12}")

print("=" * W)
