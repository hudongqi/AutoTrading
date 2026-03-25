"""
G: PEPE 多空信号拆分分析
对比：双向（当前）vs 纯做多 vs 纯做空
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

def make_params(allow_short):
    return {
        "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
        "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
        "reclaim_ema": 20, "trend_ema": 200,
        "vol_period": 20, "vol_spike_mult": 1.5,
        "atr_pct_low": 0.008, "atr_pct_high": 0.20,
        "oversold_lookback": 3, "allow_short": allow_short,
    }

def run(df, allow_short, long_only=False, short_only=False):
    if len(df) < 200:
        return None
    params = make_params(allow_short)
    strat  = SOLReversionV2Strategy1H(**params)
    df_sig = strat.generate_signals(df).copy()

    # 纯做多：把所有空头信号清零
    if long_only:
        df_sig["entry_setup"] = df_sig["entry_setup"].clip(lower=0)
        df_sig["trade_signal"] = df_sig["trade_signal"].clip(lower=0)
        df_sig["signal"] = df_sig["signal"].clip(lower=0)
    # 纯做空：把所有多头信号清零
    if short_only:
        df_sig["entry_setup"] = df_sig["entry_setup"].clip(upper=0)
        df_sig["trade_signal"] = df_sig["trade_signal"].clip(upper=0)
        df_sig["signal"] = df_sig["signal"].clip(upper=0)

    btp  = get_sol_backtest_profile("pepe_rev_v2")
    port = PerpPortfolio(
        initial_cash=TOTAL_CASH, leverage=btp["leverage"],
        taker_fee_rate=config.TAKER_FEE_RATE, maker_fee_rate=config.MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )
    bt = Backtester(
        broker=SimBroker(slippage_bps=config.SLIPPAGE_BPS),
        portfolio=port, strategy=strat,
        max_pos=5_000_000, cooldown_bars=btp["cooldown_bars"],
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
    calmar = ret / abs(dd) if dd != 0 else 0
    trades = bt.closed_trades
    longs  = [t for t in trades if t.get("side") ==  1]
    shorts = [t for t in trades if t.get("side") == -1]

    def side_exp(ts):
        if not ts: return 0.0
        pnls = [t.get("realized_net", 0) for t in ts]
        return float(np.mean(pnls))

    def side_wr(ts):
        if not ts: return 0.0
        return sum(1 for t in ts if t.get("realized_net", 0) > 0) / len(ts) * 100

    return dict(ret=ret, dd=dd, calmar=calmar, wr=wr, nc=len(trades),
                nl=len(longs), ns=len(shorts), eq=eq,
                long_exp=side_exp(longs), short_exp=side_exp(shorts),
                long_wr=side_wr(longs),   short_wr=side_wr(shorts))


print("拉取 PEPE 数据...")
ds = CCXTDataSource()
df_all = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"PEPE {len(df_all)} 根\n")

# ── 全程三路对比 ─────────────────────────────────────────
print("运行全程三路对比...")
r_both  = run(df_all, allow_short=True,  long_only=False, short_only=False)
r_long  = run(df_all, allow_short=False, long_only=True,  short_only=False)
r_short = run(df_all, allow_short=True,  long_only=False, short_only=True)

W = 95
print(f"\n{'='*W}")
print("  PEPE 多空拆分  |  全程 2024-01-01 → 2026-03-24  |  take_R=4.0R")
print(f"{'='*W}")
print(f"  {'模式':<12} │ {'收益':>8} {'DD':>7} {'Calmar':>7} │ "
      f"{'胜率':>5} {'成交':>5} │ {'多头笔':>6} {'多头期望':>9} {'多头胜率':>8} │ "
      f"{'空头笔':>6} {'空头期望':>9} {'空头胜率':>8}")
print("  " + "─" * (W - 2))
for label, r in [("双向（当前）", r_both), ("纯做多", r_long), ("纯做空", r_short)]:
    l_exp = f"${r['long_exp']:>+.1f}"  if r['nl'] > 0 else "─"
    s_exp = f"${r['short_exp']:>+.1f}" if r['ns'] > 0 else "─"
    l_wr  = f"{r['long_wr']:.0f}%"    if r['nl'] > 0 else "─"
    s_wr  = f"{r['short_wr']:.0f}%"   if r['ns'] > 0 else "─"
    print(f"  {label:<12} │ {r['ret']:>+7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['wr']:>4.0f}% {r['nc']:>4}笔 │ "
          f"{r['nl']:>5}笔 {l_exp:>9} {l_wr:>8} │ "
          f"{r['ns']:>5}笔 {s_exp:>9} {s_wr:>8}")

# ── 季度拆分三路对比 ─────────────────────────────────────
print(f"\n{'='*W}")
print("  季度拆分：双向 vs 纯做多 vs 纯做空")
print(f"{'='*W}")
print(f"  {'季度':<14} {'PEPE标的':>8} │ {'双向':>8} {'纯多':>7} {'纯空':>7} │ "
      f"{'双向DD':>7} {'纯多DD':>7} {'纯空DD':>7}")
print("  " + "─" * (W - 2))

rows_both, rows_long, rows_short = [], [], []
for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_q = df_all[(df_all.index >= ts) & (df_all.index < te)].copy()
    coin_ret = (df_q["close"].iloc[-1] - df_q["close"].iloc[0]) / df_q["close"].iloc[0] * 100

    rb = run(df_q, allow_short=True,  long_only=False, short_only=False)
    rl = run(df_q, allow_short=False, long_only=True,  short_only=False)
    rs = run(df_q, allow_short=True,  long_only=False, short_only=True)

    _z = dict(ret=0, dd=0, calmar=0, wr=0, nc=0, nl=0, ns=0,
              long_exp=0, short_exp=0, long_wr=0, short_wr=0)
    rb = rb or _z; rl = rl or _z; rs = rs or _z

    # 标记最优
    best = max(rb["ret"], rl["ret"], rs["ret"])
    marks = [" ◀" if r["ret"] == best else "  " for r in [rb, rl, rs]]

    print(f"  {label:<14} {coin_ret:>+7.1f}% │ "
          f"{rb['ret']:>+7.2f}%{marks[0]} {rl['ret']:>+6.2f}%{marks[1]} {rs['ret']:>+6.2f}%{marks[2]} │ "
          f"{rb['dd']:>6.2f}% {rl['dd']:>6.2f}% {rs['dd']:>6.2f}%")

    rows_both.append(rb); rows_long.append(rl); rows_short.append(rs)

print("  " + "─" * (W - 2))

def smry(rows):
    rets = [r["ret"] for r in rows]
    return dict(avg=np.mean(rets), wins=sum(1 for r in rets if r > 0),
                worst=min(rets), best=max(rets))

sb = smry(rows_both); sl = smry(rows_long); ss = smry(rows_short)
print(f"\n  {'指标':<14} {'双向':>10} {'纯做多':>10} {'纯做空':>10}")
print(f"  {'─'*48}")
for lbl, vb, vl, vs in [
    ("盈利季度",     f"{sb['wins']}/9",          f"{sl['wins']}/9",          f"{ss['wins']}/9"),
    ("平均季度收益", f"{sb['avg']:>+.2f}%",       f"{sl['avg']:>+.2f}%",       f"{ss['avg']:>+.2f}%"),
    ("最佳季度",     f"{sb['best']:>+.2f}%",      f"{sl['best']:>+.2f}%",      f"{ss['best']:>+.2f}%"),
    ("最差季度",     f"{sb['worst']:>+.2f}%",     f"{sl['worst']:>+.2f}%",     f"{ss['worst']:>+.2f}%"),
]:
    print(f"  {lbl:<14} {vb:>10} {vl:>10} {vs:>10}")

# ── 多空期望值分析 ───────────────────────────────────────
print(f"\n{'='*W}")
print("  多头 vs 空头 逐笔期望值分析（全程）")
print(f"{'='*W}")
print(f"  多头：{r_both['nl']}笔  胜率 {r_both['long_wr']:.0f}%  平均每笔 ${r_both['long_exp']:>+.1f}")
print(f"  空头：{r_both['ns']}笔  胜率 {r_both['short_wr']:.0f}%  平均每笔 ${r_both['short_exp']:>+.1f}")
print(f"  合计：{r_both['nc']}笔  整体胜率 {r_both['wr']:.0f}%")
print(f"\n  结论：", end="")
if r_both["long_exp"] > 0 and r_both["short_exp"] > 0:
    if r_both["short_exp"] > r_both["long_exp"]:
        print(f"多空均有正期望，空头期望更高（${r_both['short_exp']:+.1f} vs ${r_both['long_exp']:+.1f}）")
    else:
        print(f"多空均有正期望，多头期望更高（${r_both['long_exp']:+.1f} vs ${r_both['short_exp']:+.1f}）")
elif r_both["long_exp"] > 0 and r_both["short_exp"] <= 0:
    print(f"多头正期望（${r_both['long_exp']:+.1f}），空头负期望（${r_both['short_exp']:+.1f}）→ 建议仅做多")
elif r_both["long_exp"] <= 0 and r_both["short_exp"] > 0:
    print(f"空头正期望（${r_both['short_exp']:+.1f}），多头负期望（${r_both['long_exp']:+.1f}）→ 建议仅做空")
else:
    print("多空均为负期望，信号质量需要改善")
print("=" * W)
