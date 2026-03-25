"""
Walk-Forward Analysis (WFA) — PEPE 单币
滚动窗口：IS=9个月优化，OOS=3个月验证
优化参数：vol_spike_mult / take_R / stop_atr / cooldown_bars
优化指标：Calmar ratio（收益/最大回撤）
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

BASE_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20,
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
}

# 优化网格
GRID = {
    "vol_spike_mult": [0.8, 1.2, 1.5, 2.0],
    "take_R":         [3.0, 4.0, 5.0, 6.0],
    "stop_atr":       [3.0, 4.0, 5.0, 6.0],
    "cooldown_bars":  [2, 3],
}
COMBOS = list(product(*GRID.values()))
KEYS   = list(GRID.keys())
print(f"参数组合数：{len(COMBOS)} 个\n")

# 滚动窗口定义（IS=9月，OOS=3月，步长=3月）
# 全量数据 2023-05-01 → 2026-03-24
WINDOWS = []
starts = pd.date_range("2023-05-01", "2025-07-01", freq="3MS", tz="UTC")
for s in starts:
    is_end  = s + pd.DateOffset(months=9)
    oos_end = is_end + pd.DateOffset(months=3)
    if oos_end > pd.Timestamp("2026-03-24", tz="UTC"):
        oos_end = pd.Timestamp("2026-03-24", tz="UTC")
    label = f"{is_end.strftime('%Y-%m')}→{oos_end.strftime('%Y-%m')}"
    WINDOWS.append((s, is_end, oos_end, label))


def run_single(df, strat_params, stop_atr, take_r, cd):
    """返回 (ret, dd, calmar, nc)；数据不足返回 None"""
    if len(df) < 100:
        return None
    strat  = SOLReversionV2Strategy1H(**strat_params)
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
    eq  = result["equity"]
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    nc  = len(bt.closed_trades)
    calmar = ret / abs(dd) if dd != 0 else (ret if ret > 0 else -999)
    return dict(ret=ret, dd=dd, calmar=calmar, nc=nc)


# ── 预拉全量数据 ─────────────────────────────────────────
print("拉取 PEPE 全量数据（2023-05 → 2026-03）...")
ds = CCXTDataSource()
df_all = ds.load_ohlcv("1000PEPE/USDT:USDT", "2023-05-01", "2026-03-24", "1h")
print(f"PEPE {len(df_all)} 根\n")

# ── 固定参数对照组（不做 WFA，全程用当前最优）─────────────
FIXED = {"vol_spike_mult": 1.2, "stop_atr": 5.0, "take_R": 4.0, "cooldown_bars": 3}
fixed_params = {**BASE_PARAMS, "vol_spike_mult": FIXED["vol_spike_mult"]}

# ── 主循环：每个窗口优化 + OOS 验证 ─────────────────────
W = 100
print("=" * W)
print("  Walk-Forward Analysis  |  IS=9月优化  OOS=3月验证  步长3月")
print(f"  {'OOS区间':<20} │ {'IS最优参数':^40} │ {'IS Calmar':>9} │ "
      f"{'OOS收益':>8} {'OOS DD':>7} {'OOS Cal':>8} {'成交':>4} │ {'固定参数OOS':>10}")
print("  " + "─" * (W - 2))

oos_records   = []   # WFA 每窗口 OOS 结果
fixed_records = []   # 固定参数对照

total_combos = len(WINDOWS) * len(COMBOS)
done = 0

for w_idx, (is_start, is_end, oos_end, label) in enumerate(WINDOWS):
    df_is  = df_all[(df_all.index >= is_start) & (df_all.index < is_end)].copy()
    df_oos = df_all[(df_all.index >= is_end)   & (df_all.index < oos_end)].copy()

    if len(df_is) < 200 or len(df_oos) < 50:
        print(f"  {label:<20} │ 数据不足，跳过")
        continue

    # IS 优化
    best_cal   = -999
    best_combo = None
    best_is_r  = None
    for combo in COMBOS:
        params_dict = dict(zip(KEYS, combo))
        sp = {**BASE_PARAMS, "vol_spike_mult": params_dict["vol_spike_mult"]}
        r  = run_single(df_is, sp, params_dict["stop_atr"], params_dict["take_R"], params_dict["cooldown_bars"])
        done += 1
        if done % 50 == 0:
            print(f"  进度 [{done}/{total_combos}]...", end="\r", flush=True)
        if r and r["nc"] >= 3 and r["calmar"] > best_cal:
            best_cal   = r["calmar"]
            best_combo = params_dict
            best_is_r  = r

    if best_combo is None:
        print(f"  {label:<20} │ IS 无有效组合，跳过")
        continue

    # OOS 验证（用 IS 最优参数）
    sp_oos = {**BASE_PARAMS, "vol_spike_mult": best_combo["vol_spike_mult"]}
    r_oos  = run_single(df_oos, sp_oos, best_combo["stop_atr"], best_combo["take_R"], best_combo["cooldown_bars"])
    if r_oos is None:
        r_oos = dict(ret=0, dd=0, calmar=0, nc=0)

    # 固定参数对照
    r_fix = run_single(df_oos, fixed_params, FIXED["stop_atr"], FIXED["take_R"], FIXED["cooldown_bars"])
    if r_fix is None:
        r_fix = dict(ret=0, dd=0, calmar=0, nc=0)

    param_str = (f"vol={best_combo['vol_spike_mult']:.1f} "
                 f"take={best_combo['take_R']:.1f}R "
                 f"stop={best_combo['stop_atr']:.1f} "
                 f"cd={best_combo['cooldown_bars']}")

    sign = "✓" if r_oos["ret"] > 0 else "✗"
    print(f"\r  {sign} {label:<20} │ {param_str:<40} │ {best_cal:>9.2f} │ "
          f"{r_oos['ret']:>+7.1f}% {r_oos['dd']:>6.1f}% {r_oos['calmar']:>8.2f} {r_oos['nc']:>3}笔 │ "
          f"{r_fix['ret']:>+9.1f}%")

    oos_records.append({**r_oos, "params": best_combo, "is_calmar": best_cal, "label": label})
    fixed_records.append(r_fix)

# ── 汇总 ──────────────────────────────────────────────────
print("\n" + "=" * W)
print("  WFA 汇总")
print("=" * W)

if not oos_records:
    print("  无有效窗口")
else:
    oos_rets  = [r["ret"]    for r in oos_records]
    oos_cals  = [r["calmar"] for r in oos_records]
    oos_dds   = [r["dd"]     for r in oos_records]
    fix_rets  = [r["ret"]    for r in fixed_records]
    fix_cals  = [r["calmar"] for r in fixed_records]

    n   = len(oos_records)
    pos = sum(1 for r in oos_rets if r > 0)

    print(f"\n  {'指标':<20} {'WFA 自适应':>14} {'固定参数对照':>14}")
    print(f"  {'─'*50}")
    for lbl, wfa_v, fix_v in [
        ("窗口总数",       f"{n}",                    f"{n}"),
        ("OOS盈利窗口",    f"{pos}/{n} ({pos/n*100:.0f}%)", f"{sum(1 for r in fix_rets if r>0)}/{n}"),
        ("平均OOS收益",    f"{np.mean(oos_rets):>+.2f}%",   f"{np.mean(fix_rets):>+.2f}%"),
        ("平均OOS Calmar", f"{np.mean(oos_cals):.2f}",      f"{np.mean(fix_cals):.2f}"),
        ("平均OOS DD",     f"{np.mean(oos_dds):.2f}%",      f"{np.mean([r['dd'] for r in fixed_records]):.2f}%"),
        ("最差OOS窗口",    f"{min(oos_rets):>+.2f}%",       f"{min(fix_rets):>+.2f}%"),
    ]:
        print(f"  {lbl:<20} {wfa_v:>14} {fix_v:>14}")

    # IS/OOS 效率比
    is_cals = [r["is_calmar"] for r in oos_records]
    wfa_eff = np.mean(oos_cals) / np.mean(is_cals) if np.mean(is_cals) > 0 else 0
    print(f"\n  IS 平均 Calmar:  {np.mean(is_cals):.2f}")
    print(f"  OOS 平均 Calmar: {np.mean(oos_cals):.2f}")
    print(f"  IS→OOS 效率比:   {wfa_eff:.1%}  （>50% 为合格，>70% 优秀）")

    # 参数稳定性分析
    print(f"\n  OOS期间各窗口最优参数:")
    print(f"  {'OOS区间':<22} {'vol':>5} {'take':>6} {'stop':>6} {'cd':>4} │ {'OOS收益':>8} {'OOS Cal':>8}")
    print(f"  {'─'*65}")
    for r in oos_records:
        p = r["params"]
        mark = " ◀当前" if (abs(p["vol_spike_mult"]-1.2)<0.01 and
                            abs(p["take_R"]-4.0)<0.01 and
                            abs(p["stop_atr"]-5.0)<0.01 and
                            p["cooldown_bars"]==3) else ""
        sign = "✓" if r["ret"] > 0 else "✗"
        print(f"  {sign} {r['label']:<22} {p['vol_spike_mult']:>5.1f} {p['take_R']:>5.1f}R "
              f"{p['stop_atr']:>5.1f}  {p['cooldown_bars']:>3} │ "
              f"{r['ret']:>+7.1f}% {r['calmar']:>8.2f}{mark}")

    # 参数频率统计
    print(f"\n  各窗口最优参数频率：")
    for key in KEYS:
        vals = [r["params"][key] for r in oos_records]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        freq_str = "  ".join(f"{k}={v}次" for k, v in sorted(counts.items()))
        print(f"    {key:<18}: {freq_str}")

print("\n" + "=" * W)
