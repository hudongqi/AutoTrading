"""
三路对比：原始策略 vs 旧版DD止损（权益恢复）vs 新版DD止损（时间冷却）
冷静期基于历史黑天鹅恢复时间分析（P75）：SOL=456bar=19天，PEPE=648bar=27天
季度拆分（2024 Q1 → 2026 Q1）
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

DD_STOP    = 0.15   # 触发阈值：单币种从高点回撤 15%

# 基于历史黑天鹅 P75 恢复时间（shock_recovery_analysis.py 输出）
SOL_COOLDOWN  = 456   # 19天 × 24H
PEPE_COOLDOWN = 648   # 27天 × 24H


def run_coin(df, strat_params, bt_profile, max_pos, cash,
             dd_stop=0.0, dd_cooldown=0):
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
        allow_reentry=False, partial_take_R=0.0, partial_take_frac=0.0,
        break_even_after_partial=False, break_even_R=0.0,
        use_signal_exit_targets=False, max_hold_bars=btp["max_hold_bars"],
        dd_stop_pct=dd_stop,
        dd_cooldown_bars=dd_cooldown,
    )
    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])
    eq  = result["equity"]
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - cash) / cash * 100
    return dict(ret=ret, dd=dd, eq=eq,
                nc=stats.get("closed_trade_count", len(closed)))


def run_quarter(df_sol, df_pepe, dd_stop=0.0, dd_cooldown_sol=0, dd_cooldown_pepe=0):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH,
                  dd_stop=dd_stop, dd_cooldown=dd_cooldown_sol)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH,
                  dd_stop=dd_stop, dd_cooldown=dd_cooldown_pepe)
    if rs is None and rp is None:
        return None
    if rs and rp:
        eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
        eq_c = eq_s + eq_p
        dd_c  = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
        ret_c = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    elif rs:
        ret_c, dd_c = rs["ret"] / 2, rs["dd"]
    else:
        ret_c, dd_c = rp["ret"] / 2, rp["dd"]
    return dict(ret=ret_c, dd=dd_c,
                nc=(rs["nc"] if rs else 0) + (rp["nc"] if rp else 0))


# ── 预拉全量数据 ─────────────────────────────────────────────
print("拉取全量数据...")
ds = CCXTDataSource()
df_sol_all  = ds.load_ohlcv("SOL/USDT:USDT",      start="2024-01-01", end="2026-03-24", timeframe="1h")
df_pepe_all = ds.load_ohlcv("1000PEPE/USDT:USDT", start="2024-01-01", end="2026-03-24", timeframe="1h")
print(f"SOL {len(df_sol_all)} 根  PEPE {len(df_pepe_all)} 根\n")

# ── 逐季度三路对比 ─────────────────────────────────────────────
W = 120
print("=" * W)
print(f"  三路对比：原始 vs 旧版DD止损（权益恢复）vs 新版DD止损（时间冷却）")
print(f"  触发阈值: {DD_STOP*100:.0f}%  冷静期: SOL={SOL_COOLDOWN//24}天 PEPE={PEPE_COOLDOWN//24}天")
print("=" * W)
hdr = (f"  {'季度':<14}  {'原始':>7} {'权益恢复':>8} {'时间冷却':>8}  │  "
       f"{'原始DD':>7} {'权益DD':>7} {'冷却DD':>7}  │  "
       f"{'原始笔':>6} {'权益笔':>6} {'冷却笔':>6}")
print(hdr)
print("  " + "─" * (W - 2))

rows_base, rows_eq, rows_time = [], [], []

for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_s = df_sol_all [(df_sol_all.index  >= ts) & (df_sol_all.index  < te)].copy()
    df_p = df_pepe_all[(df_pepe_all.index >= ts) & (df_pepe_all.index < te)].copy()

    rb = run_quarter(df_s, df_p)                                          # 原始
    re = run_quarter(df_s, df_p, dd_stop=DD_STOP, dd_cooldown_sol=0,     # 旧版：权益恢复
                     dd_cooldown_pepe=0)
    rt = run_quarter(df_s, df_p, dd_stop=DD_STOP,                        # 新版：时间冷却
                     dd_cooldown_sol=SOL_COOLDOWN, dd_cooldown_pepe=PEPE_COOLDOWN)

    _z = dict(ret=0, dd=0, nc=0)
    rb = rb or _z; re = re or _z; rt = rt or _z

    # 变化标记（相对原始）
    diff_t = rt["ret"] - rb["ret"]
    mark = "↑" if diff_t > 0.3 else ("↓" if diff_t < -0.3 else "─")

    print(f"  {mark} {label:<12}  "
          f"{rb['ret']:>+6.1f}%  {re['ret']:>+6.1f}%  {rt['ret']:>+6.1f}%  │  "
          f"{rb['dd']:>6.1f}%  {re['dd']:>6.1f}%  {rt['dd']:>6.1f}%  │  "
          f"{rb['nc']:>5}笔  {re['nc']:>5}笔  {rt['nc']:>5}笔")

    rows_base.append(rb); rows_eq.append(re); rows_time.append(rt)

print("  " + "─" * (W - 2))

# ── 汇总 ──────────────────────────────────────────────────────
def summary(rows):
    rets = [r["ret"] for r in rows]
    dds  = [r["dd"]  for r in rows]
    return dict(
        avg_ret  = np.mean(rets),
        win_q    = sum(1 for r in rets if r > 0),
        avg_dd   = np.mean(dds),
        worst_dd = min(dds),
        trades   = sum(r["nc"] for r in rows),
    )

sb = summary(rows_base)
se = summary(rows_eq)
st = summary(rows_time)

print(f"\n  {'指标':<16} {'原始策略':>10} {'权益恢复':>10} {'时间冷却':>10}  │  "
      f"{'vs原始（权益）':>12} {'vs原始（冷却）':>12}")
print(f"  {'─'*80}")

def fmt_diff(v, base, higher_is_better=True):
    d = v - base
    if abs(d) < 0.01: return "    ─   "
    sign = "+" if d > 0 else ""
    good = (d > 0) == higher_is_better
    return f"  {'▲' if good else '▽'}{sign}{d:.2f}%"

def fmt_diff_int(v, base, higher_is_better=True):
    d = v - base
    if d == 0: return "    ─   "
    good = (d > 0) == higher_is_better
    return f"  {'▲' if good else '▽'}{d:+d}"

rows_display = [
    ("盈利季度",  f"{sb['win_q']}/9",   f"{se['win_q']}/9",   f"{st['win_q']}/9",
     se['win_q']-sb['win_q'], st['win_q']-sb['win_q'], True, "q"),
    ("平均季度收益", f"{sb['avg_ret']:>+.2f}%", f"{se['avg_ret']:>+.2f}%", f"{st['avg_ret']:>+.2f}%",
     se['avg_ret']-sb['avg_ret'], st['avg_ret']-sb['avg_ret'], True, "pct"),
    ("平均季度回撤", f"{sb['avg_dd']:.2f}%", f"{se['avg_dd']:.2f}%", f"{st['avg_dd']:.2f}%",
     se['avg_dd']-sb['avg_dd'], st['avg_dd']-sb['avg_dd'], False, "pct"),
    ("最大单季回撤", f"{sb['worst_dd']:.2f}%", f"{se['worst_dd']:.2f}%", f"{st['worst_dd']:.2f}%",
     se['worst_dd']-sb['worst_dd'], st['worst_dd']-sb['worst_dd'], False, "pct"),
    ("总成交笔数", f"{sb['trades']}", f"{se['trades']}", f"{st['trades']}",
     se['trades']-sb['trades'], st['trades']-sb['trades'], True, "int"),
]

for label, v_b, v_e, v_t, d_e, d_t, higher_good, kind in rows_display:
    if kind == "pct":
        de_str = ("  ─" if abs(d_e)<0.01 else f"  {'▲' if (d_e>0)==higher_good else '▽'}{d_e:+.2f}%").rjust(12)
        dt_str = ("  ─" if abs(d_t)<0.01 else f"  {'▲' if (d_t>0)==higher_good else '▽'}{d_t:+.2f}%").rjust(12)
    else:
        de_str = ("  ─" if d_e==0 else f"  {'▲' if (d_e>0)==higher_good else '▽'}{d_e:+d}").rjust(12)
        dt_str = ("  ─" if d_t==0 else f"  {'▲' if (d_t>0)==higher_good else '▽'}{d_t:+d}").rjust(12)
    print(f"  {label:<16} {v_b:>10} {v_e:>10} {v_t:>10}  │  {de_str} {dt_str}")

print(f"""
  机制对比：
  · 原始策略  ：无回撤保护
  · 旧版（权益恢复）：回撤 >{DD_STOP*100:.0f}% → 暂停，等权益回到峰值 85% 才恢复
                    ⚠ 无持仓时永远无法自行恢复（逻辑死锁）
  · 新版（时间冷却）：回撤 >{DD_STOP*100:.0f}% → 暂停 N 天后自动恢复
                    SOL 冷静期 {SOL_COOLDOWN//24} 天（P75历史恢复时间）
                    PEPE 冷静期 {PEPE_COOLDOWN//24} 天（P75历史恢复时间）
                    ✅ 逻辑闭环，无持仓也会自动恢复
""")
print("=" * W)
