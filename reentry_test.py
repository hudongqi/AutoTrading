"""
allow_reentry 对比测试
基线：False（当前）vs True
全程 + 季度拆分，SOL+PEPE 组合，新参数 SOL=3.0R / PEPE=4.0R
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


def run_coin(df, strat_params, bt_profile, max_pos, cash, allow_reentry):
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
        allow_reentry=allow_reentry,
        partial_take_R=0.0, partial_take_frac=0.0,
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


def run_combo(df_sol, df_pepe, allow_reentry):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH, allow_reentry)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH, allow_reentry)
    eq_s = rs["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_p = rp["eq"].reindex(rs["eq"].index.union(rp["eq"].index)).ffill()
    eq_c = eq_s + eq_p
    dd_c  = ((eq_c - eq_c.cummax()) / eq_c.cummax()).min() * 100
    ret_c = (eq_c.iloc[-1] - TOTAL_CASH) / TOTAL_CASH * 100
    calmar = ret_c / abs(dd_c) if dd_c != 0 else 0.0
    return dict(
        ret_c=ret_c, dd_c=dd_c, calmar=calmar,
        ret_s=rs["ret"], dd_s=rs["dd"], wr_s=rs["wr"], nc_s=rs["nc"],
        ret_p=rp["ret"], dd_p=rp["dd"], wr_p=rp["wr"], nc_p=rp["nc"],
    )


def run_quarter_combo(df_sol, df_pepe, allow_reentry):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH, allow_reentry)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH, allow_reentry)
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
    return dict(
        ret=ret_c, dd=dd_c,
        nc=(rs["nc"] if rs else 0) + (rp["nc"] if rp else 0),
        ret_s=rs["ret"] if rs else 0,
        ret_p=rp["ret"] if rp else 0,
    )


# ── 拉数据 ───────────────────────────────────────────────
print("拉取数据...")
ds = CCXTDataSource()
df_sol_all  = ds.load_ohlcv("SOL/USDT:USDT",      "2024-01-01", "2026-03-24", "1h")
df_pepe_all = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"SOL {len(df_sol_all)} 根  PEPE {len(df_pepe_all)} 根\n")

# ── 全程对比 ─────────────────────────────────────────────
print("运行全程对比...")
base    = run_combo(df_sol_all, df_pepe_all, allow_reentry=False)
reentry = run_combo(df_sol_all, df_pepe_all, allow_reentry=True)

W = 100
print(f"\n{'='*W}")
print("  全程对比（2024-01-01 → 2026-03-24）")
print(f"{'='*W}")
print(f"  {'':20} {'组合收益':>8} {'组合DD':>7} {'Calmar':>7} │ "
      f"{'SOL收益':>8} {'SOL胜率':>7} │ {'PEPE收益':>9} {'PEPE胜率':>8} │ {'总笔数':>6}")
print(f"  {'─'*(W-2)}")
for label, r in [("基线 (reentry=False)", base), ("+reentry=True", reentry)]:
    diff = f"  ({r['ret_c']-base['ret_c']:+.1f}%)" if r is not base else ""
    print(f"  {label:20} {r['ret_c']:>+7.1f}% {r['dd_c']:>6.1f}% {r['calmar']:>7.2f} │ "
          f"{r['ret_s']:>+7.1f}% {r['wr_s']:>6.0f}%  │ "
          f"{r['ret_p']:>+8.1f}% {r['wr_p']:>7.0f}%  │ "
          f"{r['nc_s']+r['nc_p']:>5}笔{diff}")

# ── 季度拆分对比 ─────────────────────────────────────────
print(f"\n{'='*W}")
print("  季度拆分对比")
print(f"{'='*W}")
print(f"  {'季度':<14} {'基线收益':>8} {'+reentry':>9} {'差值':>7} │ "
      f"{'基线DD':>7} {'reentryDD':>10} │ {'基线笔':>6} {'reentry笔':>9}")
print(f"  {'─'*(W-2)}")

rows_base, rows_re = [], []

for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_s = df_sol_all [(df_sol_all.index  >= ts) & (df_sol_all.index  < te)].copy()
    df_p = df_pepe_all[(df_pepe_all.index >= ts) & (df_pepe_all.index < te)].copy()

    rb = run_quarter_combo(df_s, df_p, allow_reentry=False)
    rr = run_quarter_combo(df_s, df_p, allow_reentry=True)

    _z = dict(ret=0, dd=0, nc=0, ret_s=0, ret_p=0)
    rb = rb or _z
    rr = rr or _z

    diff = rr["ret"] - rb["ret"]
    mark = "↑" if diff > 0.3 else ("↓" if diff < -0.3 else "─")

    print(f"  {mark} {label:<12} {rb['ret']:>+7.2f}% {rr['ret']:>+8.2f}% {diff:>+6.2f}% │ "
          f"{rb['dd']:>6.2f}% {rr['dd']:>9.2f}% │ "
          f"{rb['nc']:>5}笔  {rr['nc']:>7}笔")

    rows_base.append(rb)
    rows_re.append(rr)

print(f"  {'─'*(W-2)}")

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
sr = summary(rows_re)

print(f"\n  {'指标':<16} {'基线':>10} {'+reentry':>10} │ {'变化':>10}")
print(f"  {'─'*55}")
metrics = [
    ("盈利季度",     f"{sb['win_q']}/9",          f"{sr['win_q']}/9",
     sr['win_q']-sb['win_q'],       True,  "q"),
    ("平均季度收益", f"{sb['avg_ret']:>+.2f}%",   f"{sr['avg_ret']:>+.2f}%",
     sr['avg_ret']-sb['avg_ret'],   True,  "pct"),
    ("平均季度回撤", f"{sb['avg_dd']:.2f}%",      f"{sr['avg_dd']:.2f}%",
     sr['avg_dd']-sb['avg_dd'],     False, "pct"),
    ("最大单季回撤", f"{sb['worst_dd']:.2f}%",    f"{sr['worst_dd']:.2f}%",
     sr['worst_dd']-sb['worst_dd'], False, "pct"),
    ("总成交笔数",   f"{sb['trades']}",            f"{sr['trades']}",
     sr['trades']-sb['trades'],     True,  "int"),
]
for label, vb, vr, d, higher_good, kind in metrics:
    if kind == "pct":
        d_str = ("  ─" if abs(d) < 0.01 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+.2f}%").rjust(12)
    else:
        d_str = ("  ─" if d == 0 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+d}").rjust(12)
    print(f"  {label:<16} {vb:>10} {vr:>10} │  {d_str}")

print(f"\n{'='*W}")
