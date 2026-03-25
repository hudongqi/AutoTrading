"""
A: 样本外测试（OOS）
2023-05-01 → 2024-01-01（从未参与任何优化的数据）
使用当前全部优化后的参数直接测试
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
PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.2,   # 优化后
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
}
# 月度拆分（OOS 区间较短，用月代替季度）
MONTHS = [
    ("2023-05-01", "2023-06-01", "2023-05"),
    ("2023-06-01", "2023-07-01", "2023-06"),
    ("2023-07-01", "2023-08-01", "2023-07"),
    ("2023-08-01", "2023-09-01", "2023-08"),
    ("2023-09-01", "2023-10-01", "2023-09"),
    ("2023-10-01", "2023-11-01", "2023-10"),
    ("2023-11-01", "2023-12-01", "2023-11"),
    ("2023-12-01", "2024-01-01", "2023-12"),
]

def run(df, cooldown):
    if len(df) < 50:
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
        max_pos=5_000_000, cooldown_bars=cooldown,
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
    coin_ret = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    return dict(ret=ret, dd=dd, wr=wr, pf=pf, nc=nc, coin_ret=coin_ret)

print("拉取 OOS 数据（2023-05 → 2024-01）...")
ds  = CCXTDataSource()
df_oos = ds.load_ohlcv("1000PEPE/USDT:USDT", "2023-05-01", "2024-01-01", "1h")
df_is  = ds.load_ohlcv("1000PEPE/USDT:USDT", "2024-01-01", "2026-03-24", "1h")
print(f"OOS {len(df_oos)} 根  |  IS {len(df_is)} 根\n")

r_oos = run(df_oos, cooldown=3)
r_is  = run(df_is,  cooldown=3)

W = 82
print("=" * W)
print("  样本外测试（OOS）vs 样本内（IS）")
print("  参数：vol_spike=1.2  cooldown=3  take_R=4.0R  stop_atr=5.0")
print("=" * W)
print(f"  {'区间':<28} {'收益':>8} {'最大回撤':>9} {'Calmar':>7} {'胜率':>5} {'成交':>5} {'标的':>8}")
print(f"  {'─'*(W-2)}")
for label, r in [
    ("样本内  IS（2024-01 → 2026-03）", r_is),
    ("样本外 OOS（2023-05 → 2024-01）", r_oos),
]:
    if r is None:
        print(f"  {label:<28}  数据不足")
        continue
    cal  = r["ret"] / abs(r["dd"]) if r["dd"] != 0 else 0
    pf_s = f"{r['pf']:.2f}" if r["pf"] not in (float("inf"), 0) else "∞"
    print(f"  {label:<28} {r['ret']:>+7.1f}%  {r['dd']:>7.1f}%  {cal:>7.2f}  "
          f"{r['wr']:>4.0f}%  {r['nc']:>3}笔  {r['coin_ret']:>+6.1f}%")

# 月度拆分
print(f"\n{'='*W}")
print("  OOS 月度拆分（2023-05 → 2024-01）")
print("=" * W)
print(f"  {'月份':<10} {'PEPE标的':>8} │ {'策略收益':>9} {'回撤':>7} {'胜率':>5} {'成交':>5}")
print(f"  {'─'*(W-2)}")

rows = []
for start, end, label in MONTHS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_m = df_oos[(df_oos.index >= ts) & (df_oos.index < te)].copy()
    r = run(df_m, cooldown=3)
    if r is None:
        print(f"  {label:<10}  数据不足")
        continue
    sign = "✓" if r["ret"] > 0 else ("─" if r["nc"] == 0 else "✗")
    print(f"  {sign} {label:<8} {r['coin_ret']:>+7.1f}% │ "
          f"{r['ret']:>+8.2f}% {r['dd']:>6.2f}% {r['wr']:>4.0f}%  {r['nc']:>3}笔")
    rows.append(r)

valid = [r for r in rows if r["nc"] > 0]
if valid:
    rets = [r["ret"] for r in valid]
    print(f"\n  有交易的月份：{len(valid)}/{len(rows)}")
    print(f"  盈利月份：{sum(1 for r in rets if r>0)}/{len(valid)}")
    print(f"  平均月收益：{np.mean(rets):>+.2f}%")
    print(f"  最差月份：{min(rets):>+.2f}%")
print("=" * W)
