"""
对比：原始策略 vs 加入 PEPE 闪跌快进快出模块
季度拆分（2024 Q1 → 2026 Q1）
flash_risk_per_trade = 0.02（每笔 flash 最多亏 2% 权益）
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

FLASH_RISK = 0.02   # 每笔 flash 最多亏 2% 权益


def run_coin(df, strat_params, bt_profile, max_pos, cash, flash_risk=0.0):
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
        dd_stop_pct=0.0, dd_cooldown_bars=0,
        flash_risk_per_trade=flash_risk,
    )
    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])
    eq  = result["equity"]
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - cash) / cash * 100

    # 统计 flash 专项笔数
    flash_trades = [t for t in bt.closed_trades if t.get("entry_reason") == "flash_long"]
    main_trades  = [t for t in bt.closed_trades if t.get("entry_reason") != "flash_long"]

    return dict(ret=ret, dd=dd, eq=eq,
                nc=len(bt.closed_trades),
                flash_nc=len(flash_trades),
                main_nc=len(main_trades))


def run_quarter(df_sol, df_pepe, flash_risk=0.0):
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH, flash_risk=0.0)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH, flash_risk=flash_risk)
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
        flash_nc=(rp["flash_nc"] if rp else 0),
        main_nc=(rs["nc"] if rs else 0) + (rp["main_nc"] if rp else 0),
    )


# ── 预拉全量数据 ─────────────────────────────────────────────
print("拉取全量数据...")
ds = CCXTDataSource()
df_sol_all  = ds.load_ohlcv("SOL/USDT:USDT",      start="2024-01-01", end="2026-03-24", timeframe="1h")
df_pepe_all = ds.load_ohlcv("1000PEPE/USDT:USDT", start="2024-01-01", end="2026-03-24", timeframe="1h")
print(f"SOL {len(df_sol_all)} 根  PEPE {len(df_pepe_all)} 根\n")

# ── 逐季度对比 ────────────────────────────────────────────────
W = 115
print("=" * W)
print("  原始策略 vs 加入 PEPE 闪跌快进快出（flash_risk_per_trade=2%）")
print("=" * W)
hdr = (f"  {'季度':<14}  {'原始收益':>8} {'+Flash收益':>10} {'差值':>7}  │  "
       f"{'原始DD':>7} {'Flash DD':>8}  │  "
       f"{'原始笔':>6} {'主策略笔':>8} {'Flash笔':>7}")
print(hdr)
print("  " + "─" * (W - 2))

rows_base, rows_flash = [], []

for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    df_s = df_sol_all [(df_sol_all.index  >= ts) & (df_sol_all.index  < te)].copy()
    df_p = df_pepe_all[(df_pepe_all.index >= ts) & (df_pepe_all.index < te)].copy()

    rb = run_quarter(df_s, df_p, flash_risk=0.0)
    rf = run_quarter(df_s, df_p, flash_risk=FLASH_RISK)

    _z = dict(ret=0, dd=0, nc=0, flash_nc=0, main_nc=0)
    rb = rb or _z
    rf = rf or _z

    diff = rf["ret"] - rb["ret"]
    mark = "↑" if diff > 0.3 else ("↓" if diff < -0.3 else "─")

    print(f"  {mark} {label:<12}  "
          f"{rb['ret']:>+7.2f}%  {rf['ret']:>+9.2f}%  {diff:>+6.2f}%  │  "
          f"{rb['dd']:>6.2f}%  {rf['dd']:>7.2f}%  │  "
          f"{rb['nc']:>5}笔  {rf['main_nc']:>7}笔  {rf['flash_nc']:>6}笔")

    rows_base.append(rb)
    rows_flash.append(rf)

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
        trades   = sum(r["nc"]    for r in rows),
        flash_nc = sum(r.get("flash_nc", 0) for r in rows),
    )

sb = summary(rows_base)
sf = summary(rows_flash)

print(f"\n  {'指标':<16} {'原始策略':>10} {'+Flash':>10}  │  {'变化':>10}")
print(f"  {'─'*60}")

metrics = [
    ("盈利季度",     f"{sb['win_q']}/9",         f"{sf['win_q']}/9",
     sf['win_q']-sb['win_q'],        False, "q"),
    ("平均季度收益", f"{sb['avg_ret']:>+.2f}%",  f"{sf['avg_ret']:>+.2f}%",
     sf['avg_ret']-sb['avg_ret'],    True,  "pct"),
    ("平均季度回撤", f"{sb['avg_dd']:.2f}%",     f"{sf['avg_dd']:.2f}%",
     sf['avg_dd']-sb['avg_dd'],      False, "pct"),
    ("最大单季回撤", f"{sb['worst_dd']:.2f}%",   f"{sf['worst_dd']:.2f}%",
     sf['worst_dd']-sb['worst_dd'],  False, "pct"),
    ("总成交笔数",   f"{sb['trades']}",           f"{sf['trades']}",
     sf['trades']-sb['trades'],      True,  "int"),
    ("其中 Flash笔", "─",                         f"{sf['flash_nc']}",
     sf['flash_nc'],                 True,  "int"),
]

for label, v_b, v_f, d, higher_good, kind in metrics:
    if kind == "pct":
        d_str = ("  ─" if abs(d) < 0.01 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+.2f}%").rjust(12)
    elif kind == "q":
        d_str = ("  ─" if d == 0 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+d}").rjust(12)
    else:
        d_str = ("  ─" if d == 0 else
                 f"  {'▲' if (d>0)==higher_good else '▽'}{d:+d}").rjust(12)
    print(f"  {label:<16} {v_b:>10} {v_f:>10}  │  {d_str}")

print(f"""
  说明：
  · Flash 模块仅作用于 PEPE（SOL 不启用）
  · 触发条件：前一根 1H K 线下影线（open→low）超过 15%
  · 入场：当前 bar 收盘价，止损 = 触发 bar 低点 × 0.995，止盈 = 入场 × 1.03
  · 仓位大小：flash_risk_per_trade ({FLASH_RISK*100:.0f}%) × 权益 ÷ 止损距离
  · Flash 交易独立于主策略冷静期，但遵守 DD 止损冷静期
""")
print("=" * W)
