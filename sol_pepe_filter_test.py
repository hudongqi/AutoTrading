"""
A/B 对比测试：原始策略 vs BTC过滤 + 资金费率过滤
同样季度拆分（2024 Q1 → 2026 Q1），展示逐季度差异及汇总
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


def inject_btc_cols(df_coin, df_btc):
    """将 BTC 价格及 EMA200 注入到 coin dataframe，按时间对齐。"""
    btc = df_btc[["close"]].copy()
    btc.columns = ["btc_close"]
    btc["btc_ema200"] = btc["btc_close"].ewm(span=200, adjust=False).mean()
    # reindex 到 coin 的时间轴，用前值填充缺口
    btc_aligned = btc.reindex(df_coin.index.union(btc.index)).ffill().reindex(df_coin.index)
    return pd.concat([df_coin, btc_aligned], axis=1)


def inject_funding_col(df_coin, funding_series):
    """将 8H 资金费率 ffill 到 1H，注入到 coin dataframe。"""
    if funding_series is None or len(funding_series) == 0:
        return df_coin
    fr = funding_series.reindex(df_coin.index.union(funding_series.index)).ffill().reindex(df_coin.index)
    df_coin = df_coin.copy()
    df_coin["funding_rate"] = fr
    return df_coin


def run_coin(df, strat_params, bt_profile, max_pos, cash):
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
    )
    result = bt.run(df_sig)
    stats  = result.attrs.get("stats", {})
    closed = result.attrs.get("closed_trades", [])
    eq = result["equity"]
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - cash) / cash * 100
    return dict(
        ret=ret, dd=dd, eq=eq,
        nc=stats.get("closed_trade_count", len(closed)),
        nl=int((df_sig["entry_setup"] == 1).sum()),
        ns=int((df_sig["entry_setup"] == -1).sum()),
    )


def run_quarter(df_sol, df_pepe):
    """运行一个季度的双币种组合，返回 (ret_c, dd_c, nc_s, nc_p, nl_s, ns_s, nl_p, ns_p)"""
    rs = run_coin(df_sol,  SOL_PARAMS,  "sol_rev_v2",  500,       HALF_CASH)
    rp = run_coin(df_pepe, PEPE_PARAMS, "pepe_rev_v2", 5_000_000, HALF_CASH)
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
        nc_s=rs["nc"] if rs else 0, nc_p=rp["nc"] if rp else 0,
        nl_s=rs["nl"] if rs else 0, ns_s=rs["ns"] if rs else 0,
        nl_p=rp["nl"] if rp else 0, ns_p=rp["ns"] if rp else 0,
    )


# ── 预拉全量数据 ────────────────────────────────────────────────
print("拉取全量数据...")
ds = CCXTDataSource()
START, END = "2024-01-01", "2026-03-24"

df_sol_all  = ds.load_ohlcv("SOL/USDT:USDT",      start=START, end=END, timeframe="1h")
df_pepe_all = ds.load_ohlcv("1000PEPE/USDT:USDT", start=START, end=END, timeframe="1h")
df_btc_all  = ds.load_ohlcv("BTC/USDT:USDT",      start=START, end=END, timeframe="1h")
print(f"SOL {len(df_sol_all)}根  PEPE {len(df_pepe_all)}根  BTC {len(df_btc_all)}根")

print("拉取资金费率...")
fr_sol  = ds.load_funding_rates("SOL/USDT:USDT",      start=START, end=END)
fr_pepe = ds.load_funding_rates("1000PEPE/USDT:USDT", start=START, end=END)
print(f"SOL 资金费率 {len(fr_sol)} 条  PEPE 资金费率 {len(fr_pepe)} 条\n")

# ── 逐季度 A/B 对比 ─────────────────────────────────────────────
W = 110
print("=" * W)
print("  A/B 对比：原始策略 vs [BTC过滤 + 资金费率过滤]")
print("=" * W)
hdr = (f"  {'季度':<12}  {'原始':>8} {'过滤后':>8} {'差值':>7}  │  "
       f"{'原始DD':>7} {'过滤DD':>7}  │  "
       f"{'原始成交':>7} {'过滤成交':>8}  │  "
       f"{'信号变化（多/空）':>18}")
print(hdr)
print("  " + "─" * (W - 2))

rows_base, rows_filt = [], []

for start, end, label in QUARTERS:
    ts = pd.Timestamp(start, tz="UTC")
    te = pd.Timestamp(end,   tz="UTC")
    mask_s = (df_sol_all.index  >= ts) & (df_sol_all.index  < te)
    mask_p = (df_pepe_all.index >= ts) & (df_pepe_all.index < te)

    # ── Baseline（无外部过滤）
    df_s_base  = df_sol_all[mask_s].copy()
    df_p_base  = df_pepe_all[mask_p].copy()
    rb = run_quarter(df_s_base, df_p_base)

    # ── Filtered（注入 BTC + 资金费率）
    df_s_filt = inject_btc_cols(df_sol_all[mask_s].copy(), df_btc_all)
    df_p_filt = inject_btc_cols(df_pepe_all[mask_p].copy(), df_btc_all)
    df_s_filt = inject_funding_col(df_s_filt, fr_sol)
    df_p_filt = inject_funding_col(df_p_filt, fr_pepe)
    rf = run_quarter(df_s_filt, df_p_filt)

    if rb is None and rf is None:
        continue

    rb = rb or {"ret": 0, "dd": 0, "nc_s": 0, "nc_p": 0,
                "nl_s": 0, "ns_s": 0, "nl_p": 0, "ns_p": 0}
    rf = rf or {"ret": 0, "dd": 0, "nc_s": 0, "nc_p": 0,
                "nl_s": 0, "ns_s": 0, "nl_p": 0, "ns_p": 0}

    diff = rf["ret"] - rb["ret"]
    nc_b = rb["nc_s"] + rb["nc_p"]
    nc_f = rf["nc_s"] + rf["nc_p"]
    sign_diff = ("+" if diff >= 0 else "") + f"{diff:.1f}%"

    # 信号变化：过滤后 vs 原始（多头/空头）
    nl_b = rb["nl_s"] + rb["nl_p"]
    ns_b = rb["ns_s"] + rb["ns_p"]
    nl_f = rf["nl_s"] + rf["nl_p"]
    ns_f = rf["ns_s"] + rf["ns_p"]
    sig_str = f"多{nl_b}→{nl_f}  空{ns_b}→{ns_f}"

    marker = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "─")
    print(f"  {marker} {label:<10}  {rb['ret']:>+7.1f}%  {rf['ret']:>+7.1f}%  {sign_diff:>7}  │  "
          f"{rb['dd']:>6.1f}%  {rf['dd']:>6.1f}%  │  "
          f"{nc_b:>6}笔   {nc_f:>6}笔  │  {sig_str}")

    rows_base.append(rb)
    rows_filt.append(rf)

print("  " + "─" * (W - 2))

# ── 汇总 ──────────────────────────────────────────────────────────
def summary(rows, label):
    rets = [r["ret"] for r in rows]
    dds  = [r["dd"]  for r in rows]
    pos  = sum(1 for r in rets if r > 0)
    return dict(
        label   = label,
        avg_ret = np.mean(rets),
        win_q   = pos,
        total_q = len(rets),
        avg_dd  = np.mean(dds),
        worst_dd= min(dds),
        trades  = sum(r["nc_s"] + r["nc_p"] for r in rows),
    )

sb = summary(rows_base, "原始策略")
sf = summary(rows_filt, "过滤后策略")

print(f"\n  {'指标':<16} {'原始策略':>12} {'过滤后策略':>12} {'变化':>10}")
print(f"  {'─'*54}")
print(f"  {'盈利季度数':<16} {sb['win_q']}/{sb['total_q']:>10}  {sf['win_q']}/{sf['total_q']:>10}")
print(f"  {'平均季度收益':<16} {sb['avg_ret']:>+11.2f}%  {sf['avg_ret']:>+11.2f}%  "
      f"{sf['avg_ret'] - sb['avg_ret']:>+9.2f}%")
print(f"  {'平均季度回撤':<16} {sb['avg_dd']:>11.2f}%  {sf['avg_dd']:>11.2f}%  "
      f"{sf['avg_dd'] - sb['avg_dd']:>+9.2f}%")
print(f"  {'最大单季回撤':<16} {sb['worst_dd']:>11.2f}%  {sf['worst_dd']:>11.2f}%  "
      f"{sf['worst_dd'] - sb['worst_dd']:>+9.2f}%")
print(f"  {'总成交笔数':<16} {sb['trades']:>12}  {sf['trades']:>12}  "
      f"{sf['trades'] - sb['trades']:>+9}")

print("\n  过滤器说明：")
print("  · BTC 过滤：BTC 收盘 < EMA200 且 EMA200 7日斜率为负 → 压制山寨多头信号")
print("  · 资金费率过滤：多头费率 > 0.08%/8H 时压制多头；空头费率 < -0.05%/8H 时压制空头")
print("\n" + "=" * W)
