"""
历史黑天鹅冲击恢复时间分析
目标：经验性地确定 DD 止损后的冷静期时长
数据：BTC / SOL / PEPE 1H，2020-01 → 2026-03
"""

import pandas as pd
import numpy as np
from data import CCXTDataSource

# ── 参数 ─────────────────────────────────────────────────────
SHOCK_WINDOW   = 72    # 判断冲击的滚动窗口（1H bar，= 3天）
SHOCK_THRESH   = -0.15 # 窗口内跌幅超过 15% 视为冲击
MIN_GAP_BARS   = 240   # 两次冲击之间最小间隔（= 10天，防重复计数）

# 恢复条件窗口（连续 N 根 bar 满足条件才算恢复）
RECOVERY_CONFIRM = 24  # 连续 24H（1天）满足才算稳定恢复

# 恢复条件阈值
ATR_NORM_MULT  = 1.5   # ATR/close < 冲击前基线 × 1.5
EMA_SPAN       = 20    # 短期 EMA
ADX_PERIOD     = 14

# ── 辅助函数 ─────────────────────────────────────────────────
def compute_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx_di(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    up   = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = compute_atr(df, period)
    plus_di  = pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / atr * 100
    minus_di = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di

def find_shocks(close_series):
    """返回冲击事件列表：(bar_index, 滚动跌幅)"""
    roll_ret = close_series.pct_change(SHOCK_WINDOW)
    shocks = []
    last_shock_idx = -MIN_GAP_BARS - 1
    for i, (ts, ret) in enumerate(roll_ret.items()):
        if pd.isna(ret):
            continue
        if ret < SHOCK_THRESH and (i - last_shock_idx) >= MIN_GAP_BARS:
            shocks.append((i, ts, float(ret)))
            last_shock_idx = i
    return shocks

def measure_recovery(df, shock_bar_idx, atr_series, ema_series, plus_di, minus_di):
    """
    从冲击低点（shock_bar_idx）开始，测量恢复所需 bar 数。
    恢复定义：连续 RECOVERY_CONFIRM 根 bar 同时满足：
      1. atr/close < 冲击前 ATR 基线 × ATR_NORM_MULT
      2. close > EMA20
      3. DI+ > DI-
    返回恢复 bar 数（未恢复则返回 None）
    """
    # 冲击前 ATR 基线：用冲击前 5 天（120H）中位数
    baseline_start = max(0, shock_bar_idx - 120)
    atr_pct_baseline = (atr_series / df["close"]).iloc[baseline_start:shock_bar_idx].median()
    if pd.isna(atr_pct_baseline) or atr_pct_baseline <= 0:
        return None

    # 从低点后的 bar 开始检测
    n = len(df)
    confirm_count = 0
    for j in range(shock_bar_idx + 1, n):
        cur_close = float(df["close"].iloc[j])
        cur_atr_pct = float(atr_series.iloc[j]) / cur_close if cur_close > 0 else 9
        cond_vol  = cur_atr_pct < atr_pct_baseline * ATR_NORM_MULT
        cond_ema  = cur_close > float(ema_series.iloc[j])
        cond_di   = float(plus_di.iloc[j]) > float(minus_di.iloc[j])

        if cond_vol and cond_ema and cond_di:
            confirm_count += 1
            if confirm_count >= RECOVERY_CONFIRM:
                return j - shock_bar_idx  # 从低点到恢复的 bar 数
        else:
            confirm_count = 0  # 条件中断，重置

    return None  # 在数据范围内未恢复


# ── 主流程 ────────────────────────────────────────────────────
print("拉取数据（BTC / SOL / PEPE，2020-2026）...")
ds = CCXTDataSource()

# BTC 从 2020 年起以覆盖更多历史黑天鹅
df_btc  = ds.load_ohlcv("BTC/USDT:USDT",       start="2020-01-01", end="2026-03-24", timeframe="1h")
df_sol  = ds.load_ohlcv("SOL/USDT:USDT",        start="2021-01-01", end="2026-03-24", timeframe="1h")
df_pepe = ds.load_ohlcv("1000PEPE/USDT:USDT",   start="2023-05-01", end="2026-03-24", timeframe="1h")

datasets = [
    ("BTC", df_btc),
    ("SOL", df_sol),
    ("PEPE", df_pepe),
]

all_recovery_bars = []   # 收集所有币种的恢复时间
event_rows = []

for symbol, df in datasets:
    print(f"\n  [{symbol}] {len(df)} 根 K 线，分析中...")

    atr_s    = compute_atr(df, ADX_PERIOD)
    ema_s    = df["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    _, pd_s, md_s = compute_adx_di(df, ADX_PERIOD)

    shocks = find_shocks(df["close"])
    print(f"  [{symbol}] 检测到 {len(shocks)} 次冲击事件（{SHOCK_WINDOW}H内跌幅>{abs(SHOCK_THRESH)*100:.0f}%）")

    for bar_idx, ts, shock_ret in shocks:
        recovery = measure_recovery(df, bar_idx, atr_s, ema_s, pd_s, md_s)
        recovery_h  = recovery if recovery is not None else None
        recovery_d  = f"{recovery/24:.1f}天" if recovery is not None else "未恢复"

        # 冲击前后价格
        shock_price  = float(df["close"].iloc[bar_idx])
        pre_price    = float(df["close"].iloc[max(0, bar_idx - SHOCK_WINDOW)])

        print(f"    {str(ts)[:10]}  跌幅 {shock_ret*100:>+6.1f}%  "
              f"恢复 bar: {str(recovery_h):>5}  ({recovery_d})")

        event_rows.append(dict(
            symbol=symbol, date=str(ts)[:10],
            drop_pct=shock_ret * 100,
            recovery_bars=recovery_h,
            recovery_days=recovery_h / 24 if recovery_h else None,
        ))
        if recovery_h is not None:
            all_recovery_bars.append(recovery_h)

# ── 统计汇总 ──────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  历史黑天鹅冲击恢复时间统计")
print("=" * 65)

recovered = [r for r in event_rows if r["recovery_bars"] is not None]
unrecovered = [r for r in event_rows if r["recovery_bars"] is None]

print(f"\n  总冲击事件：{len(event_rows)} 次（{len(recovered)} 次恢复，{len(unrecovered)} 次未恢复）\n")

if recovered:
    bars = [r["recovery_bars"] for r in recovered]
    days = [r["recovery_days"] for r in recovered]

    print(f"  {'统计量':<16} {'bar数':>8} {'天数':>8}")
    print(f"  {'─'*34}")
    for label, func in [("最短", min), ("中位数", np.median),
                         ("平均", np.mean), ("75百分位", lambda x: np.percentile(x, 75)),
                         ("90百分位", lambda x: np.percentile(x, 90)), ("最长", max)]:
        b = func(bars)
        d = func(days)
        print(f"  {label:<16} {b:>8.0f}  {d:>8.1f}天")

    print(f"\n  按币种细分：")
    for sym in ["BTC", "SOL", "PEPE"]:
        sym_bars = [r["recovery_bars"] for r in recovered if r["symbol"] == sym]
        if sym_bars:
            print(f"    {sym:<6} 均值 {np.mean(sym_bars)/24:.1f}天  "
                  f"中位数 {np.median(sym_bars)/24:.1f}天  "
                  f"P75 {np.percentile(sym_bars,75)/24:.1f}天  "
                  f"（样本 {len(sym_bars)} 次）")

    median_bars = int(np.median(bars))
    p75_bars    = int(np.percentile(bars, 75))
    p90_bars    = int(np.percentile(bars, 90))

    print(f"""
  ── 建议冷静期（用于 DD 止损恢复） ──────────────────────
  保守（P75）：{p75_bars} bar  = {p75_bars/24:.0f} 天
  适中（中位数）：{median_bars} bar  = {median_bars/24:.0f} 天
  激进（P25）：{int(np.percentile(bars,25))} bar  = {np.percentile(bars,25)/24:.0f} 天

  推荐：使用 P75 = {p75_bars} bar（{p75_bars/24:.0f} 天）作为冷静期
  理由：75% 的历史冲击在此时间内恢复，平衡"保护"与"不错过行情"
    """)

if unrecovered:
    print(f"  未恢复事件（数据区间内信号质量持续低下）：")
    for r in unrecovered:
        print(f"    {r['symbol']}  {r['date']}  跌幅 {r['drop_pct']:>+.1f}%")

print("=" * 65)
