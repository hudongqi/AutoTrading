"""
PEPE 抄底策略 · 止损/止盈参数扫描
核心问题：赔率如何设置才能让期望值最大化？
测试 5 组不同止损参考点 × 4 种止盈倍数 = 20 个组合
"""

import pandas as pd
import numpy as np
from data import CCXTDataSource

# ── 冲击检测（固定不变）──────────────────────────────────
SHOCK_WINDOW = 72
SHOCK_THRESH = -0.15
MIN_GAP_BARS = 240

# ── 入场信号（固定不变）──────────────────────────────────
RSI_PERIOD      = 14
RSI_EXTREME     = 32
VOL_MA_PERIOD   = 20
VOL_CAP_MULT    = 1.8
STABLE_LOOKBACK = 6
MAX_WAIT_BARS   = 120
MAX_HOLD_BARS   = 120

# ── 止损参考点（5种）─────────────────────────────────────
# 每种定义：(名称, 低点回看H, ATR倍数)
# 止损价 = rolling_min(low, lookback_H) - atr_mult × ATR
STOP_CONFIGS = [
    ("紧: 6H低-0.3ATR",   6,  0.3),
    ("标: 12H低-0.5ATR",  12, 0.5),   # 当前默认
    ("中: 12H低-1.0ATR",  12, 1.0),
    ("宽: 24H低-0.5ATR",  24, 0.5),
    ("宽+: 24H低-1.0ATR", 24, 1.0),
]

# ── 止盈倍数（4种）───────────────────────────────────────
TAKE_RS = [1.5, 2.0, 2.5, 3.0]


# ── 指标 ─────────────────────────────────────────────────
def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def compute_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def find_shocks(close):
    roll_ret = close.pct_change(SHOCK_WINDOW)
    shocks, last_idx = [], -MIN_GAP_BARS - 1
    for i, (ts, ret) in enumerate(roll_ret.items()):
        if pd.isna(ret): continue
        if ret < SHOCK_THRESH and (i - last_idx) >= MIN_GAP_BARS:
            shocks.append((i, ts, float(ret)))
            last_idx = i
    return shocks

def find_entry(df, shock_bar_idx, rsi, vol_ma):
    n = len(df)
    for i in range(shock_bar_idx, min(shock_bar_idx + MAX_WAIT_BARS, n)):
        if pd.isna(rsi.iloc[i]) or rsi.iloc[i] >= RSI_EXTREME: continue
        if pd.isna(vol_ma.iloc[i]) or df["volume"].iloc[i] < vol_ma.iloc[i] * VOL_CAP_MULT: continue
        if df["close"].iloc[i] <= df["open"].iloc[i]: continue
        if i < STABLE_LOOKBACK: continue
        if df["low"].iloc[i] < df["low"].iloc[i - STABLE_LOOKBACK:i].min(): continue
        return i
    return None

def simulate(df, entry_bar, atr, stop_lookback, stop_atr_mult, take_R):
    """用指定止损/止盈参数模拟一笔交易，返回结果字典。"""
    entry_price = float(df["close"].iloc[entry_bar])
    atr_val     = float(atr.iloc[entry_bar])
    low_ref     = df["low"].iloc[max(0, entry_bar - stop_lookback):entry_bar + 1].min()
    stop        = float(low_ref) - stop_atr_mult * atr_val
    risk        = entry_price - stop

    if risk <= 0 or np.isnan(risk) or risk / entry_price > 0.50:
        return None   # 风险超过 50% 本金，跳过（异常数据）

    target  = entry_price + take_R * risk
    risk_pct = risk / entry_price * 100
    n = len(df)

    for j in range(entry_bar + 1, min(entry_bar + MAX_HOLD_BARS + 1, n)):
        if df["low"].iloc[j] <= stop:
            # 止损后追踪 72H 继续跌幅
            post_end = min(j + 72, n)
            post_low = df["low"].iloc[j:post_end].min()
            further  = (post_low - stop) / stop * 100
            verdict  = "飞刀" if further < -5 else "被洗/横"
            return dict(outcome="STOP", pnl_R=-1.0, hold=j-entry_bar,
                        risk_pct=risk_pct, further_drop=further, verdict=verdict)
        if df["high"].iloc[j] >= target:
            return dict(outcome="TAKE", pnl_R=take_R, hold=j-entry_bar,
                        risk_pct=risk_pct, further_drop=None, verdict=None)

    exit_price = float(df["close"].iloc[min(entry_bar + MAX_HOLD_BARS, n-1)])
    pnl_R = (exit_price - entry_price) / risk
    return dict(outcome="TIME", pnl_R=float(pnl_R), hold=MAX_HOLD_BARS,
                risk_pct=risk_pct, further_drop=None, verdict=None)


# ══════════════════════════════════════════════════════════
print("拉取 PEPE 数据...")
ds   = CCXTDataSource()
df   = ds.load_ohlcv("1000PEPE/USDT:USDT", "2023-05-01", "2026-03-24", "1h")
rsi  = compute_rsi(df["close"], RSI_PERIOD)
atr  = compute_atr(df, 14)
vol_ma = df["volume"].rolling(VOL_MA_PERIOD).mean()

shocks = find_shocks(df["close"])
print(f"PEPE 冲击事件：{len(shocks)} 次")

# 先找出所有有效入场 bar（固定，不随止损变化）
entries = []
for shock_bar, shock_ts, shock_ret in shocks:
    eb = find_entry(df, shock_bar, rsi, vol_ma)
    if eb is not None:
        entries.append((shock_bar, shock_ts, shock_ret, eb))
print(f"有效入场信号：{len(entries)} 次（共 {len(shocks)} 次冲击）\n")

# ── 逐组合模拟 ────────────────────────────────────────────
results = {}   # key: (stop_name, take_R)

for stop_name, stop_lb, stop_atr in STOP_CONFIGS:
    for take_R in TAKE_RS:
        trades = []
        for shock_bar, shock_ts, shock_ret, entry_bar in entries:
            t = simulate(df, entry_bar, atr, stop_lb, stop_atr, take_R)
            if t is None: continue
            t.update(date=str(shock_ts)[:10], drop=shock_ret*100)
            trades.append(t)
        results[(stop_name, take_R)] = trades

# ── 打印每笔明细（以默认止损为基准，展示各止盈倍数的差异）──
W = 110
print("=" * W)
print("  逐笔明细（止损 = 12H低-0.5ATR，对比 4 种止盈）")
print("=" * W)
print(f"  {'日期':<12} {'跌幅':>7} │ "
      f"{'1.5R':>7} {'2.0R':>7} {'2.5R':>7} {'3.0R':>7} │ 止损后（标准止损）")
print("  " + "─" * (W-2))

base_stop = "标: 12H低-0.5ATR"
for shock_bar, shock_ts, shock_ret, entry_bar in entries:
    row_parts = []
    stop_verdict = ""
    for tr in TAKE_RS:
        t = next((x for x in results[(base_stop, tr)]
                  if x["date"] == str(shock_ts)[:10]), None)
        if t is None:
            row_parts.append("  跳过")
            continue
        icon = "✓" if t["outcome"]=="TAKE" else ("✗" if t["outcome"]=="STOP" else "─")
        row_parts.append(f"{icon}{t['pnl_R']:>+.1f}R")
        if t["outcome"] == "STOP" and stop_verdict == "":
            stop_verdict = t.get("verdict","") + (f"↓{t['further_drop']:.1f}%" if t.get("further_drop") and t["further_drop"]<-5 else "")
    print(f"  {str(shock_ts)[:10]}  {shock_ret*100:>+6.1f}% │ "
          + "  ".join(f"{p:>7}" for p in row_parts)
          + f" │ {stop_verdict}")

# ── 汇总对比表 ────────────────────────────────────────────
print(f"\n\n{'='*W}")
print("  止损/止盈参数组合 · 综合对比表")
print(f"{'='*W}")
print(f"\n  胜率需满足 > 1/(1+take_R) 才能保证正期望：")
for tr in TAKE_RS:
    be = 1/(1+tr)*100
    print(f"    止盈 {tr}R → 最低胜率 {be:.0f}%")

print(f"\n  {'止损方式':<20} {'止盈':>5} │ {'触发':>4} {'胜率':>6} {'平均R':>7} "
      f"{'期望R':>7} │ {'飞刀%':>6} {'平均风险%':>9}")
print("  " + "─" * 80)

# 计算并收集所有结果，按期望值排序
all_rows = []
for (stop_name, take_R), trades in results.items():
    if not trades: continue
    wins  = [t for t in trades if t["outcome"]=="TAKE"]
    stops = [t for t in trades if t["outcome"]=="STOP"]
    times = [t for t in trades if t["outcome"]=="TIME"]
    wr    = len(wins)/len(trades)*100
    avg_r = np.mean([t["pnl_R"] for t in trades])
    exp   = (len(wins)/len(trades)*take_R
             + len(stops)/len(trades)*(-1.0)
             + (len(times)/len(trades)*np.mean([t["pnl_R"] for t in times]) if times else 0))
    knife_pct = sum(1 for t in stops if t.get("verdict")=="飞刀")/len(stops)*100 if stops else 0
    avg_risk  = np.mean([t["risk_pct"] for t in trades])
    all_rows.append((stop_name, take_R, len(trades), wr, avg_r, exp, knife_pct, avg_risk))

# 按期望值降序排列
all_rows.sort(key=lambda x: -x[5])

best_exp = all_rows[0][5]
for i, (sn, tr, n, wr, avg_r, exp, kf, risk) in enumerate(all_rows):
    marker = " ◀ 最优" if i == 0 else (" ◀ 次优" if i == 1 else "")
    be_wr  = 1/(1+tr)*100
    wr_ok  = "✅" if wr >= be_wr else "❌"
    print(f"  {sn:<20} {tr:>5.1f}R │ {n:>4}  {wr:>5.0f}%{wr_ok}  {avg_r:>+6.2f}R  "
          f"{exp:>+6.2f}R │ {kf:>5.0f}%   {risk:>7.1f}%{marker}")

# ── 推荐的 3 组赔率 ──────────────────────────────────────
print(f"\n\n{'='*W}")
print("  推荐参考赔率（PEPE 抄底专用）")
print(f"{'='*W}")

# 找最优：最高期望值
r1 = all_rows[0]
# 次优：期望值前5中风险最低
conservative = sorted([r for r in all_rows[:6] if r[7] < r1[7]], key=lambda x: x[7])
r2 = conservative[0] if conservative else all_rows[1]
# 激进：take_R >= 2.5 中最高期望值
aggressive = sorted([r for r in all_rows if r[1] >= 2.5], key=lambda x: -x[5])
r3 = aggressive[0] if aggressive else all_rows[2]

for label, r in [("A 保守稳健", r2), ("B 最优期望", r1), ("C 激进高赔", r3)]:
    sn, tr, n, wr, avg_r, exp, kf, risk = r
    be_wr = 1/(1+tr)*100
    print(f"""
  [{label}]
    止损：{sn}   止盈：{tr}R
    历史胜率：{wr:.0f}%（需 >{be_wr:.0f}% 才盈利）
    每笔风险：约 {risk:.1f}% 本金
    平均收益：{avg_r:+.2f}R / 笔
    期望值：  {exp:+.2f}R / 笔
    飞刀率：  {kf:.0f}%（止损后继续跌 >5% 的比例）""")

print(f"""
  选用建议：
  · 若担心接飞刀 → 选 A（止损更宽，飞刀率低，但单笔风险大）
  · 追求最高长期期望 → 选 B
  · 相信 PEPE V形反弹明显 → 选 C（止盈高，需要耐心等待）
  · 无论哪组：仓位控制在总资金 2-3%（这是投机性交易，不是主策略）
""")
print("=" * W)
