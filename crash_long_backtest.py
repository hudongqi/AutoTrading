"""
路线 A（第二轮）：暴跌抄底策略 · 放宽参数 + 止损质量分析
新增：止损触发后追踪——是"接了飞刀"还是"被洗出去了"
"""

import pandas as pd
import numpy as np
from data import CCXTDataSource

# ── 冲击检测（与 shock_recovery_analysis.py 一致）──
SHOCK_WINDOW  = 72
SHOCK_THRESH  = -0.15
MIN_GAP_BARS  = 240

# ── 抄底信号参数（放宽版）────────────────────────────
RSI_PERIOD      = 14
RSI_EXTREME     = 32     # 放宽：25 → 32
VOL_MA_PERIOD   = 20
VOL_CAP_MULT    = 1.8    # 放宽：2.5 → 1.8
STABLE_LOOKBACK = 6      # 放宽：12H → 6H
MAX_WAIT_BARS   = 120    # 冲击后最多等 5 天找入场

# ── 交易参数 ─────────────────────────────────────────
ATR_PERIOD    = 14
STOP_ATR_MULT = 0.5      # 止损 = 12H最低点 下方 0.5 ATR
TAKE_R        = 1.5      # 止盈 1.5R
MAX_HOLD_BARS = 120      # 时间止损 5 天

# ── 止损后追踪窗口 ────────────────────────────────────
POST_STOP_BARS = 72      # 止损触发后观察 72H（3天）价格走向


# ── 指标 ─────────────────────────────────────────────
def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

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
        if pd.isna(ret):
            continue
        if ret < SHOCK_THRESH and (i - last_idx) >= MIN_GAP_BARS:
            shocks.append((i, ts, float(ret)))
            last_idx = i
    return shocks


def find_entry(df, shock_bar_idx, rsi, vol_ma):
    n = len(df)
    for i in range(shock_bar_idx, min(shock_bar_idx + MAX_WAIT_BARS, n)):
        if pd.isna(rsi.iloc[i]) or rsi.iloc[i] >= RSI_EXTREME:
            continue
        if pd.isna(vol_ma.iloc[i]) or df["volume"].iloc[i] < vol_ma.iloc[i] * VOL_CAP_MULT:
            continue
        if df["close"].iloc[i] <= df["open"].iloc[i]:
            continue
        if i < STABLE_LOOKBACK:
            continue
        if df["low"].iloc[i] < df["low"].iloc[i - STABLE_LOOKBACK:i].min():
            continue
        return i
    return None


def simulate_trade(df, entry_bar_idx, atr):
    """返回交易结果，并额外记录止损后 POST_STOP_BARS 内的价格走势。"""
    entry_price = float(df["close"].iloc[entry_bar_idx])
    atr_val     = float(atr.iloc[entry_bar_idx])
    low_12h     = df["low"].iloc[max(0, entry_bar_idx - 12):entry_bar_idx + 1].min()
    stop        = float(low_12h) - STOP_ATR_MULT * atr_val
    risk        = entry_price - stop

    if risk <= 0 or np.isnan(risk):
        return None

    target = entry_price + TAKE_R * risk
    n      = len(df)
    stop_bar = None

    for j in range(entry_bar_idx + 1, min(entry_bar_idx + MAX_HOLD_BARS + 1, n)):
        if df["low"].iloc[j] <= stop:
            stop_bar = j
            break
        if df["high"].iloc[j] >= target:
            return dict(
                outcome="TAKE", pnl_R=TAKE_R,
                exit_price=target, hold_bars=j - entry_bar_idx,
                entry_price=entry_price, stop=stop, target=target, risk=risk,
                post_stop_drop=None, post_stop_verdict=None,
            )

    if stop_bar is not None:
        # ── 止损后追踪 ────────────────────────────────
        post_end = min(stop_bar + POST_STOP_BARS, n)
        post_low = df["low"].iloc[stop_bar:post_end].min()
        post_close = float(df["close"].iloc[post_end - 1])
        further_drop = (post_low - stop) / stop * 100   # 相对止损价的继续跌幅
        post_close_chg = (post_close - stop) / stop * 100

        # 判定：止损后继续大跌 = 飞刀；止损后反弹 = 被洗
        if further_drop < -5:
            verdict = f"飞刀 ↓{further_drop:.1f}%"
        elif post_close_chg > 3:
            verdict = f"被洗 ↑{post_close_chg:.1f}%"
        else:
            verdict = f"横盘 {post_close_chg:+.1f}%"

        return dict(
            outcome="STOP", pnl_R=-1.0,
            exit_price=stop, hold_bars=stop_bar - entry_bar_idx,
            entry_price=entry_price, stop=stop, target=target, risk=risk,
            post_stop_drop=further_drop,
            post_stop_close_chg=post_close_chg,
            post_stop_verdict=verdict,
        )

    # 时间止损
    exit_bar   = min(entry_bar_idx + MAX_HOLD_BARS, n - 1)
    exit_price = float(df["close"].iloc[exit_bar])
    pnl_R      = (exit_price - entry_price) / risk
    return dict(
        outcome="TIME", pnl_R=float(pnl_R),
        exit_price=exit_price, hold_bars=MAX_HOLD_BARS,
        entry_price=entry_price, stop=stop, target=target, risk=risk,
        post_stop_drop=None, post_stop_verdict=None,
    )


# ══════════════════════════════════════════════════════
print("拉取数据...")
ds = CCXTDataSource()
datasets = [
    ("BTC",  ds.load_ohlcv("BTC/USDT:USDT",      "2020-01-01", "2026-03-24", "1h")),
    ("SOL",  ds.load_ohlcv("SOL/USDT:USDT",       "2021-01-01", "2026-03-24", "1h")),
    ("PEPE", ds.load_ohlcv("1000PEPE/USDT:USDT",  "2023-05-01", "2026-03-24", "1h")),
]

all_trades = []
W = 120

print(f"\n{'='*W}")
print(f"  暴跌抄底策略（放宽版）· 逐事件回测 + 止损质量分析")
print(f"  入场：RSI<{RSI_EXTREME} + 放量×{VOL_CAP_MULT} + 收阳 + {STABLE_LOOKBACK}H底部稳定")
print(f"  出场：止盈{TAKE_R}R  止损12H低点-{STOP_ATR_MULT}ATR  时间止损{MAX_HOLD_BARS}H")
print(f"  止损追踪：触发后 {POST_STOP_BARS}H 内最大跌幅 & 收盘变化")
print(f"{'='*W}")

for symbol, df in datasets:
    rsi    = compute_rsi(df["close"], RSI_PERIOD)
    atr    = compute_atr(df, ATR_PERIOD)
    vol_ma = df["volume"].rolling(VOL_MA_PERIOD).mean()
    shocks = find_shocks(df["close"])

    print(f"\n  [{symbol}]  {len(shocks)} 次冲击\n")
    print(f"  {'日期':<12} {'跌幅':>7} │ {'入场':>5} {'延迟':>6} │ "
          f"{'结果':>5} {'PnL':>6} {'持仓':>5} │ 止损后追踪（{POST_STOP_BARS}H）")
    print(f"  {'─'*W}")

    for shock_bar_idx, shock_ts, shock_ret in shocks:
        entry_bar = find_entry(df, shock_bar_idx, rsi, vol_ma)

        if entry_bar is None:
            print(f"  {str(shock_ts)[:10]}  {shock_ret*100:>+6.1f}% │  ✗ 无信号")
            continue

        trade = simulate_trade(df, entry_bar, atr)
        if trade is None:
            print(f"  {str(shock_ts)[:10]}  {shock_ret*100:>+6.1f}% │  ✗ 止损无效")
            continue

        delay = entry_bar - shock_bar_idx
        icon  = "✓" if trade["outcome"] == "TAKE" else ("✗" if trade["outcome"] == "STOP" else "─")
        pnl_s = f"{'+' if trade['pnl_R']>0 else ''}{trade['pnl_R']:.2f}R"

        post_str = ""
        if trade["outcome"] == "STOP" and trade.get("post_stop_verdict"):
            post_str = f"→ {trade['post_stop_verdict']}"
        elif trade["outcome"] == "TAKE":
            post_str = "（止盈，无需追踪）"
        elif trade["outcome"] == "TIME":
            post_str = f"超时出场 {'+' if trade['pnl_R']>0 else ''}{trade['pnl_R']:.2f}R"

        print(f"  {str(shock_ts)[:10]}  {shock_ret*100:>+6.1f}% │  "
              f"✓  {delay:>4}H │ "
              f"{icon} {trade['outcome']:>5}  {pnl_s:>6}  {trade['hold_bars']:>3}H │ {post_str}")

        trade.update(symbol=symbol, date=str(shock_ts)[:10], drop=shock_ret*100)
        all_trades.append(trade)

# ── 全局汇总 ─────────────────────────────────────────
print(f"\n\n{'='*W}")
print(f"  全币种汇总")
print(f"{'='*W}")

taken  = [t for t in all_trades if "outcome" in t]
wins   = [t for t in taken if t["outcome"] == "TAKE"]
stops  = [t for t in taken if t["outcome"] == "STOP"]
times  = [t for t in taken if t["outcome"] == "TIME"]

total_shocks = sum(len(find_shocks(df["close"])) for _, df in datasets)
print(f"\n  冲击事件总数:  {total_shocks}")
print(f"  信号触发率:    {len(taken)}/{total_shocks}  ({len(taken)/total_shocks*100:.0f}%)\n")

if taken:
    wr    = len(wins) / len(taken) * 100
    avg_r = np.mean([t["pnl_R"] for t in taken])
    exp   = (len(wins)/len(taken)*TAKE_R
             + len(stops)/len(taken)*(-1.0)
             + (len(times)/len(taken)*np.mean([t["pnl_R"] for t in times]) if times else 0))

    print(f"  成交笔数:  {len(taken)}  │  TAKE {len(wins)}  STOP {len(stops)}  TIME {len(times)}")
    print(f"  胜率:      {wr:.1f}%")
    print(f"  平均PnL:   {avg_r:+.2f}R")
    print(f"  期望值:    {exp:+.2f}R/笔\n")

    # ── 止损质量分析 ──────────────────────────────────
    if stops:
        knives  = [t for t in stops if t.get("post_stop_drop") is not None
                   and t["post_stop_drop"] < -5]
        washed  = [t for t in stops if t.get("post_stop_close_chg") is not None
                   and t["post_stop_close_chg"] > 3]
        lateral = [t for t in stops if t not in knives and t not in washed
                   and t.get("post_stop_verdict") is not None]

        print(f"  ── 止损质量分析（共 {len(stops)} 次止损）──────────────────────")
        print(f"  飞刀（止损后继续跌>5%）: {len(knives)} 笔  "
              f"({len(knives)/len(stops)*100:.0f}%)  ← 止损正确，避免更大损失")
        print(f"  被洗（止损后反弹>3%）:   {len(washed)} 笔  "
              f"({len(washed)/len(stops)*100:.0f}%)  ← 止损过早，错失反弹")
        print(f"  横盘（止损后震荡）:       {len(lateral)} 笔  "
              f"({len(lateral)/len(stops)*100:.0f}%)")

        if knives:
            avg_saved = np.mean([abs(t["post_stop_drop"]) for t in knives])
            print(f"\n  飞刀详情（止损正确的案例）：")
            for t in knives:
                print(f"    {t['symbol']} {t['date']}  跌幅{t['drop']:>+.1f}%  "
                      f"止损后{POST_STOP_BARS}H最大再跌 {t['post_stop_drop']:.1f}%  "
                      f"→ 止损为你节省了 {abs(t['post_stop_drop']):.1f}% 额外损失")
            print(f"  平均节省跌幅: {avg_saved:.1f}%")

        if washed:
            avg_missed = np.mean([t["post_stop_close_chg"] for t in washed])
            print(f"\n  被洗详情（止损过早的案例）：")
            for t in washed:
                print(f"    {t['symbol']} {t['date']}  跌幅{t['drop']:>+.1f}%  "
                      f"止损后{POST_STOP_BARS}H收盘反弹 {t['post_stop_close_chg']:+.1f}%  "
                      f"→ 若不止损可额外获利约 {t['post_stop_close_chg']:.1f}%")
            print(f"  平均错失反弹: {avg_missed:.1f}%")

    # ── 按币种汇总 ────────────────────────────────────
    print(f"\n  ── 按币种细分 ───────────────────────────────────────────")
    print(f"  {'币种':<6} {'触发':>5} {'胜率':>6} {'平均R':>7} {'期望R':>7} │ "
          f"{'飞刀':>5} {'被洗':>5} {'横盘':>5}")
    print(f"  {'─'*65}")
    for sym in ["BTC", "SOL", "PEPE"]:
        st = [t for t in taken if t["symbol"] == sym]
        sw = [t for t in st if t["outcome"] == "TAKE"]
        ss = [t for t in st if t["outcome"] == "STOP"]
        if not st:
            continue
        s_wr  = len(sw)/len(st)*100
        s_avg = np.mean([t["pnl_R"] for t in st])
        s_exp = len(sw)/len(st)*TAKE_R + len(ss)/len(st)*(-1.0)
        knf   = sum(1 for t in ss if t.get("post_stop_drop") and t["post_stop_drop"] < -5)
        wsh   = sum(1 for t in ss if t.get("post_stop_close_chg") and t["post_stop_close_chg"] > 3)
        lat   = len(ss) - knf - wsh
        print(f"  {sym:<6} {len(st):>5}  {s_wr:>5.0f}%  {s_avg:>+6.2f}R  {s_exp:>+6.2f}R │ "
              f"{knf:>5} {wsh:>5} {lat:>5}")

    # ── 综合判断 ──────────────────────────────────────
    trigger_ok = len(taken)/total_shocks >= 0.20
    exp_ok     = exp > 0.15
    knife_ok   = len(knives)/len(stops) >= 0.5 if stops else True

    print(f"\n  ── 综合评估 ─────────────────────────────────────────────")
    print(f"  信号触发率 ≥20%:       {'✅' if trigger_ok else '❌'}  ({len(taken)/total_shocks*100:.0f}%)")
    print(f"  期望值 ≥+0.15R:        {'✅' if exp_ok else '❌'}  ({exp:+.2f}R)")
    print(f"  止损多为飞刀（≥50%）:  {'✅' if knife_ok else '❌'}  "
          f"({len(knives)}/{len(stops)} = {len(knives)/len(stops)*100:.0f}%)" if stops else "  止损多为飞刀：无止损样本")

    all_pass = trigger_ok and exp_ok
    print(f"\n  结论: {'→ 可考虑集成（路线 B）' if all_pass else '→ 信号需进一步优化，或仅限 PEPE 应用'}")

print(f"\n{'='*W}")
