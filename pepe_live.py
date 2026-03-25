#!/usr/bin/env python3
"""
PEPE 实盘运行器 — 1000PEPE/USDT:USDT 永续合约
========================================================
设计为每小时收盘后 2 分钟由 cron 触发一次：
  2 * * * * cd /path/to/AutoTrading && .venv/bin/python pepe_live.py >> logs/pepe_cron.log 2>&1

环境变量：
  BINANCE_API_KEY       Binance API Key
  BINANCE_API_SECRET    Binance API Secret
  PEPE_LIVE_DRY_RUN     1=只记录不下单(默认)  0=真实下单
  TELEGRAM_BOT_TOKEN    Telegram 机器人 Token（可选）
  TELEGRAM_CHAT_ID      Telegram 接收 Chat ID（可选）
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import numpy as np
import requests

from strategy import SOLReversionV2Strategy1H

# ═══════════════════════════════════════════════════════
#  策略参数（与回测完全一致，禁止随意修改）
# ═══════════════════════════════════════════════════════
SYMBOL_CCXT = "1000PEPE/USDT:USDT"
TIMEFRAME   = "1h"
DATA_LIMIT  = 350       # 拉取根数（策略预热需约 200 根）

STRAT_PARAMS = {
    "atr_period": 14, "bb_period": 20, "bb_std": 2.0,
    "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60,
    "reclaim_ema": 20, "trend_ema": 200,
    "vol_period": 20, "vol_spike_mult": 1.2,
    "atr_pct_low": 0.008, "atr_pct_high": 0.20,
    "oversold_lookback": 3, "allow_short": True,
}
STOP_ATR     = 5.0      # 止损 = entry ± 5 × ATR
TAKE_R       = 4.0      # 止盈 = entry ± 4 × 止损距离
COOLDOWN     = 3        # 平仓后等待 3 根 K 线再入场
MAX_HOLD     = 240      # 最多持仓 240 根（10 天）强制平仓
LEVERAGE     = 2        # 杠杆（需在 Binance 后台预先设置）
RISK_PT      = 0.040    # 每笔风险 4% 净值
MAX_QTY      = 5_000_000  # 单笔最大数量上限（单位：1000PEPE）

# ═══════════════════════════════════════════════════════
#  运行配置（从环境变量读取）
# ═══════════════════════════════════════════════════════
DRY_RUN   = os.getenv("PEPE_LIVE_DRY_RUN", "1") == "1"
TG_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")
API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SEC   = os.getenv("BINANCE_API_SECRET", "")

STATE_PATH = Path("state/pepe_live_state.json")
LOG_PATH   = Path("logs/pepe_live.log")

# ═══════════════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════════════
LOG_PATH.parent.mkdir(exist_ok=True)
STATE_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pepe_live")


# ═══════════════════════════════════════════════════════
#  Telegram 通知
# ═══════════════════════════════════════════════════════
def tg(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram 发送失败: {e}")


# ═══════════════════════════════════════════════════════
#  状态管理（持久化到 JSON）
# ═══════════════════════════════════════════════════════
EMPTY_STATE = {
    "position": None,       # None 或 {side, entry_price, stop_price, take_price, qty, sl_id, tp_id, entry_bar, bars_held}
    "cooldown_remaining": 0,
    "last_bar_time": None,
    "trade_count": 0,
    "initial_equity": None,
}


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(EMPTY_STATE)


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════
#  交易所连接
# ═══════════════════════════════════════════════════════
def build_exchange() -> ccxt.binanceusdm:
    ex = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": API_SEC,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    return ex


# ═══════════════════════════════════════════════════════
#  数据拉取 + 信号生成
# ═══════════════════════════════════════════════════════
def fetch_signals(ex: ccxt.binanceusdm) -> pd.DataFrame:
    """返回含 entry_setup / atr 等列的信号 DataFrame（已去掉当前正在形成的 K 线）"""
    raw = ex.fetch_ohlcv(SYMBOL_CCXT, timeframe=TIMEFRAME, limit=DATA_LIMIT)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # 去掉正在形成的当前 K 线（最后一根），只使用已收盘的 K 线
    df = df.iloc[:-1]

    strat = SOLReversionV2Strategy1H(**STRAT_PARAMS)
    return strat.generate_signals(df)


# ═══════════════════════════════════════════════════════
#  仓位大小计算（与回测一致）
# ═══════════════════════════════════════════════════════
def calc_qty(equity: float, atr: float) -> tuple[float, float]:
    """返回 (qty, stop_distance)"""
    stop_distance = STOP_ATR * atr
    if stop_distance <= 0:
        return 0.0, 0.0
    qty = (equity * RISK_PT) / stop_distance
    qty = min(qty, MAX_QTY)
    return round(qty, 0), stop_distance


# ═══════════════════════════════════════════════════════
#  交易所下单封装（支持 dry_run）
# ═══════════════════════════════════════════════════════
def place_market(ex, side: str, qty: float) -> dict:
    if DRY_RUN:
        log.info(f"[DRY] market {side} {qty:.0f}")
        return {"id": "DRY_MARKET", "average": None, "status": "closed"}
    return ex.create_order(SYMBOL_CCXT, "market", side, qty)


def place_sl(ex, side: str, qty: float, stop_price: float) -> dict:
    close_side = "sell" if side == "buy" else "buy"
    params = {"stopPrice": stop_price, "reduceOnly": True}
    if DRY_RUN:
        log.info(f"[DRY] stop_market {close_side} {qty:.0f} @ {stop_price:.8f}")
        return {"id": "DRY_SL"}
    return ex.create_order(SYMBOL_CCXT, "stop_market", close_side, qty, None, params)


def place_tp(ex, side: str, qty: float, take_price: float) -> dict:
    close_side = "sell" if side == "buy" else "buy"
    params = {"stopPrice": take_price, "reduceOnly": True}
    if DRY_RUN:
        log.info(f"[DRY] take_profit_market {close_side} {qty:.0f} @ {take_price:.8f}")
        return {"id": "DRY_TP"}
    return ex.create_order(SYMBOL_CCXT, "take_profit_market", close_side, qty, None, params)


def cancel_order(ex, order_id: str):
    if DRY_RUN or order_id.startswith("DRY"):
        return
    try:
        ex.cancel_order(order_id, SYMBOL_CCXT)
    except Exception as e:
        log.warning(f"撤单 {order_id} 失败: {e}")


def close_position_market(ex, side: str, qty: float):
    """强制市价平仓"""
    close_side = "sell" if side == "long" else "buy"
    if DRY_RUN:
        log.info(f"[DRY] 强制平仓 {close_side} {qty:.0f}")
        return
    try:
        ex.create_order(SYMBOL_CCXT, "market", close_side, qty, None, {"reduceOnly": True})
    except Exception as e:
        log.error(f"强制平仓失败: {e}")
        tg(f"⚠️ PEPE 强制平仓失败，请手动处理！\n{e}")


def get_order_status(ex, order_id: str) -> str:
    """返回 'open' / 'closed' / 'canceled' / 'unknown'"""
    if DRY_RUN or order_id.startswith("DRY"):
        return "open"
    try:
        od = ex.fetch_order(order_id, SYMBOL_CCXT)
        return str(od.get("status", "unknown"))
    except Exception as e:
        log.warning(f"查询订单 {order_id} 失败: {e}")
        return "unknown"


def get_equity(ex) -> float:
    if DRY_RUN:
        return 10_000.0
    try:
        bal = ex.fetch_balance()
        return float(bal.get("USDT", {}).get("total", 10_000.0))
    except Exception as e:
        log.error(f"获取余额失败: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════
#  主逻辑
# ═══════════════════════════════════════════════════════
def main():
    mode_tag = "[DRY RUN]" if DRY_RUN else "[LIVE]"
    log.info(f"{'='*60}")
    log.info(f"PEPE 实盘运行器启动 {mode_tag}  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    state = load_state()
    ex    = build_exchange()

    # ── 1. 拉数据 + 生成信号 ────────────────────────────
    try:
        df_sig = fetch_signals(ex)
    except Exception as e:
        log.error(f"数据拉取失败，跳过本轮: {e}")
        tg(f"⚠️ PEPE 数据拉取失败: {e}")
        return

    last_bar = df_sig.iloc[-1]
    bar_time = str(df_sig.index[-1])
    signal   = int(last_bar.get("entry_setup", 0))
    atr      = float(last_bar.get("atr", 0.0))
    close    = float(last_bar["close"])

    log.info(f"最新已收盘 K 线: {bar_time}  close={close:.8f}  ATR={atr:.8f}  signal={signal}")

    # 防止同一根 K 线重复处理
    if bar_time == state.get("last_bar_time"):
        log.info("当前 K 线已处理过，跳过")
        return
    state["last_bar_time"] = bar_time

    # ── 2. 持仓管理：检查 SL / TP 是否已触发 ────────────
    pos = state.get("position")
    if pos:
        sl_status = get_order_status(ex, pos["sl_id"])
        tp_status = get_order_status(ex, pos["tp_id"])
        bars_held = int(pos.get("bars_held", 0)) + 1
        pos["bars_held"] = bars_held

        exited = False
        exit_reason = ""

        if sl_status == "closed":
            exited = True
            exit_reason = "止损"
        elif tp_status == "closed":
            exited = True
            exit_reason = "止盈"
        elif bars_held >= MAX_HOLD:
            # 超过最大持仓 K 线数，强制平仓
            log.warning(f"持仓超过 {MAX_HOLD} 根 K 线，强制平仓")
            cancel_order(ex, pos["sl_id"])
            cancel_order(ex, pos["tp_id"])
            close_position_market(ex, pos["side"], pos["qty"])
            exited = True
            exit_reason = f"超时({MAX_HOLD}bars)"

        if exited:
            log.info(f"仓位已平：{exit_reason}  side={pos['side']}  qty={pos['qty']:.0f}"
                     f"  entry={pos['entry_price']:.8f}  bars={bars_held}")
            tg(f"📤 PEPE 平仓 <b>{exit_reason}</b>\n"
               f"方向: {'做多' if pos['side']=='long' else '做空'}\n"
               f"入场: {pos['entry_price']:.8f}\n"
               f"持仓: {bars_held} 根K线\n"
               f"止损线: {pos['stop_price']:.8f}\n"
               f"止盈线: {pos['take_price']:.8f}")

            # 撤销未触发的另一侧订单
            if sl_status == "closed":
                cancel_order(ex, pos["tp_id"])
            elif tp_status == "closed":
                cancel_order(ex, pos["sl_id"])

            state["position"] = None
            state["cooldown_remaining"] = COOLDOWN
            state["trade_count"] = state.get("trade_count", 0) + 1
            pos = None

    # ── 3. 冷却倒计时 ────────────────────────────────────
    if state["cooldown_remaining"] > 0:
        state["cooldown_remaining"] -= 1
        log.info(f"冷却中，剩余 {state['cooldown_remaining']} 根 K 线")

    # ── 4. 入场判断 ──────────────────────────────────────
    if pos is None and state["cooldown_remaining"] == 0 and signal != 0 and atr > 0:
        equity = get_equity(ex)
        if state.get("initial_equity") is None:
            state["initial_equity"] = equity

        qty, stop_dist = calc_qty(equity, atr)
        if qty <= 0:
            log.warning("计算仓位为 0，跳过入场")
            save_state(state)
            return

        side      = "buy"  if signal == 1 else "sell"
        direction = "long" if signal == 1 else "short"

        # 用当前收盘价估算入场价（实际以市价成交为准）
        est_entry = close
        if signal == 1:
            stop_price = est_entry - stop_dist
            take_price = est_entry + TAKE_R * stop_dist
        else:
            stop_price = est_entry + stop_dist
            take_price = est_entry - TAKE_R * stop_dist

        log.info(f"信号触发 {direction}  qty={qty:.0f}  预计入场={est_entry:.8f}"
                 f"  SL={stop_price:.8f}  TP={take_price:.8f}  净值={equity:.2f}")

        # 下单
        try:
            entry_od = place_market(ex, side, qty)
            actual_entry = float(entry_od.get("average") or est_entry)

            # 用实际成交价重新计算 SL / TP
            if signal == 1:
                stop_price = actual_entry - stop_dist
                take_price = actual_entry + TAKE_R * stop_dist
            else:
                stop_price = actual_entry + stop_dist
                take_price = actual_entry - TAKE_R * stop_dist

            sl_od = place_sl(ex, side, qty, stop_price)
            tp_od = place_tp(ex, side, qty, take_price)

            state["position"] = {
                "side":        direction,
                "entry_price": actual_entry,
                "stop_price":  stop_price,
                "take_price":  take_price,
                "qty":         qty,
                "sl_id":       sl_od.get("id", "DRY_SL"),
                "tp_id":       tp_od.get("id", "DRY_TP"),
                "entry_bar":   bar_time,
                "bars_held":   0,
            }

            log.info(f"入场成功  成交价={actual_entry:.8f}  SL_ID={sl_od.get('id')}  TP_ID={tp_od.get('id')}")
            tg(f"📥 PEPE 入场 <b>{'做多 🟢' if direction=='long' else '做空 🔴'}</b>\n"
               f"成交价: {actual_entry:.8f}\n"
               f"数量: {qty:.0f} (1000PEPE)\n"
               f"止损: {stop_price:.8f}  ({STOP_ATR:.0f}×ATR)\n"
               f"止盈: {take_price:.8f}  ({TAKE_R:.0f}R)\n"
               f"风险金额: ${equity * RISK_PT:.2f}  净值: ${equity:.2f}\n"
               f"{'[DRY RUN]' if DRY_RUN else ''}")

        except Exception as e:
            log.error(f"下单失败: {e}")
            tg(f"🚨 PEPE 下单失败！\n{e}")

    else:
        if pos:
            log.info(f"持仓中  side={pos['side']}  bars_held={pos.get('bars_held',0)}"
                     f"  entry={pos['entry_price']:.8f}  SL={pos['stop_price']:.8f}  TP={pos['take_price']:.8f}")
        elif signal == 0:
            log.info("无信号")
        elif state["cooldown_remaining"] > 0:
            pass  # 已在上方记录

    save_state(state)
    log.info("本轮完成")


if __name__ == "__main__":
    main()
