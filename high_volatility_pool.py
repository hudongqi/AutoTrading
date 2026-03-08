import io
import json
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

import ccxt

from data import CCXTDataSource
from strategy import BTCPerpTrendStrategy1H
from portfolio import PerpPortfolio
from broker import SimBroker
from backtest import Backtester
from config import (
    START,
    END,
    INITIAL_CASH,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    SLIPPAGE_BPS,
)


# 固定 5 币观察池
POOL_SYMBOLS_RAW = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]
SYMBOL_MAP = {
    "SOLUSDT": "SOL/USDT:USDT",
    "XRPUSDT": "XRP/USDT:USDT",
    "DOGEUSDT": "DOGE/USDT:USDT",
    "SUIUSDT": "SUI/USDT:USDT",
    # Binance USDⓈ-M 上为 1000PEPE 合约
    "PEPEUSDT": "1000PEPE/USDT:USDT",
}

# 评分与选币规则
MIN_TOTAL_SCORE = 0.55
MAX_DAILY_PICKS = 2

# 仓位类别规则（高波动池内部相对风险系数）
RISK_BUCKET = {
    "SOLUSDT": 1.0,
    "XRPUSDT": 1.0,
    "DOGEUSDT": 0.7,
    "SUIUSDT": 0.7,
    "PEPEUSDT": 0.4,
}


@dataclass
class EventDecision:
    label: str  # ALLOW / REDUCE_RISK / BLOCK
    reason: str


class EventFilterAgent:
    """
    第二个 agent（事件过滤）：只给放行标签，不下单。
    当前版本先做可扩展框架：
    - 宏观窗口（可在 event_signals.json 注入）
    - 热点新闻/币种突发（可在 event_signals.json 注入）
    输出 ALLOW / REDUCE_RISK / BLOCK
    """

    def __init__(self, signal_file: str = "event_signals.json"):
        self.signal_file = signal_file

    def _load_signal_file(self):
        try:
            with open(self.signal_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def evaluate(self, symbol_raw: str, now_utc: datetime) -> EventDecision:
        data = self._load_signal_file()
        macro = data.get("macro", {})
        symbol_events = data.get("symbols", {}).get(symbol_raw, {})

        # 优先级：BLOCK > REDUCE_RISK > ALLOW
        if macro.get("block", False) or symbol_events.get("block", False):
            return EventDecision("BLOCK", "macro/symbol high-risk event window")
        if macro.get("reduce_risk", False) or symbol_events.get("reduce_risk", False):
            return EventDecision("REDUCE_RISK", "elevated uncertainty (macro/news)")

        return EventDecision("ALLOW", "no active macro/news block signal")


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def breakout_stats(df_1h: pd.DataFrame, lookahead_hours: int = 6):
    window = 24 * 7
    out = df_1h.copy()
    out["res7"] = out["high"].rolling(window, min_periods=window).quantile(0.975)
    out["sup7"] = out["low"].rolling(window, min_periods=window).quantile(0.025)

    long_br = out["close"] > out["res7"]
    short_br = out["close"] < out["sup7"]

    fwd_ret = (out["close"].shift(-lookahead_hours) - out["close"]) / out["close"]
    cont = (long_br & (fwd_ret > 0)) | (short_br & (fwd_ret < 0))

    total = int((long_br | short_br).sum())
    cont_n = int(cont.sum())

    continuation_rate = (cont_n / total) if total > 0 else np.nan
    fake_breakout_rate = (1 - continuation_rate) if pd.notna(continuation_rate) else np.nan
    return continuation_rate, fake_breakout_rate


def safe_fetch_exchange_metrics(ex: ccxt.Exchange, symbol_ccxt: str):
    spread = np.nan
    quote_vol_24h = np.nan
    funding_rate = np.nan
    open_interest = np.nan

    try:
        t = ex.fetch_ticker(symbol_ccxt)
        bid, ask = t.get("bid"), t.get("ask")
        if bid and ask and (bid + ask) > 0:
            spread = (ask - bid) / ((ask + bid) / 2)
        quote_vol_24h = t.get("quoteVolume", np.nan)
    except Exception:
        pass

    try:
        fr = ex.fetch_funding_rate(symbol_ccxt)
        funding_rate = fr.get("fundingRate", np.nan)
    except Exception:
        pass

    try:
        oi = ex.fetch_open_interest(symbol_ccxt)
        open_interest = oi.get("openInterestAmount", oi.get("openInterestValue", np.nan))
    except Exception:
        pass

    return dict(
        spread=spread,
        quote_vol_24h=quote_vol_24h,
        funding_rate=funding_rate,
        open_interest=open_interest,
    )


def normalize_cross_section(df: pd.DataFrame, col: str, higher_better: bool = True):
    x = df[col].astype(float)
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=df.index)
    z = (x - mn) / (mx - mn)
    return z if higher_better else (1 - z)


def run_aggressive_backtest_for_symbol(symbol_ccxt: str):
    ds = CCXTDataSource()
    df = ds.load_ohlcv(symbol_ccxt, START, END)

    strat = BTCPerpTrendStrategy1H(
        fast=5,
        slow=15,
        atr_pct_threshold=0.0038,
        use_regime_filter=True,
        adx_threshold_4h=32,
        trend_strength_threshold_4h=0.010,
        slow_slope_lookback_4h=2,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        sig = strat.generate_signals(df)

    pf = PerpPortfolio(
        initial_cash=INITIAL_CASH,
        leverage=3.6,
        taker_fee_rate=TAKER_FEE_RATE,
        maker_fee_rate=MAKER_FEE_RATE,
        maint_margin_rate=0.005,
    )

    bt = Backtester(
        broker=SimBroker(slippage_bps=SLIPPAGE_BPS),
        portfolio=pf,
        strategy=strat,
        max_pos=0.8,
        cooldown_bars=3,
        stop_atr=1.3,
        take_R=2.7,
        trail_start_R=0.8,
        trail_atr=3.0,
        use_trailing=True,
        check_liq=True,
        entry_is_maker=True,
        funding_rate_per_8h=0.0,
    )

    res = bt.run(sig)
    stats = res.attrs.get("stats", {})

    eq = res["equity"]
    max_dd = float((eq / eq.cummax() - 1).min())
    final_equity = float(eq.iloc[-1])

    pnls = np.array(bt.closed_trade_pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    profit_factor = float((wins.sum() / abs(losses.sum())) if len(losses) else np.nan)
    continuation_6h, fake_6h = breakout_stats(df, lookahead_hours=6)
    continuation_12h, fake_12h = breakout_stats(df, lookahead_hours=12)

    return {
        "final_equity": final_equity,
        "max_drawdown": max_dd,
        "trades": int(stats.get("trade_count", 0)),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "pnl_ratio": float(stats.get("pnl_ratio", np.nan)),
        "profit_factor": profit_factor,
        "total_fees": float(stats.get("total_fees", 0.0)),
        "funding_total": float(stats.get("funding_total", 0.0)),
        "breakout_continuation_6h": continuation_6h,
        "fake_breakout_6h": fake_6h,
        "breakout_continuation_12h": continuation_12h,
        "fake_breakout_12h": fake_12h,
    }


def main():
    ds = CCXTDataSource()
    ex = ccxt.binanceusdm({"enableRateLimit": True})

    rows = []
    now = datetime.now(timezone.utc)
    evt = EventFilterAgent()

    for sym_raw in POOL_SYMBOLS_RAW:
        sym = SYMBOL_MAP[sym_raw]
        df_1h = ds.load_ohlcv(sym, START, END, timeframe="1h")
        df_4h = ds.load_ohlcv(sym, START, END, timeframe="4h")

        # 1h / 4h features
        df_1h = df_1h.copy()
        df_1h["atr"] = _atr(df_1h, 14)
        atr_pct = float((df_1h["atr"] / df_1h["close"]).iloc[-1])
        atr_val = float(df_1h["atr"].iloc[-1])

        df_4h = df_4h.copy()
        df_4h["ma_fast"] = df_4h["close"].rolling(5).mean()
        df_4h["ma_slow"] = df_4h["close"].rolling(15).mean()
        trend_dir = int(np.sign((df_4h["ma_fast"] - df_4h["ma_slow"]).iloc[-1]))
        adx_4h = float(_adx(df_4h, 14).iloc[-1])
        trend_strength_4h = float(
            abs((df_4h["ma_fast"] - df_4h["ma_slow"]).iloc[-1]) / df_4h["close"].iloc[-1]
        )

        cont6, fake6 = breakout_stats(df_1h, 6)
        cont12, fake12 = breakout_stats(df_1h, 12)

        exm = safe_fetch_exchange_metrics(ex, sym)
        ev = evt.evaluate(sym_raw, now)

        rows.append(
            {
                "symbol": sym_raw,
                "symbol_ccxt": sym,
                "atr": atr_val,
                "atr_pct": atr_pct,
                "quote_vol_24h": exm["quote_vol_24h"],
                "spread": exm["spread"],
                "open_interest": exm["open_interest"],
                "funding_rate": exm["funding_rate"],
                "trend_dir_4h": trend_dir,
                "adx_4h": adx_4h,
                "trend_strength_4h": trend_strength_4h,
                "continuation_6h": cont6,
                "continuation_12h": cont12,
                "fake_breakout_rate_6h": fake6,
                "fake_breakout_rate_12h": fake12,
                "event_label": ev.label,
                "event_reason": ev.reason,
                "risk_bucket": RISK_BUCKET[sym_raw],
            }
        )

    score_df = pd.DataFrame(rows)

    # 分项打分
    score_df["volatility_score"] = normalize_cross_section(score_df, "atr_pct", higher_better=True)
    # 流动性：高成交额 + 低点差
    vol_score = normalize_cross_section(score_df, "quote_vol_24h", higher_better=True)
    spr_score = normalize_cross_section(score_df, "spread", higher_better=False)
    score_df["liquidity_score"] = 0.7 * vol_score + 0.3 * spr_score

    adx_score = normalize_cross_section(score_df, "adx_4h", higher_better=True)
    ts_score = normalize_cross_section(score_df, "trend_strength_4h", higher_better=True)
    score_df["trend_score"] = 0.6 * adx_score + 0.4 * ts_score

    c6 = score_df["continuation_6h"].fillna(0.0)
    c12 = score_df["continuation_12h"].fillna(0.0)
    score_df["continuation_score"] = 0.6 * c6 + 0.4 * c12

    f6 = score_df["fake_breakout_rate_6h"].fillna(1.0)
    f12 = score_df["fake_breakout_rate_12h"].fillna(1.0)
    score_df["fake_breakout_penalty"] = 0.7 * f6 + 0.3 * f12

    # funding 绝对值越大惩罚越高
    score_df["funding_penalty"] = normalize_cross_section(
        score_df.assign(abs_funding=score_df["funding_rate"].abs().fillna(0.0)),
        "abs_funding",
        higher_better=True,
    )

    # OI 过热惩罚：同池截面 z-score 的正值部分
    oi = score_df["open_interest"].astype(float)
    oi_z = (oi - oi.mean()) / (oi.std(ddof=0) if oi.std(ddof=0) not in [0, np.nan] else 1.0)
    score_df["oi_crowding_penalty"] = oi_z.clip(lower=0).fillna(0.0)
    # 归一化到 [0,1]
    score_df["oi_crowding_penalty"] = normalize_cross_section(score_df, "oi_crowding_penalty", higher_better=True)

    score_df["total_score"] = (
        0.22 * score_df["volatility_score"]
        + 0.20 * score_df["liquidity_score"]
        + 0.20 * score_df["trend_score"]
        + 0.18 * score_df["continuation_score"]
        - 0.10 * score_df["fake_breakout_penalty"]
        - 0.05 * score_df["funding_penalty"]
        - 0.05 * score_df["oi_crowding_penalty"]
    )

    # 事件标签约束
    score_df.loc[score_df["event_label"] == "BLOCK", "total_score"] = -1.0
    score_df.loc[score_df["event_label"] == "REDUCE_RISK", "total_score"] *= 0.85

    # PEPE 额外规则：趋势延续 + 事件都必须通过
    pepe_mask = score_df["symbol"] == "PEPEUSDT"
    score_df.loc[
        pepe_mask
        & (
            (score_df["continuation_score"] < 0.55)
            | (score_df["event_label"] != "ALLOW")
        ),
        "total_score",
    ] = -1.0

    score_df = score_df.sort_values("total_score", ascending=False).reset_index(drop=True)

    # 每天只选前1~2名，且要过阈值
    picks = score_df[score_df["total_score"] >= MIN_TOTAL_SCORE].head(MAX_DAILY_PICKS).copy()

    # 仓位建议（高波动池内部）：高情绪币低权重
    if len(picks) > 0:
        base = picks["risk_bucket"]
        w = base / base.sum()
        picks["pool_weight_suggestion"] = w
        # 如果 REDUCE_RISK，进一步降权
        picks.loc[picks["event_label"] == "REDUCE_RISK", "pool_weight_suggestion"] *= 0.7
        picks["pool_weight_suggestion"] = picks["pool_weight_suggestion"] / picks["pool_weight_suggestion"].sum()
    else:
        picks["pool_weight_suggestion"] = []

    # 回测结果（每个币独立）
    bt_rows = []
    for sym_raw in POOL_SYMBOLS_RAW:
        sym = SYMBOL_MAP[sym_raw]
        m = run_aggressive_backtest_for_symbol(sym)
        bt_rows.append({"symbol": sym_raw, **m})
    bt_df = pd.DataFrame(bt_rows)

    # 输出文件
    score_df.to_csv("high_vol_pool_scores.csv", index=False)
    picks.to_csv("high_vol_pool_daily_picks.csv", index=False)
    bt_df.to_csv("high_vol_pool_backtest_report.csv", index=False)

    # JSON 输出（结构化）
    payload = {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "pool": POOL_SYMBOLS_RAW,
        "rules": {
            "max_daily_picks": MAX_DAILY_PICKS,
            "min_total_score": MIN_TOTAL_SCORE,
            "regime_filter_required": True,
            "aggressive_only": True,
            "btc_core_not_replaced": True,
            "internal_principle": [
                "SOL/XRP 主力池",
                "DOGE/SUI 高弹性补充",
                "PEPE 高情绪高噪音，仅在延续性+事件过滤通过时可交易",
                "不允许因单币波动大而放宽 aggressive 条件",
            ],
        },
        "scores_ranked": score_df.to_dict(orient="records"),
        "today_picks": picks.to_dict(orient="records"),
        "backtest_report": bt_df.to_dict(orient="records"),
    }

    with open("high_vol_pool_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== HIGH VOL POOL SCORE RANKING ===")
    print(
        score_df[
            [
                "symbol",
                "total_score",
                "volatility_score",
                "liquidity_score",
                "trend_score",
                "continuation_score",
                "fake_breakout_penalty",
                "funding_penalty",
                "oi_crowding_penalty",
                "event_label",
            ]
        ].to_string(index=False)
    )

    print("\n=== TODAY PICKS (TOP 1~2) ===")
    if len(picks) == 0:
        print("No symbol passed threshold => high-vol pool stays flat today.")
    else:
        print(
            picks[
                [
                    "symbol",
                    "total_score",
                    "event_label",
                    "pool_weight_suggestion",
                    "risk_bucket",
                ]
            ].to_string(index=False)
        )

    print("\n=== BACKTEST REPORT (AGGRESSIVE + REGIME FILTER) ===")
    print(
        bt_df[
            [
                "symbol",
                "final_equity",
                "max_drawdown",
                "trades",
                "win_rate",
                "pnl_ratio",
                "profit_factor",
                "total_fees",
                "funding_total",
                "breakout_continuation_6h",
                "fake_breakout_6h",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("- high_vol_pool_scores.csv")
    print("- high_vol_pool_daily_picks.csv")
    print("- high_vol_pool_backtest_report.csv")
    print("- high_vol_pool_report.json")


if __name__ == "__main__":
    main()
