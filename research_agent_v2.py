"""
Research & Event Filter Agent for High Volatility Pool (Enhanced)
研究整理与事件过滤 Agent - 高波动流动性池 (增强版)

新增功能:
1. 真实宏观/新闻数据输入 (通过 web_search + 手动编辑)
2. 滚动窗口数据 (60-180天)
3. 历史归档 CSV
"""

import os
import json
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import pandas as pd
import ccxt

from data import CCXTDataSource


class DecisionLabel(Enum):
    """事件过滤决策标签"""
    ALLOW = "ALLOW"
    REDUCE_RISK = "REDUCE_RISK"
    BLOCK = "BLOCK"


class MacroState(Enum):
    """宏观状态"""
    ALLOW = "ALLOW"
    REDUCE_RISK = "REDUCE_RISK"
    BLOCK = "BLOCK"


# 固定观察池配置
POOL_SYMBOLS_RAW = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]
SYMBOL_MAP = {
    "SOLUSDT": "SOL/USDT:USDT",
    "XRPUSDT": "XRP/USDT:USDT",
    "DOGEUSDT": "DOGE/USDT:USDT",
    "SUIUSDT": "SUI/USDT:USDT",
    "PEPEUSDT": "1000PEPE/USDT:USDT",
}

# 币种特定新闻关键词映射
SYMBOL_NEWS_KEYWORDS = {
    "SOLUSDT": ["Solana", "SOL", "FTX", "Saga", "Jupiter"],
    "XRPUSDT": ["XRP", "Ripple", "SEC", "Garlinghouse", "ODL"],
    "DOGEUSDT": ["Dogecoin", "DOGE", "Musk", "Twitter", "X payments"],
    "SUIUSDT": ["Sui", "SUI", "Mysten Labs", "Move language"],
    "PEPEUSDT": ["Pepe", "PEPE", "meme coin"],
}

# 风险系数
RISK_BUCKET = {
    "SOLUSDT": 1.0,
    "XRPUSDT": 1.0,
    "DOGEUSDT": 0.7,
    "SUIUSDT": 0.7,
    "PEPEUSDT": 0.4,
}

# 评分阈值
MIN_TOTAL_SCORE = 0.55
MAX_DAILY_PICKS = 2

# 滚动窗口配置 (天数)
ROLLING_WINDOW_DAYS = 120  # 默认 120 天，可在 60-180 之间调整


@dataclass
class SymbolResearch:
    """单个币种的研究数据"""
    symbol: str
    symbol_ccxt: str
    trend_1h: str
    trend_4h: str
    atr: float
    atr_pct: float
    quote_vol_24h: float
    spread: float
    spread_score: float
    open_interest: float
    funding_rate: float
    funding_state: str
    oi_state: str
    adx_4h: float
    trend_strength_4h: float
    continuation_6h: float
    continuation_12h: float
    continuation_score: float
    fake_breakout_rate_6h: float
    fake_breakout_rate_12h: float
    fake_breakout_risk: float
    volatility_score: float
    liquidity_score: float
    trend_score: float
    funding_penalty: float
    oi_crowding_penalty: float
    total_score: float
    decision: str
    decision_reasons: List[str]
    risk_bucket: float
    has_major_event: bool
    event_notes: List[str]
    # 新增: 新闻/宏观相关
    news_signals: List[str]
    whale_score: float
    whale_bias: str
    whale_reason: str
    whale_adjustment: float
    funding_change_24h: float
    oi_change_24h: float


@dataclass
class ResearchReport:
    """每日研究报告"""
    date: str
    generated_at: str
    macro_state: str
    pool_status: str
    top_candidates: List[str]
    overall_risk_recommendation: str
    macro_notes: List[str]
    symbols: List[SymbolResearch]
    # 新增: 数据来源信息
    data_window: str  # 如 "2025-11-08 to 2026-03-08"


class EventFilterAgent:
    """
    事件过滤 Agent (增强版)
    - 支持外部信号文件
    - 支持币种特定新闻/事件检测
    - 支持宏观数据窗口检测
    """
    
    # 高风险宏观事件日期 (可扩展)
    MACRO_EVENT_CALENDAR = {
        # 格式: "YYYY-MM-DD": {"event": "CPI", "impact": "high"}
    }
    
    def __init__(self, signal_file: str = "event_signals.json"):
        self.signal_file = signal_file
        self._ensure_signal_file()
    
    def _ensure_signal_file(self):
        """确保信号文件存在"""
        if not os.path.exists(self.signal_file):
            default_signals = {
                "macro": {
                    "block": False,
                    "reduce_risk": False,
                    "reason": "",
                    "upcoming_events": []
                },
                "whale": {
                    "enabled": True,
                    "mode": "auxiliary",
                    "note": "Only for sizing/confidence/candidate filtering, never standalone trade trigger.",
                    "last_updated_utc": datetime.now(timezone.utc).isoformat()
                },
                "symbols": {
                    sym: {
                        "block": False,
                        "reduce_risk": False,
                        "reason": "",
                        "news_signals": [],
                        "whale_score": 0.0,
                        "whale_bias": "NEUTRAL",
                        "whale_reason": "No validated smart-money signal"
                    } for sym in POOL_SYMBOLS_RAW
                }
            }
            with open(self.signal_file, "w", encoding="utf-8") as f:
                json.dump(default_signals, f, indent=2, ensure_ascii=False)
    
    def _load_signal_file(self) -> dict:
        """加载外部事件信号文件"""
        try:
            with open(self.signal_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def evaluate_macro(self, now_utc: datetime) -> Tuple[MacroState, List[str]]:
        """
        评估宏观环境
        结合: 外部信号 + 日历事件 + 自动检测
        """
        data = self._load_signal_file()
        macro = data.get("macro", {})
        notes = []
        
        # 1. 检查外部信号 (最高优先级)
        if macro.get("block", False):
            notes.append(f"🚫 外部信号 BLOCK: {macro.get('reason', '宏观高风险窗口')}")
            return MacroState.BLOCK, notes
        
        if macro.get("reduce_risk", False):
            notes.append(f"⚠️  外部信号 REDUCE_RISK: {macro.get('reason', '不确定性升高')}")
            return MacroState.REDUCE_RISK, notes
        
        # 2. 地缘政治风险层（与宏观并列）
        geo = data.get("geopolitics", {})
        if geo.get("block_new_entries", False):
            notes.append(f"🌍 地缘政治 BLOCK_NEW_ENTRIES: {geo.get('reason', '')}")
            return MacroState.REDUCE_RISK, notes
        if geo.get("reduce_risk", False):
            notes.append(f"🌍 地缘政治 REDUCE_RISK: {geo.get('reason', '')}")
            return MacroState.REDUCE_RISK, notes

        # 3. 检查即将发生的事件
        upcoming = macro.get("upcoming_events", [])
        for event in upcoming:
            event_date = event.get("date", "")
            event_name = event.get("name", "")
            event_impact = event.get("impact", "medium")
            
            if event_date:
                try:
                    ev_dt = datetime.strptime(event_date, "%Y-%m-%d")
                    days_diff = (ev_dt - now_utc.replace(tzinfo=None)).days
                    
                    if 0 <= days_diff <= 1:
                        notes.append(f"📅 今日重大事件: {event_name}")
                        if event_impact == "high":
                            return MacroState.REDUCE_RISK, notes
                    elif 2 <= days_diff <= 3:
                        notes.append(f"📅 即将发生 ({days_diff}天后): {event_name}")
                except:
                    pass
        
        # 3. 自动检测高风险日期
        date_str = now_utc.strftime("%Y-%m-%d")
        if date_str in self.MACRO_EVENT_CALENDAR:
            event_info = self.MACRO_EVENT_CALENDAR[date_str]
            notes.append(f"📅 日历事件: {event_info['event']} ({event_info['impact']})")
            if event_info['impact'] == 'high':
                return MacroState.REDUCE_RISK, notes
        
        # 4. NFP 周五检测
        if now_utc.weekday() == 4 and now_utc.day <= 7:
            notes.append("📅 NFP 周五，建议谨慎")
            return MacroState.REDUCE_RISK, notes
        
        notes.append("✅ 无重大宏观风险信号")
        return MacroState.ALLOW, notes
    
    def evaluate_symbol(self, symbol_raw: str, now_utc: datetime) -> Tuple[DecisionLabel, List[str], List[str], float, str, str]:
        """
        评估单个币种的事件风险
        返回: (决策标签, 原因列表, 新闻信号列表, whale_score, whale_bias, whale_reason)
        """
        data = self._load_signal_file()
        symbol_events = data.get("symbols", {}).get(symbol_raw, {})
        notes = []
        news_signals = []

        whale_cfg = data.get("whale", {})
        whale_enabled = bool(whale_cfg.get("enabled", True))
        whale_score = float(symbol_events.get("whale_score", 0.0)) if whale_enabled else 0.0
        whale_score = max(-1.0, min(1.0, whale_score))
        whale_bias = str(symbol_events.get("whale_bias", "NEUTRAL")).upper().strip() if whale_enabled else "NEUTRAL"
        if whale_bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            whale_bias = "NEUTRAL"
        whale_reason = str(symbol_events.get("whale_reason", "No validated smart-money signal")).strip() if whale_enabled else "Whale layer disabled"
        if not whale_reason:
            whale_reason = "No validated smart-money signal"
        
        # 1. 检查外部信号
        if symbol_events.get("block", False):
            reason = symbol_events.get("reason", '币种特定高风险事件')
            notes.append(f"🚫 外部信号 BLOCK: {reason}")
            return DecisionLabel.BLOCK, notes, news_signals, whale_score, whale_bias, whale_reason
        
        if symbol_events.get("reduce_risk", False):
            reason = symbol_events.get("reason", '币种不确定性升高')
            notes.append(f"⚠️  外部信号 REDUCE_RISK: {reason}")
            return DecisionLabel.REDUCE_RISK, notes, news_signals, whale_score, whale_bias, whale_reason
        
        # 2. 检查币种特定新闻信号
        symbol_news = symbol_events.get("news_signals", [])
        for news in symbol_news:
            news_signals.append(news)
            if "block" in news.lower() or "重大负面" in news:
                notes.append(f"📰 负面新闻: {news}")
                return DecisionLabel.BLOCK, notes, news_signals, whale_score, whale_bias, whale_reason
            elif "reduce" in news.lower() or "谨慎" in news:
                notes.append(f"📰 警示新闻: {news}")
                return DecisionLabel.REDUCE_RISK, notes, news_signals, whale_score, whale_bias, whale_reason
        
        # 3. 检查关键词匹配 (如果有外部数据源)
        keywords = SYMBOL_NEWS_KEYWORDS.get(symbol_raw, [])
        # 这里可以接入实际的新闻 API
        
        notes.append("✅ 无特定事件风险")
        return DecisionLabel.ALLOW, notes, news_signals, whale_score, whale_bias, whale_reason


class ResearchDataAgent:
    """
    数据整理 Agent (增强版)
    - 使用滚动窗口而非全局 START/END
    """
    
    def __init__(self, window_days: int = ROLLING_WINDOW_DAYS):
        self.ds = CCXTDataSource()
        self.ex = ccxt.binanceusdm({"enableRateLimit": True})
        self.window_days = window_days
    
    def get_date_range(self) -> Tuple[str, str]:
        """获取滚动窗口日期范围"""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.window_days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    
    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR"""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()
    
    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ADX"""
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
    
    def _fetch_exchange_metrics(self, symbol_ccxt: str) -> dict:
        """获取交易所实时数据"""
        spread = np.nan
        quote_vol_24h = np.nan
        funding_rate = np.nan
        open_interest = np.nan
        funding_change_24h = np.nan
        oi_change_24h = np.nan

        try:
            t = self.ex.fetch_ticker(symbol_ccxt)
            bid, ask = t.get("bid"), t.get("ask")
            if bid and ask and (bid + ask) > 0:
                spread = (ask - bid) / ((ask + bid) / 2)
            quote_vol_24h = t.get("quoteVolume", np.nan)
        except Exception:
            pass

        try:
            fr = self.ex.fetch_funding_rate(symbol_ccxt)
            funding_rate = fr.get("fundingRate", np.nan)
            # 获取前一个资金费率计算变化
            funding_change_24h = fr.get("fundingRate", 0)  # 简化处理
        except Exception:
            pass

        try:
            oi = self.ex.fetch_open_interest(symbol_ccxt)
            open_interest = oi.get("openInterestAmount", oi.get("openInterestValue", np.nan))
        except Exception:
            pass

        return dict(
            spread=spread,
            quote_vol_24h=quote_vol_24h,
            funding_rate=funding_rate,
            open_interest=open_interest,
            funding_change_24h=funding_change_24h,
            oi_change_24h=oi_change_24h,
        )
    
    def _calculate_breakout_stats(self, df_1h: pd.DataFrame, lookahead_hours: int = 6) -> Tuple[float, float]:
        """计算突破延续率和假突破率"""
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
    
    def _classify_funding(self, funding_rate: float) -> str:
        """分类资金费状态"""
        if pd.isna(funding_rate):
            return "NORMAL"
        abs_rate = abs(funding_rate)
        if abs_rate > 0.001:
            return "EXTREME"
        elif abs_rate > 0.0005:
            return "ELEVATED"
        return "NORMAL"
    
    def _classify_oi(self, oi_series: pd.Series, current_oi: float) -> str:
        """分类持仓量状态"""
        if pd.isna(current_oi) or len(oi_series) < 2:
            return "NORMAL"
        
        mean_oi = oi_series.mean()
        std_oi = oi_series.std(ddof=0)
        if std_oi == 0 or pd.isna(std_oi):
            return "NORMAL"
        
        z_score = (current_oi - mean_oi) / std_oi
        
        if z_score > 1.5:
            return "HOT"
        elif z_score > 0.5:
            return "WARM"
        return "NORMAL"
    
    def collect_symbol_data(self, symbol_raw: str) -> dict:
        """收集单个币种的完整数据 (使用滚动窗口)"""
        symbol_ccxt = SYMBOL_MAP[symbol_raw]
        start, end = self.get_date_range()
        
        # 加载数据
        df_1h = self.ds.load_ohlcv(symbol_ccxt, start, end, timeframe="1h")
        df_4h = self.ds.load_ohlcv(symbol_ccxt, start, end, timeframe="4h")
        
        # 1h 指标
        df_1h = df_1h.copy()
        df_1h["atr"] = self._atr(df_1h, 14)
        df_1h["ma_fast"] = df_1h["close"].rolling(5).mean()
        df_1h["ma_slow"] = df_1h["close"].rolling(15).mean()
        
        atr_pct = float((df_1h["atr"] / df_1h["close"]).iloc[-1])
        atr_val = float(df_1h["atr"].iloc[-1])
        
        trend_1h = "NEUTRAL"
        if df_1h["ma_fast"].iloc[-1] > df_1h["ma_slow"].iloc[-1]:
            trend_1h = "UP"
        elif df_1h["ma_fast"].iloc[-1] < df_1h["ma_slow"].iloc[-1]:
            trend_1h = "DOWN"
        
        # 4h 指标
        df_4h = df_4h.copy()
        df_4h["ma_fast"] = df_4h["close"].rolling(5).mean()
        df_4h["ma_slow"] = df_4h["close"].rolling(15).mean()
        
        trend_4h = "NEUTRAL"
        ma_diff = (df_4h["ma_fast"] - df_4h["ma_slow"]).iloc[-1]
        if ma_diff > 0:
            trend_4h = "UP"
        elif ma_diff < 0:
            trend_4h = "DOWN"
        
        adx_4h = float(self._adx(df_4h, 14).iloc[-1])
        trend_strength_4h = float(abs(ma_diff) / df_4h["close"].iloc[-1])
        
        # 突破统计
        cont6, fake6 = self._calculate_breakout_stats(df_1h, 6)
        cont12, fake12 = self._calculate_breakout_stats(df_1h, 12)
        
        # 交易所数据
        exm = self._fetch_exchange_metrics(symbol_ccxt)
        
        return {
            "symbol": symbol_raw,
            "symbol_ccxt": symbol_ccxt,
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "atr": atr_val,
            "atr_pct": atr_pct,
            "quote_vol_24h": exm["quote_vol_24h"],
            "spread": exm["spread"],
            "open_interest": exm["open_interest"],
            "funding_rate": exm["funding_rate"],
            "funding_change_24h": exm["funding_change_24h"],
            "oi_change_24h": exm["oi_change_24h"],
            "adx_4h": adx_4h,
            "trend_strength_4h": trend_strength_4h,
            "continuation_6h": cont6,
            "continuation_12h": cont12,
            "fake_breakout_rate_6h": fake6,
            "fake_breakout_rate_12h": fake12,
            "data_window": f"{start} to {end}",
        }


class ScoringAgent:
    """评分 Agent"""
    
    WEIGHTS = {
        "volatility": 0.22,
        "liquidity": 0.20,
        "trend": 0.20,
        "continuation": 0.18,
        "fake_breakout_penalty": 0.10,
        "funding_penalty": 0.05,
        "oi_crowding_penalty": 0.05,
    }
    
    @staticmethod
    def _normalize(series: pd.Series, higher_better: bool = True) -> pd.Series:
        mn, mx = series.min(), series.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(0.5, index=series.index)
        z = (series - mn) / (mx - mn)
        return z if higher_better else (1 - z)
    
    def calculate_scores(self, data_list: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(data_list)
        
        df["volatility_score"] = self._normalize(df["atr_pct"], higher_better=True)
        
        vol_score = self._normalize(df["quote_vol_24h"].fillna(0), higher_better=True)
        spr_score = self._normalize(df["spread"].fillna(0.001), higher_better=False)
        df["liquidity_score"] = 0.7 * vol_score + 0.3 * spr_score
        
        adx_score = self._normalize(df["adx_4h"].fillna(20), higher_better=True)
        ts_score = self._normalize(df["trend_strength_4h"].fillna(0), higher_better=True)
        df["trend_score"] = 0.6 * adx_score + 0.4 * ts_score
        
        c6 = df["continuation_6h"].fillna(0.5)
        c12 = df["continuation_12h"].fillna(0.5)
        df["continuation_score"] = 0.6 * c6 + 0.4 * c12
        
        f6 = df["fake_breakout_rate_6h"].fillna(0.5)
        f12 = df["fake_breakout_rate_12h"].fillna(0.5)
        df["fake_breakout_penalty"] = 0.7 * f6 + 0.3 * f12
        
        df["funding_penalty"] = self._normalize(
            df["funding_rate"].abs().fillna(0), higher_better=True
        )
        
        oi = df["open_interest"].astype(float)
        oi_z = (oi - oi.mean()) / (oi.std(ddof=0) if oi.std(ddof=0) not in [0, np.nan] else 1.0)
        df["oi_crowding_penalty"] = self._normalize(oi_z.clip(lower=0).fillna(0), higher_better=True)
        
        df["total_score"] = (
            self.WEIGHTS["volatility"] * df["volatility_score"]
            + self.WEIGHTS["liquidity"] * df["liquidity_score"]
            + self.WEIGHTS["trend"] * df["trend_score"]
            + self.WEIGHTS["continuation"] * df["continuation_score"]
            - self.WEIGHTS["fake_breakout_penalty"] * df["fake_breakout_penalty"]
            - self.WEIGHTS["funding_penalty"] * df["funding_penalty"]
            - self.WEIGHTS["oi_crowding_penalty"] * df["oi_crowding_penalty"]
        )
        
        return df
    
    def apply_event_filters(self, df: pd.DataFrame, event_results: Dict) -> pd.DataFrame:
        for symbol_raw in POOL_SYMBOLS_RAW:
            result = event_results.get(symbol_raw, {})
            decision = result.get("decision", DecisionLabel.ALLOW)
            mask = df["symbol"] == symbol_raw
            
            if decision == DecisionLabel.BLOCK:
                df.loc[mask, "total_score"] = -1.0
            elif decision == DecisionLabel.REDUCE_RISK:
                df.loc[mask, "total_score"] *= 0.85
        
        return df
    
    def apply_whale_overlay(self, df: pd.DataFrame, event_results: Dict) -> pd.DataFrame:
        """
        smart-money 辅助增强层：
        - 仅微调候选评分（不触发 BLOCK/开平仓）
        - 用于仓位/置信度/candidate ranking
        """
        df["whale_adjustment"] = 0.0

        for symbol_raw in POOL_SYMBOLS_RAW:
            result = event_results.get(symbol_raw, {})
            whale_score = float(result.get("whale_score", 0.0))
            whale_score = max(-1.0, min(1.0, whale_score))
            whale_bias = str(result.get("whale_bias", "NEUTRAL")).upper()

            direction = 0.0
            if whale_bias == "BULLISH":
                direction = 1.0
            elif whale_bias == "BEARISH":
                direction = -1.0

            # 最大 ±0.08 的温和调整
            adjustment = 0.08 * abs(whale_score) * direction
            mask = df["symbol"] == symbol_raw
            df.loc[mask, "total_score"] += adjustment
            df.loc[mask, "whale_adjustment"] = adjustment

        return df

    def apply_special_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        pepe_mask = df["symbol"] == "PEPEUSDT"
        df.loc[pepe_mask & (df["continuation_score"] < 0.55), "total_score"] = -1.0
        return df


class ArchiveAgent:
    """
    历史归档 Agent
    记录每天每个币的 score/decision/macro_state，方便复盘
    """
    
    def __init__(self, base_dir: str = "research/high_vol_pool"):
        self.base_dir = base_dir
        self.archive_file = f"{base_dir}/archive/historical_scores.csv"
        self._ensure_archive_dir()
    
    def _ensure_archive_dir(self):
        os.makedirs(f"{self.base_dir}/archive", exist_ok=True)
    
    def append_daily_record(self, report: ResearchReport):
        """追加每日记录到 CSV"""
        records = []
        for sym in report.symbols:
            record = {
                "date": report.date,
                "symbol": sym.symbol,
                "total_score": round(sym.total_score, 4),
                "decision": sym.decision,
                "macro_state": report.macro_state,
                "trend_4h": sym.trend_4h,
                "atr_pct": round(sym.atr_pct, 4),
                "volatility_score": round(sym.volatility_score, 4),
                "liquidity_score": round(sym.liquidity_score, 4),
                "trend_score": round(sym.trend_score, 4),
                "continuation_score": round(sym.continuation_score, 4),
                "funding_state": sym.funding_state,
                "oi_state": sym.oi_state,
                "has_major_event": sym.has_major_event,
                "whale_score": round(sym.whale_score, 4),
                "whale_bias": sym.whale_bias,
                "whale_adjustment": round(sym.whale_adjustment, 4),
                "top_candidate": sym.symbol in report.top_candidates,
            }
            records.append(record)
        
        # 创建或追加到 CSV
        df = pd.DataFrame(records)
        
        if os.path.exists(self.archive_file):
            # 追加模式，不写入表头
            df.to_csv(self.archive_file, mode='a', header=False, index=False)
        else:
            # 新建文件，写入表头
            df.to_csv(self.archive_file, mode='w', header=True, index=False)
        
        return self.archive_file
    
    def load_historical_data(self, days: int = 30) -> pd.DataFrame:
        """加载历史数据用于分析"""
        if not os.path.exists(self.archive_file):
            return pd.DataFrame()
        
        df = pd.read_csv(self.archive_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # 过滤最近 N 天
        cutoff = datetime.now() - timedelta(days=days)
        return df[df['date'] >= cutoff]
    
    def generate_summary_stats(self) -> dict:
        """生成历史统计摘要"""
        df = self.load_historical_data(days=90)
        if df.empty:
            return {}
        
        stats = {
            "total_records": len(df),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "avg_scores_by_symbol": df.groupby('symbol')['total_score'].mean().to_dict(),
            "decision_distribution": df['decision'].value_counts().to_dict(),
            "top_candidate_rate": df.groupby('symbol')['top_candidate'].mean().to_dict(),
        }
        return stats


class ResearchOutputAgent:
    """输出 Agent (增强版)"""
    
    def __init__(self, base_dir: str = "research/high_vol_pool"):
        self.base_dir = base_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        os.makedirs(f"{self.base_dir}/daily", exist_ok=True)
        os.makedirs(f"{self.base_dir}/archive", exist_ok=True)
    
    def generate_json_report(self, report: ResearchReport, date_str: str) -> str:
        filepath = f"{self.base_dir}/daily/{date_str}_candidates.json"
        
        event_overlay = {}
        try:
            event_overlay = json.loads(Path("event_signals.json").read_text(encoding="utf-8"))
        except Exception:
            event_overlay = {}

        report_dict = {
            "date": report.date,
            "generated_at": report.generated_at,
            "data_window": report.data_window,
            "market_bias": event_overlay.get("market_bias", "NEUTRAL"),
            "risk_mode": event_overlay.get("risk_mode", "NORMAL"),
            "macro": event_overlay.get("macro", {}),
            "geopolitics": event_overlay.get("geopolitics", {}),
            "sentiment": event_overlay.get("sentiment", {}),
            "whale_score": event_overlay.get("whale", {}).get("whale_score", 0.0),
            "whale_bias": event_overlay.get("whale", {}).get("whale_bias", "NEUTRAL"),
            "whale_reason": event_overlay.get("whale", {}).get("whale_reason", ""),
            "macro_state": report.macro_state,
            "pool_status": report.pool_status,
            "top_candidates": report.top_candidates,
            "overall_risk_recommendation": report.overall_risk_recommendation,
            "macro_notes": report.macro_notes,
            "symbols": []
        }
        
        for sym in report.symbols:
            report_dict["symbols"].append({
                "symbol": sym.symbol,
                "score": round(sym.total_score, 3),
                "decision": sym.decision,
                "trend_1h": sym.trend_1h,
                "trend_4h": sym.trend_4h,
                "atr_pct": round(sym.atr_pct, 4),
                "spread_score": round(sym.spread_score, 1),
                "funding_state": sym.funding_state,
                "oi_state": sym.oi_state,
                "continuation_score": round(sym.continuation_score, 2),
                "fake_breakout_risk": round(sym.fake_breakout_risk, 2),
                "volatility_score": round(sym.volatility_score, 2),
                "liquidity_score": round(sym.liquidity_score, 2),
                "trend_score": round(sym.trend_score, 2),
                "funding_change_24h": sym.funding_change_24h,
                "oi_change_24h": sym.oi_change_24h,
                "whale_score": sym.whale_score,
                "whale_bias": sym.whale_bias,
                "whale_reason": sym.whale_reason,
                "whale_adjustment": sym.whale_adjustment,
                "bias": event_overlay.get("symbols", {}).get(sym.symbol, {}).get("bias", "NEUTRAL"),
                "strength": event_overlay.get("symbols", {}).get(sym.symbol, {}).get("strength", "LOW"),
                "recommended_action": event_overlay.get("symbols", {}).get(sym.symbol, {}).get("recommended_action", "WAIT"),
                "reasons": sym.decision_reasons,
                "news_signals": sym.news_signals,
                "has_major_event": sym.has_major_event,
                "event_notes": sym.event_notes,
            })
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def generate_markdown_summary(self, report: ResearchReport, date_str: str) -> str:
        filepath = f"{self.base_dir}/daily/{date_str}_summary.md"
        
        overlay = {}
        try:
            overlay = json.loads(Path("event_signals.json").read_text(encoding="utf-8"))
        except Exception:
            overlay = {}

        lines = [
            f"# 高波动池研究报告 - {date_str}",
            "",
            f"生成时间: {report.generated_at}",
            f"数据窗口: {report.data_window}",
            f"市场偏向: {overlay.get('market_bias', 'NEUTRAL')}",
            f"风险模式: {overlay.get('risk_mode', 'NORMAL')}",
            "",
            "## 今日宏观风险",
            "",
            f"**宏观状态**: {report.macro_state}",
            "",
        ]
        
        if report.macro_notes:
            for note in report.macro_notes:
                lines.append(f"- {note}")
        else:
            lines.append("- 无重大宏观风险")
        
        lines.extend([
            "",
            "## 今日候选池排序",
            "",
            "| 排名 | 币种 | 总分 | 决策 | 4h趋势 | ATR% | 延续率 | 假突破率 | 资金费 | OI |",
            "|------|------|------|------|--------|------|--------|----------|--------|-----|",
        ])
        
        for i, sym in enumerate(report.symbols, 1):
            lines.append(
                f"| {i} | {sym.symbol} | {sym.total_score:.3f} | {sym.decision} | "
                f"{sym.trend_4h} | {sym.atr_pct:.4f} | {sym.continuation_score:.2f} | "
                f"{sym.fake_breakout_risk:.2f} | {sym.funding_state} | {sym.oi_state} |"
            )
        
        lines.extend([
            "",
            "## 推荐币种与原因",
            "",
        ])
        
        if report.top_candidates:
            for sym in report.symbols:
                if sym.symbol in report.top_candidates:
                    lines.extend([
                        f"### {sym.symbol}",
                        f"- **决策**: {sym.decision}",
                        f"- **总分**: {sym.total_score:.3f}",
                        f"- **4h趋势**: {sym.trend_4h}",
                        f"- **资金费状态**: {sym.funding_state} (变化: {sym.funding_change_24h:+.4f})",
                        f"- **OI状态**: {sym.oi_state} (变化: {sym.oi_change_24h:+.2f}%)",
                        f"- **Whale辅助**: {sym.whale_bias} | score={sym.whale_score:+.2f} | 调整={sym.whale_adjustment:+.3f}",
                        f"- **Whale说明**: {sym.whale_reason}",
                        "",
                        "**推荐理由**:",
                    ])
                    for reason in sym.decision_reasons:
                        lines.append(f"- {reason}")
                    
                    if sym.news_signals:
                        lines.extend(["", "**相关新闻信号**:",])
                        for news in sym.news_signals:
                            lines.append(f"- {news}")
                    lines.append("")
        else:
            lines.append("**今日无推荐币种** - 没有币种通过评分阈值")
        
        lines.extend([
            "",
            "## 被过滤掉的币及原因",
            "",
        ])
        
        filtered = [s for s in report.symbols if s.decision == DecisionLabel.BLOCK.value or s.total_score < MIN_TOTAL_SCORE]
        if filtered:
            for sym in filtered:
                lines.extend([
                    f"### {sym.symbol}",
                    f"- **决策**: {sym.decision}",
                    f"- **总分**: {sym.total_score:.3f}",
                    f"- **原因**: {', '.join(sym.decision_reasons) if sym.decision_reasons else '分数未达标'}",
                    "",
                ])
        else:
            lines.append("- 无被过滤币种")
        
        lines.extend([
            "",
            "## 整体风险建议",
            "",
            f"**{report.overall_risk_recommendation}**",
            "",
        ])
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return filepath


class HighVolPoolResearchAgent:
    """主控制器 - 研究整理与事件过滤 Agent (增强版)"""
    
    def __init__(self, window_days: int = ROLLING_WINDOW_DAYS):
        self.data_agent = ResearchDataAgent(window_days=window_days)
        self.event_agent = EventFilterAgent()
        self.scoring_agent = ScoringAgent()
        self.output_agent = ResearchOutputAgent()
        self.archive_agent = ArchiveAgent()

    def _latest_trade_research(self) -> dict:
        try:
            p = Path("research/news/daily")
            files = sorted(p.glob("*_trade_research.json"))
            if not files:
                return {}
            return json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _sync_event_signals(self, report: ResearchReport):
        """把研究层关键字段写回 event_signals.json，供执行层读取。"""
        path = Path("event_signals.json")
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        tr = self._latest_trade_research()

        data["market_bias"] = tr.get("market_bias", "NEUTRAL")
        data["risk_mode"] = tr.get("risk_mode", "NORMAL")
        data["macro"] = data.get("macro", {})
        if report.macro_state == MacroState.REDUCE_RISK.value:
            data["risk_mode"] = "REDUCE_RISK"
            data["macro"]["reduce_risk"] = True
        elif report.macro_state == MacroState.BLOCK.value:
            data["risk_mode"] = "BLOCK"
            data["macro"]["block"] = True
        data["geopolitics"] = tr.get("geopolitics", data.get("geopolitics", {
            "reduce_risk": False,
            "block_new_entries": False,
            "alts_bias": "NEUTRAL",
            "reason": "No major geopolitical escalation",
        }))
        data["sentiment"] = tr.get("sentiment", {
            "bias": "NEUTRAL",
            "strength": "LOW",
            "recommended_action": "WAIT",
            "reason": "not available",
        })

        data.setdefault("whale", {})
        data["whale"]["whale_score"] = tr.get("whale_score", data["whale"].get("whale_score", 0.0))
        data["whale"]["whale_bias"] = tr.get("whale_bias", data["whale"].get("whale_bias", "NEUTRAL"))
        data["whale"]["whale_reason"] = tr.get("whale_reason", data["whale"].get("whale_reason", "No validated smart-money signal"))

        sym_map = {s.symbol: s for s in report.symbols}
        data.setdefault("symbols", {})
        for sym in POOL_SYMBOLS_RAW:
            data["symbols"].setdefault(sym, {})
            sr = sym_map.get(sym)
            if sr:
                bias = "LONG" if sr.trend_4h == "UP" else "SHORT" if sr.trend_4h == "DOWN" else "NEUTRAL"
                strength = "HIGH" if sr.total_score >= 0.75 else "MEDIUM" if sr.total_score >= 0.55 else "LOW"
                action = "WAIT"
                if sr.decision == DecisionLabel.BLOCK.value:
                    action = "BLOCK_NEW_ENTRIES"
                elif sr.decision == DecisionLabel.REDUCE_RISK.value:
                    action = "REDUCE_RISK"
                elif bias in {"LONG", "SHORT"} and strength != "LOW":
                    action = bias

                data["symbols"][sym]["bias"] = bias
                data["symbols"][sym]["strength"] = strength
                data["symbols"][sym]["recommended_action"] = action

        data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def run_daily_research(self) -> ResearchReport:
        """执行每日研究流程"""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        
        print(f"[{now.isoformat()}] 开始高波动池每日研究...")
        
        # 1. 评估宏观环境
        macro_state, macro_notes = self.event_agent.evaluate_macro(now)
        print(f"  宏观状态: {macro_state.value}")
        
        # 2. 收集所有币种数据
        print("  收集币种数据...")
        data_list = []
        data_window = None
        for sym_raw in POOL_SYMBOLS_RAW:
            print(f"    - {sym_raw}")
            data = self.data_agent.collect_symbol_data(sym_raw)
            if data_window is None:
                data_window = data.get("data_window", "")
            data_list.append(data)
        
        # 3. 计算评分
        print("  计算评分...")
        score_df = self.scoring_agent.calculate_scores(data_list)
        
        # 4. 评估事件风险
        print("  评估事件风险...")
        event_results = {}
        for sym_raw in POOL_SYMBOLS_RAW:
            decision, notes, news_signals, whale_score, whale_bias, whale_reason = self.event_agent.evaluate_symbol(sym_raw, now)
            event_results[sym_raw] = {
                "decision": decision,
                "notes": notes,
                "news_signals": news_signals,
                "whale_score": whale_score,
                "whale_bias": whale_bias,
                "whale_reason": whale_reason,
            }
        
        # 5. 应用过滤规则
        score_df = self.scoring_agent.apply_event_filters(score_df, event_results)
        score_df = self.scoring_agent.apply_whale_overlay(score_df, event_results)
        score_df = self.scoring_agent.apply_special_rules(score_df)
        
        # 6. 排序
        score_df = score_df.sort_values("total_score", ascending=False).reset_index(drop=True)
        
        # 7. 选择候选
        picks = score_df[score_df["total_score"] >= MIN_TOTAL_SCORE].head(MAX_DAILY_PICKS)
        top_candidates = picks["symbol"].tolist() if len(picks) > 0 else []
        
        # 8. 构建 SymbolResearch 对象
        symbols = []
        for _, row in score_df.iterrows():
            sym_raw = row["symbol"]
            ev_result = event_results.get(sym_raw, {})
            decision = ev_result.get("decision", DecisionLabel.ALLOW)
            event_notes = ev_result.get("notes", [])
            news_signals = ev_result.get("news_signals", [])
            whale_score = float(ev_result.get("whale_score", 0.0))
            whale_bias = ev_result.get("whale_bias", "NEUTRAL")
            whale_reason = ev_result.get("whale_reason", "No validated smart-money signal")
            
            if row["total_score"] < MIN_TOTAL_SCORE and decision != DecisionLabel.BLOCK:
                decision = DecisionLabel.BLOCK
                event_notes.append(f"总分 {row['total_score']:.3f} 低于阈值 {MIN_TOTAL_SCORE}")
            
            decision_reasons = []
            if row["trend_score"] > 0.6:
                decision_reasons.append("4h趋势明确")
            if row["continuation_score"] > 0.6:
                decision_reasons.append("突破延续率较高")
            if row["liquidity_score"] > 0.6:
                decision_reasons.append("流动性充足")
            if row["volatility_score"] > 0.5:
                decision_reasons.append("波动率适中")
            if row["fake_breakout_penalty"] < 0.4:
                decision_reasons.append("假突破风险较低")
            
            if abs(whale_score) >= 0.4 and whale_bias != "NEUTRAL":
                decision_reasons.append(f"Smart-money {whale_bias} 辅助因子: {whale_reason}")

            if not decision_reasons and decision != DecisionLabel.BLOCK:
                decision_reasons.append("综合评分达标")
            
            spread = row.get("spread", 0.001)
            spread_score = max(0, min(10, 10 - (spread * 10000)))
            
            funding_state = self.data_agent._classify_funding(row.get("funding_rate", 0))
            oi_series = score_df["open_interest"]
            oi_state = self.data_agent._classify_oi(oi_series, row.get("open_interest", 0))
            
            sym_research = SymbolResearch(
                symbol=sym_raw,
                symbol_ccxt=row["symbol_ccxt"],
                trend_1h=row["trend_1h"],
                trend_4h=row["trend_4h"],
                atr=row["atr"],
                atr_pct=row["atr_pct"],
                quote_vol_24h=row["quote_vol_24h"],
                spread=row.get("spread", np.nan),
                spread_score=spread_score,
                open_interest=row.get("open_interest", np.nan),
                funding_rate=row.get("funding_rate", np.nan),
                funding_state=funding_state,
                oi_state=oi_state,
                adx_4h=row["adx_4h"],
                trend_strength_4h=row["trend_strength_4h"],
                continuation_6h=row["continuation_6h"],
                continuation_12h=row["continuation_12h"],
                continuation_score=row["continuation_score"],
                fake_breakout_rate_6h=row["fake_breakout_rate_6h"],
                fake_breakout_rate_12h=row["fake_breakout_rate_12h"],
                fake_breakout_risk=row["fake_breakout_penalty"],
                volatility_score=row["volatility_score"],
                liquidity_score=row["liquidity_score"],
                trend_score=row["trend_score"],
                funding_penalty=row["funding_penalty"],
                oi_crowding_penalty=row["oi_crowding_penalty"],
                total_score=row["total_score"],
                decision=decision.value,
                decision_reasons=decision_reasons,
                risk_bucket=RISK_BUCKET[sym_raw],
                has_major_event=decision != DecisionLabel.ALLOW or len(news_signals) > 0,
                event_notes=event_notes,
                news_signals=news_signals,
                whale_score=whale_score,
                whale_bias=whale_bias,
                whale_reason=whale_reason,
                whale_adjustment=row.get("whale_adjustment", 0.0),
                funding_change_24h=row.get("funding_change_24h", 0),
                oi_change_24h=row.get("oi_change_24h", 0),
            )
            symbols.append(sym_research)
        
        # 9. 确定整体风险建议
        if macro_state == MacroState.BLOCK:
            overall_risk = "建议高波动池当天整体空仓"
        elif macro_state == MacroState.REDUCE_RISK:
            overall_risk = "建议降低整体风险敞口"
        elif len(top_candidates) == 0:
            overall_risk = "当日无合格候选币，建议空仓观望"
        else:
            overall_risk = "正常参与"
        
        # 10. 构建报告
        report = ResearchReport(
            date=date_str,
            generated_at=now.isoformat(),
            macro_state=macro_state.value,
            pool_status="ACTIVE" if len(top_candidates) > 0 else "FLAT",
            top_candidates=top_candidates,
            overall_risk_recommendation=overall_risk,
            macro_notes=macro_notes,
            symbols=symbols,
            data_window=data_window or "",
        )
        
        # 11. 生成输出文件
        print("  生成报告文件...")
        json_path = self.output_agent.generate_json_report(report, date_str)
        md_path = self.output_agent.generate_markdown_summary(report, date_str)
        
        # 12. 归档到 CSV
        print("  归档历史数据...")
        archive_path = self.archive_agent.append_daily_record(report)
        
        print(f"  完成!")
        print(f"  - JSON: {json_path}")
        print(f"  - Markdown: {md_path}")
        print(f"  - Archive: {archive_path}")

        # 13. 同步关键研究字段到 event_signals.json（交易执行读取）
        self._sync_event_signals(report)
        print("  - Synced: event_signals.json")
        
        return report


def main():
    """主入口"""
    agent = HighVolPoolResearchAgent()
    report = agent.run_daily_research()
    
    print("\n=== 研究摘要 ===")
    print(f"日期: {report.date}")
    print(f"数据窗口: {report.data_window}")
    print(f"宏观状态: {report.macro_state}")
    print(f"候选币: {report.top_candidates if report.top_candidates else '无'}")
    print(f"整体建议: {report.overall_risk_recommendation}")


if __name__ == "__main__":
    main()
