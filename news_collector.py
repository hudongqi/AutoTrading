"""
News & Macro Data Collector
新闻与宏观数据收集器

职责：
- 自动抓取新闻和宏观数据
- 初步分类和结构化
- 生成人工审核用的 daily_news_digest.json
- 生成可服务交易决策的结构化 research 输出

输出：
- research/news/daily/YYYY-MM-DD_digest.json (供人工审核)
- research/news/daily/YYYY-MM-DD_trade_research.json (供程序读取/交易研究)
- research/news/daily/YYYY-MM-DD_summary.md (供复盘)
- 人工确认后写入 event_signals.json
"""

import os
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import urllib.request
import urllib.parse

import feedparser


class NewsCategory(Enum):
    TECH_UPGRADE = "技术升级"
    REGULATION = "监管"
    MARKET = "市场"
    SECURITY = "安全"
    MACRO = "宏观"
    SOCIAL = "社媒"
    OTHER = "其他"


class UrgencyLevel(Enum):
    IMMEDIATE = "immediate"
    TODAY = "today"
    THIS_WEEK = "week"
    BACKGROUND = "bg"


class ImpactScope(Enum):
    SINGLE = "single"
    SECTOR = "sector"
    MARKET_WIDE = "market"


@dataclass
class RawSignal:
    id: str
    timestamp: str
    title: str
    content: str
    source: str
    source_url: str
    category: str
    sentiment: float
    urgency: str
    impact_scope: str
    confidence: float
    related_symbols: List[str]
    keywords_matched: List[str]
    raw_data: dict


@dataclass
class DailyDigest:
    date: str
    generated_at: str
    macro_signals: List[RawSignal]
    symbol_signals: Dict[str, List[RawSignal]]
    unclassified: List[RawSignal]
    stats: dict


SYMBOL_KEYWORDS = {
    "SOLUSDT": [("solana", True), ("$sol", False), ("#sol", False)],
    "XRPUSDT": [("ripple", True), ("$xrp", False), ("#xrp", False)],
    "DOGEUSDT": [("dogecoin", True), ("$doge", False), ("#doge", False)],
    "SUIUSDT": [("$sui", False), ("#sui", False), ("sui network", False), ("mysten labs", False)],
    "PEPEUSDT": [("$pepe", False), ("#pepe", False), ("pepecoin", True)],
}

POOL_SYMBOLS_RAW = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]


class CryptoPanicCollector:
    API_BASE = "https://cryptopanic.com/api/free/v1/posts/"

    def fetch(self, hours_back: int = 24) -> List[RawSignal]:
        signals = []
        url = self.API_BASE + "?" + urllib.parse.urlencode({"public": "true", "kind": "news"})

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                for post in data.get("results", []):
                    signal = self._parse_post(post)
                    if signal:
                        signals.append(signal)
        except Exception as e:
            print(f"CryptoPanic fetch error: {e}")

        return signals

    def _match_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        related_symbols, keywords_matched = [], []
        text_lower = text.lower()
        for symbol, keywords in SYMBOL_KEYWORDS.items():
            for kw, word_boundary in keywords:
                kw_lower = kw.lower()
                if word_boundary:
                    if re.search(r"\\b" + re.escape(kw_lower) + r"\\b", text_lower):
                        related_symbols.append(symbol)
                        keywords_matched.append(kw)
                        break
                else:
                    if kw_lower in text_lower:
                        related_symbols.append(symbol)
                        keywords_matched.append(kw)
                        break
        return related_symbols, keywords_matched

    def _parse_post(self, post: dict) -> Optional[RawSignal]:
        title = post.get("title", "")
        content = post.get("title", "")
        related_symbols, keywords_matched = self._match_keywords(title + " " + content)
        if not related_symbols:
            return None

        category = self._classify(title, content)
        sentiment = self._analyze_sentiment(title, content)
        urgency = self._assess_urgency(title, category)
        impact_scope = ImpactScope.SINGLE.value if len(related_symbols) == 1 else ImpactScope.SECTOR.value

        return RawSignal(
            id=f"cp_{post.get('id', '')}",
            timestamp=post.get("published_at", datetime.now(timezone.utc).isoformat()),
            title=title,
            content=content,
            source="CryptoPanic",
            source_url=post.get("url", ""),
            category=category.value,
            sentiment=sentiment,
            urgency=urgency.value,
            impact_scope=impact_scope,
            confidence=0.7,
            related_symbols=list(set(related_symbols)),
            keywords_matched=list(set(keywords_matched)),
            raw_data=post,
        )

    def _classify(self, title: str, content: str) -> NewsCategory:
        text = (title + " " + content).lower()
        if any(k in text for k in ["hack", "exploit", "vulnerability", "bug", "down", "outage", "attack"]):
            return NewsCategory.SECURITY
        if any(k in text for k in ["sec", "regulation", "lawsuit", "court", "legal", "compliance", "etf"]):
            return NewsCategory.REGULATION
        if any(k in text for k in ["upgrade", "update", "mainnet", "testnet", "release", "launch"]):
            return NewsCategory.TECH_UPGRADE
        if any(k in text for k in ["listing", "partnership", "adoption", "integration", "collaboration"]):
            return NewsCategory.MARKET
        return NewsCategory.OTHER

    def _analyze_sentiment(self, title: str, content: str) -> float:
        text = (title + " " + content).lower()
        positive = ["surge", "rally", "bull", "gain", "moon", "pump", "breakthrough", "win", "victory"]
        negative = ["crash", "dump", "bear", "loss", "hack", "scam", "lawsuit", "ban", "delay", "cancel"]
        pos_count = sum(1 for p in positive if p in text)
        neg_count = sum(1 for n in negative if n in text)
        if pos_count > neg_count:
            return min(0.8, 0.3 + pos_count * 0.2)
        if neg_count > pos_count:
            return max(-0.8, -0.3 - neg_count * 0.2)
        return 0.0

    def _assess_urgency(self, title: str, category: NewsCategory) -> UrgencyLevel:
        if category == NewsCategory.SECURITY:
            return UrgencyLevel.IMMEDIATE
        if any(k in title.lower() for k in ["breaking", "urgent", "alert", "now", "today"]):
            return UrgencyLevel.TODAY
        if category in [NewsCategory.REGULATION, NewsCategory.MACRO]:
            return UrgencyLevel.THIS_WEEK
        return UrgencyLevel.BACKGROUND


class CointelegraphCollector:
    RSS_URL = "https://cointelegraph.com/rss"

    def fetch(self, hours_back: int = 24) -> List[RawSignal]:
        signals = []
        try:
            feed = feedparser.parse(self.RSS_URL)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            for entry in feed.entries:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                    if pub_dt < cutoff:
                        continue
                signal = self._parse_entry(entry)
                if signal:
                    signals.append(signal)
        except Exception as e:
            print(f"Cointelegraph fetch error: {e}")
        return signals

    def _parse_entry(self, entry: dict) -> Optional[RawSignal]:
        title = entry.get("title", "")
        content = entry.get("summary", "")[:500]
        cp = CryptoPanicCollector()
        related_symbols, keywords_matched = cp._match_keywords(title + " " + content)
        if not related_symbols:
            return None

        category = cp._classify(title, content)
        sentiment = cp._analyze_sentiment(title, content)
        urgency = cp._assess_urgency(title, category)
        impact_scope = ImpactScope.SINGLE.value if len(related_symbols) == 1 else ImpactScope.SECTOR.value

        return RawSignal(
            id=f"ct_{hash(title) % 10000000}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=title,
            content=content,
            source="Cointelegraph",
            source_url=entry.get("link", ""),
            category=category.value,
            sentiment=sentiment,
            urgency=urgency.value,
            impact_scope=impact_scope,
            confidence=0.6,
            related_symbols=list(set(related_symbols)),
            keywords_matched=list(set(keywords_matched)),
            raw_data={"author": entry.get("author", "")},
        )


class BinanceAnnouncementCollector:
    API_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query"

    def fetch(self, hours_back: int = 24) -> List[RawSignal]:
        signals = []
        for catalog_id in [1, 48, 49]:
            try:
                url = self.API_URL + "?" + urllib.parse.urlencode({"catalogId": catalog_id, "pageNo": 1, "pageSize": 20})
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode())
                    for article in data.get("data", {}).get("articles", []):
                        signal = self._parse_article(article, catalog_id)
                        if signal:
                            signals.append(signal)
            except Exception as e:
                print(f"Binance catalog {catalog_id} error: {e}")
        return signals

    def _parse_article(self, article: dict, catalog_id: int) -> Optional[RawSignal]:
        title = article.get("title", "")
        content = article.get("summary", "") or title
        cp = CryptoPanicCollector()
        related_symbols, keywords_matched = cp._match_keywords(title + " " + content)
        if not related_symbols:
            return None

        if catalog_id == 48:
            category = NewsCategory.MARKET
        elif "upgrade" in title.lower() or "maintenance" in title.lower():
            category = NewsCategory.TECH_UPGRADE
        else:
            category = NewsCategory.OTHER

        return RawSignal(
            id=f"bn_{article.get('id', '')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=title,
            content=content,
            source="Binance",
            source_url=f"https://www.binance.com/en/support/announcement/{article.get('code', '')}",
            category=category.value,
            sentiment=0.2 if catalog_id == 48 else 0.0,
            urgency=UrgencyLevel.TODAY.value if catalog_id == 48 else UrgencyLevel.BACKGROUND.value,
            impact_scope=ImpactScope.MARKET_WIDE.value if catalog_id != 48 else ImpactScope.SINGLE.value,
            confidence=0.9,
            related_symbols=list(set(related_symbols)),
            keywords_matched=list(set(keywords_matched)),
            raw_data=article,
        )


class MacroDataCollector:
    """宏观数据收集器（含交易解释字段模板）"""

    @staticmethod
    def _macro_template(event_name: str, event_date: str, source_url: str, previous: str, forecast: str,
                        market_bias: str, if_above: str, if_below: str, pre_action: str, post_action: str):
        return {
            "event_name": event_name,
            "event_date": event_date,
            "previous": previous,
            "forecast": forecast,
            "market_bias": market_bias,
            "if_above_forecast": if_above,
            "if_below_forecast": if_below,
            "pre_event_action": pre_action,
            "post_event_action": post_action,
            "source_url": source_url,
        }

    def fetch(self, days_ahead: int = 7) -> List[RawSignal]:
        signals = []
        now = datetime.now(timezone.utc)

        friday = now + timedelta(days=(4 - now.weekday()) % 7)
        if friday.day <= 7:
            raw = self._macro_template(
                event_name="NFP",
                event_date=friday.strftime("%Y-%m-%d"),
                source_url="https://www.bls.gov/schedule/news_release/",
                previous="TBD",
                forecast="TBD",
                market_bias="数据偏强通常利空风险资产，偏弱通常利多风险资产（需结合失业率与薪资）",
                if_above="若显著高于预期 -> 倾向 SHORT / REDUCE_RISK",
                if_below="若显著低于预期 -> 倾向 LONG（择强）",
                pre_action="事件前1-3小时降低杠杆与仓位，避免追单",
                post_action="等待首波波动收敛后，按方向与成交量确认再跟随",
            )
            signals.append(RawSignal(
                id=f"macro_nfp_{friday.strftime('%Y%m%d')}",
                timestamp=now.isoformat(),
                title=f"NFP 非农就业数据 - {friday.strftime('%Y-%m-%d')}",
                content="美国非农就业数据发布，影响市场风险偏好",
                source="BLS",
                source_url=raw["source_url"],
                category=NewsCategory.MACRO.value,
                sentiment=0.0,
                urgency=UrgencyLevel.THIS_WEEK.value,
                impact_scope=ImpactScope.MARKET_WIDE.value,
                confidence=0.95,
                related_symbols=[],
                keywords_matched=["NFP", "nonfarm"],
                raw_data=raw,
            ))

        if 10 <= now.day <= 15:
            cpi_date = now + timedelta(days=(13 - now.day) % 7)
            raw = self._macro_template(
                event_name="CPI",
                event_date=cpi_date.strftime("%Y-%m-%d"),
                source_url="https://www.bls.gov/cpi/",
                previous="TBD",
                forecast="TBD",
                market_bias="通胀高于预期通常偏空风险资产；低于预期通常偏多",
                if_above="高于预期 -> 倾向 SHORT / REDUCE_RISK",
                if_below="低于预期 -> 倾向 LONG（优先高质量趋势币）",
                pre_action="事件前减仓、收紧回撤阈值，避免重仓隔数据",
                post_action="公布后等待方向确认（5-30分钟），再按突破/回踩执行",
            )
            signals.append(RawSignal(
                id=f"macro_cpi_{cpi_date.strftime('%Y%m%d')}",
                timestamp=now.isoformat(),
                title=f"CPI 消费者物价指数 - {cpi_date.strftime('%Y-%m-%d')}",
                content="美国CPI数据发布，影响美联储利率预期",
                source="BLS",
                source_url=raw["source_url"],
                category=NewsCategory.MACRO.value,
                sentiment=0.0,
                urgency=UrgencyLevel.THIS_WEEK.value,
                impact_scope=ImpactScope.MARKET_WIDE.value,
                confidence=0.95,
                related_symbols=[],
                keywords_matched=["CPI", "inflation"],
                raw_data=raw,
            ))

        return signals


class NewsDigestAgent:
    def __init__(self, base_dir: str = "research/news"):
        self.base_dir = base_dir
        self.collectors = [CryptoPanicCollector(), CointelegraphCollector(), BinanceAnnouncementCollector()]
        self.macro_collector = MacroDataCollector()
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(f"{self.base_dir}/daily", exist_ok=True)
        os.makedirs(f"{self.base_dir}/archive", exist_ok=True)

    def collect_daily(self) -> DailyDigest:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        print(f"[{now.isoformat()}] 开始收集新闻...")

        all_signals = []
        for collector in self.collectors:
            print(f"  - {collector.__class__.__name__}...")
            signals = collector.fetch(hours_back=24)
            all_signals.extend(signals)
            print(f"    获取 {len(signals)} 条")

        print("  - MacroDataCollector...")
        macro_signals = self.macro_collector.fetch(days_ahead=7)
        all_signals.extend(macro_signals)
        print(f"    获取 {len(macro_signals)} 条")

        symbol_signals = {sym: [] for sym in POOL_SYMBOLS_RAW}
        unclassified = []
        for signal in all_signals:
            if signal.related_symbols:
                for sym in signal.related_symbols:
                    if sym in symbol_signals:
                        symbol_signals[sym].append(signal)
            elif signal.category != NewsCategory.MACRO.value:
                unclassified.append(signal)

        stats = {"total_signals": len(all_signals), "by_source": {}, "by_category": {}, "by_urgency": {}}
        for signal in all_signals:
            stats["by_source"][signal.source] = stats["by_source"].get(signal.source, 0) + 1
            stats["by_category"][signal.category] = stats["by_category"].get(signal.category, 0) + 1
            stats["by_urgency"][signal.urgency] = stats["by_urgency"].get(signal.urgency, 0) + 1

        return DailyDigest(
            date=date_str,
            generated_at=now.isoformat(),
            macro_signals=macro_signals,
            symbol_signals=symbol_signals,
            unclassified=unclassified,
            stats=stats,
        )

    @staticmethod
    def _symbol_bias(sentiment_avg: float) -> str:
        if sentiment_avg >= 0.2:
            return "LONG"
        if sentiment_avg <= -0.2:
            return "SHORT"
        return "NEUTRAL"

    @staticmethod
    def _strength(score: float) -> str:
        if score >= 0.7:
            return "HIGH"
        if score >= 0.4:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _recommended_action(bias: str, strength: str) -> str:
        if bias == "LONG" and strength in {"HIGH", "MEDIUM"}:
            return "LONG"
        if bias == "SHORT" and strength in {"HIGH", "MEDIUM"}:
            return "SHORT"
        if strength == "LOW":
            return "WAIT"
        return "REDUCE_RISK"

    def build_trade_research(self, digest: DailyDigest) -> dict:
        macro_items = []
        for sig in digest.macro_signals:
            rd = sig.raw_data or {}
            macro_items.append({
                "event": sig.title,
                "date": rd.get("event_date", sig.timestamp[:10]),
                "source": sig.source,
                "source_url": sig.source_url,
                "previous": rd.get("previous", "TBD"),
                "forecast": rd.get("forecast", "TBD"),
                "market_bias": rd.get("market_bias", "TBD"),
                "if_above_forecast": rd.get("if_above_forecast", "REDUCE_RISK"),
                "if_below_forecast": rd.get("if_below_forecast", "WAIT"),
                "pre_event_action": rd.get("pre_event_action", "REDUCE_RISK"),
                "post_event_action": rd.get("post_event_action", "WAIT_CONFIRMATION"),
                "confidence": sig.confidence,
            })

        symbol_research = {}
        for symbol, signals in digest.symbol_signals.items():
            if not signals:
                symbol_research[symbol] = {
                    "bias": "NEUTRAL",
                    "strength": "LOW",
                    "reason": "No strong symbol-specific news signal",
                    "recommended_action": "WAIT",
                    "signal_count": 0,
                }
                continue

            weighted_sent = sum(s.sentiment * s.confidence for s in signals)
            conf_sum = sum(s.confidence for s in signals)
            avg_sent = weighted_sent / conf_sum if conf_sum > 0 else 0.0
            strength_score = min(1.0, conf_sum / max(1, len(signals)))

            bias = self._symbol_bias(avg_sent)
            strength = self._strength(strength_score)
            action = self._recommended_action(bias, strength)

            headlines = [s.title for s in signals[:3]]
            reason = f"avg_sentiment={avg_sent:+.2f}, confidence={strength_score:.2f}, headlines={headlines}"

            # 安全类信号直接提高防守优先级
            if any(s.category == NewsCategory.SECURITY.value for s in signals):
                action = "REDUCE_RISK"
                if bias == "LONG":
                    bias = "NEUTRAL"
                reason += " | security risk detected"

            symbol_research[symbol] = {
                "bias": bias,
                "strength": strength,
                "reason": reason,
                "recommended_action": action,
                "signal_count": len(signals),
            }

        action_rank = {"LONG": 3, "SHORT": 3, "REDUCE_RISK": 2, "WAIT": 1}
        overall_action = "WAIT"
        if symbol_research:
            overall_action = max((v["recommended_action"] for v in symbol_research.values()), key=lambda x: action_rank.get(x, 0))

        return {
            "date": digest.date,
            "generated_at": digest.generated_at,
            "macro": macro_items,
            "symbols": symbol_research,
            "overall_recommendation": overall_action,
            "note": "Decision-support only. Final trading decision requires manual confirmation and risk checks.",
        }

    def generate_digest_json(self, digest: DailyDigest) -> str:
        filepath = f"{self.base_dir}/daily/{digest.date}_digest.json"
        data = {
            "date": digest.date,
            "generated_at": digest.generated_at,
            "note": "此文件供人工审核，确认后写入 event_signals.json",
            "macro_signals": [asdict(s) for s in digest.macro_signals],
            "symbol_signals": {sym: [asdict(s) for s in signals] for sym, signals in digest.symbol_signals.items()},
            "unclassified": [asdict(s) for s in digest.unclassified],
            "stats": digest.stats,
            "action_required": {
                "step1": "人工审核每条信号",
                "step2": "确认后编辑 event_signals.json",
                "step3": "运行研究 Agent 生成最终决策",
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def generate_trade_research_json(self, digest: DailyDigest) -> str:
        filepath = f"{self.base_dir}/daily/{digest.date}_trade_research.json"
        data = self.build_trade_research(digest)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def generate_summary_md(self, digest: DailyDigest) -> str:
        filepath = f"{self.base_dir}/daily/{digest.date}_summary.md"
        trade = self.build_trade_research(digest)

        lines = [
            f"# 每日新闻摘要 - {digest.date}",
            "",
            f"生成时间: {digest.generated_at}",
            f"整体建议: **{trade['overall_recommendation']}**",
            "",
            "## 统计概览",
            "",
            f"- 总信号数: {digest.stats['total_signals']}",
            "",
            "### 按来源",
        ]

        for source, count in sorted(digest.stats['by_source'].items(), key=lambda x: -x[1]):
            lines.append(f"- {source}: {count}")

        lines.extend(["", "## 宏观交易解释", ""])
        if trade["macro"]:
            for m in trade["macro"]:
                lines.extend([
                    f"### {m['event']}",
                    f"- previous: {m['previous']}",
                    f"- forecast: {m['forecast']}",
                    f"- market_bias: {m['market_bias']}",
                    f"- if_above_forecast: {m['if_above_forecast']}",
                    f"- if_below_forecast: {m['if_below_forecast']}",
                    f"- pre_event_action: {m['pre_event_action']}",
                    f"- post_event_action: {m['post_event_action']}",
                    "",
                ])
        else:
            lines.append("- 暂无宏观事件")

        lines.extend(["", "## 币种方向建议", ""])
        for symbol in POOL_SYMBOLS_RAW:
            s = trade["symbols"][symbol]
            lines.extend([
                f"### {symbol}",
                f"- bias: {s['bias']}",
                f"- strength: {s['strength']}",
                f"- recommended_action: {s['recommended_action']}",
                f"- reason: {s['reason']}",
                "",
            ])

        lines.extend([
            "## 待办事项",
            "",
            "1. [ ] 审核 *_digest.json（原始信号）",
            "2. [ ] 审核 *_trade_research.json（方向建议）",
            "3. [ ] 确认后更新 event_signals.json",
            "",
            "---",
            "",
            "**注意**: 此输出是决策辅助，不是自动交易指令。",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def run(self):
        print("=" * 50)
        print("新闻收集与摘要生成")
        print("=" * 50)
        digest = self.collect_daily()

        print("\n生成输出文件...")
        json_path = self.generate_digest_json(digest)
        trade_path = self.generate_trade_research_json(digest)
        md_path = self.generate_summary_md(digest)

        print("\n完成!")
        print(f"  - JSON (供审核): {json_path}")
        print(f"  - JSON (交易研究): {trade_path}")
        print(f"  - Markdown (供复盘): {md_path}")
        print("\n下一步: 人工审核后编辑 event_signals.json")


def main():
    agent = NewsDigestAgent()
    agent.run()


if __name__ == "__main__":
    main()
