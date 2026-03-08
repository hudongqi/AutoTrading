"""
News & Macro Data Collector
新闻与宏观数据收集器

职责：
- 自动抓取新闻和宏观数据
- 初步分类和结构化
- 生成人工审核用的 daily_news_digest.json
- 不直接生成交易指令

输出：
- research/news/daily/YYYY-MM-DD_digest.json (供人工审核)
- research/news/daily/YYYY-MM-DD_summary.md (供复盘)
- 人工确认后写入 event_signals.json
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import urllib.request
import urllib.error
import urllib.parse

import feedparser


class NewsCategory(Enum):
    """新闻分类"""
    TECH_UPGRADE = "技术升级"      # 网络升级、新功能
    REGULATION = "监管"            # SEC、法律、合规
    MARKET = "市场"                # 上市、合作、采用
    SECURITY = "安全"              # 漏洞、攻击、停机
    MACRO = "宏观"                 # 经济数据、利率
    SOCIAL = "社媒"                # KOL、社区、情绪
    OTHER = "其他"


class UrgencyLevel(Enum):
    """紧急程度"""
    IMMEDIATE = "immediate"    # 立即处理（安全事件）
    TODAY = "today"            # 当日关注
    THIS_WEEK = "week"         # 本周关注
    BACKGROUND = "bg"          # 背景信息


class ImpactScope(Enum):
    """影响范围"""
    SINGLE = "single"          # 单币种
    SECTOR = "sector"          # 板块（如Layer1、Meme）
    MARKET_WIDE = "market"     # 全市场


@dataclass
class RawSignal:
    """原始信号（未人工确认）"""
    id: str
    timestamp: str
    title: str
    content: str
    source: str                    # 来源：CryptoPanic/Cointelegraph/Binance/FRED/BLS
    source_url: str
    category: str
    sentiment: float               # -1.0 ~ +1.0
    urgency: str
    impact_scope: str
    confidence: float              # 0.0 ~ 1.0
    related_symbols: List[str]     # ["SOLUSDT", "XRPUSDT"]
    keywords_matched: List[str]    # 匹配到的关键词
    raw_data: dict                 # 原始抓取数据


@dataclass
class DailyDigest:
    """每日摘要"""
    date: str
    generated_at: str
    macro_signals: List[RawSignal]
    symbol_signals: Dict[str, List[RawSignal]]  # 按币种分组
    unclassified: List[RawSignal]
    stats: dict


# 币种关键词映射（使用更严格的匹配模式）
# 格式: (关键词, 是否要求词边界)
SYMBOL_KEYWORDS = {
    "SOLUSDT": [
        ("solana", True),
        ("$sol", False),
        ("#sol", False),
    ],
    "XRPUSDT": [
        ("ripple", True),
        ("$xrp", False),
        ("#xrp", False),
    ],
    "DOGEUSDT": [
        ("dogecoin", True),
        ("$doge", False),
        ("#doge", False),
    ],
    "SUIUSDT": [
        ("$sui", False),
        ("#sui", False),
        ("sui network", False),
        ("mysten labs", False),
    ],
    "PEPEUSDT": [
        ("$pepe", False),
        ("#pepe", False),
        ("pepecoin", True),
    ],
}

POOL_SYMBOLS_RAW = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]


class CryptoPanicCollector:
    """CryptoPanic 新闻收集器"""
    
    API_BASE = "https://cryptopanic.com/api/free/v1/posts/"
    
    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
    
    def fetch(self, hours_back: int = 24) -> List[RawSignal]:
        """获取最近新闻"""
        signals = []
        
        params = {
            "public": "true",
            "kind": "news",
        }
        
        url = self.API_BASE + "?" + urllib.parse.urlencode(params)
        
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            
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
        """匹配关键词，返回 (相关币种, 匹配到的关键词)"""
        related_symbols = []
        keywords_matched = []
        text_lower = text.lower()
        
        for symbol, keywords in SYMBOL_KEYWORDS.items():
            for kw, word_boundary in keywords:
                kw_lower = kw.lower()
                if word_boundary:
                    pattern = r'\b' + re.escape(kw_lower) + r'\b'
                    if re.search(pattern, text_lower):
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
        """解析单条新闻"""
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
            raw_data=post
        )
    
    def _classify(self, title: str, content: str) -> NewsCategory:
        """分类新闻"""
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
        """简单情感分析"""
        text = (title + " " + content).lower()
        
        positive = ["surge", "rally", "bull", "gain", "moon", "pump", "breakthrough", "win", "victory"]
        negative = ["crash", "dump", "bear", "loss", "hack", "scam", "lawsuit", "ban", "delay", "cancel"]
        
        pos_count = sum(1 for p in positive if p in text)
        neg_count = sum(1 for n in negative if n in text)
        
        if pos_count > neg_count:
            return min(0.8, 0.3 + pos_count * 0.2)
        elif neg_count > pos_count:
            return max(-0.8, -0.3 - neg_count * 0.2)
        return 0.0
    
    def _assess_urgency(self, title: str, category: NewsCategory) -> UrgencyLevel:
        """评估紧急程度"""
        if category == NewsCategory.SECURITY:
            return UrgencyLevel.IMMEDIATE
        
        text = title.lower()
        if any(k in text for k in ["breaking", "urgent", "alert", "now", "today"]):
            return UrgencyLevel.TODAY
        
        if category in [NewsCategory.REGULATION, NewsCategory.MACRO]:
            return UrgencyLevel.THIS_WEEK
        
        return UrgencyLevel.BACKGROUND


class CointelegraphCollector:
    """Cointelegraph RSS 收集器"""
    
    RSS_URL = "https://cointelegraph.com/rss"
    
    def fetch(self, hours_back: int = 24) -> List[RawSignal]:
        """获取 RSS 新闻"""
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
        """解析 RSS 条目"""
        title = entry.get("title", "")
        content = entry.get("summary", "")[:500]
        
        # 使用 CryptoPanicCollector 的匹配方法
        cp = CryptoPanicCollector()
        related_symbols, keywords_matched = cp._match_keywords(title + " " + content)
        
        if not related_symbols:
            return None
        
        cp = CryptoPanicCollector()
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
            raw_data={"author": entry.get("author", "")}
        )


class BinanceAnnouncementCollector:
    """Binance 公告收集器"""
    
    API_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query"
    
    def fetch(self, hours_back: int = 24) -> List[RawSignal]:
        """获取 Binance 公告"""
        signals = []
        catalogs = [1, 48, 49]
        
        for catalog_id in catalogs:
            try:
                params = {
                    "catalogId": catalog_id,
                    "pageNo": 1,
                    "pageSize": 20
                }
                
                url = self.API_URL + "?" + urllib.parse.urlencode(params)
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
        """解析公告"""
        title = article.get("title", "")
        content = article.get("summary", "") or title
        
        # 使用 CryptoPanicCollector 的匹配方法
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
            raw_data=article
        )


class MacroDataCollector:
    """宏观数据收集器"""
    
    def fetch(self, days_ahead: int = 7) -> List[RawSignal]:
        """获取即将发布的宏观数据"""
        signals = []
        now = datetime.now(timezone.utc)
        
        # 检测本周五是否是 NFP
        friday = now + timedelta(days=(4 - now.weekday()) % 7)
        if friday.day <= 7:
            signals.append(RawSignal(
                id=f"macro_nfp_{friday.strftime('%Y%m%d')}",
                timestamp=now.isoformat(),
                title=f"NFP 非农就业数据 - {friday.strftime('%Y-%m-%d')}",
                content="美国非农就业数据发布，影响市场风险偏好",
                source="BLS",
                source_url="https://www.bls.gov/schedule/news_release/",
                category=NewsCategory.MACRO.value,
                sentiment=0.0,
                urgency=UrgencyLevel.THIS_WEEK.value,
                impact_scope=ImpactScope.MARKET_WIDE.value,
                confidence=0.95,
                related_symbols=[],
                keywords_matched=["NFP", "nonfarm"],
                raw_data={"event_date": friday.isoformat()}
            ))
        
        # 检测 CPI（简化：每月 10-15 号）
        if 10 <= now.day <= 15:
            cpi_date = now + timedelta(days=(13 - now.day) % 7)
            signals.append(RawSignal(
                id=f"macro_cpi_{cpi_date.strftime('%Y%m%d')}",
                timestamp=now.isoformat(),
                title=f"CPI 消费者物价指数 - {cpi_date.strftime('%Y-%m-%d')}",
                content="美国CPI数据发布，影响美联储利率预期",
                source="BLS",
                source_url="https://www.bls.gov/cpi/",
                category=NewsCategory.MACRO.value,
                sentiment=0.0,
                urgency=UrgencyLevel.THIS_WEEK.value,
                impact_scope=ImpactScope.MARKET_WIDE.value,
                confidence=0.95,
                related_symbols=[],
                keywords_matched=["CPI", "inflation"],
                raw_data={"event_date": cpi_date.isoformat()}
            ))
        
        return signals


class NewsDigestAgent:
    """新闻摘要 Agent"""
    
    def __init__(self, base_dir: str = "research/news"):
        self.base_dir = base_dir
        self.collectors = [
            CryptoPanicCollector(),
            CointelegraphCollector(),
            BinanceAnnouncementCollector(),
        ]
        self.macro_collector = MacroDataCollector()
        self._ensure_directories()
    
    def _ensure_directories(self):
        os.makedirs(f"{self.base_dir}/daily", exist_ok=True)
        os.makedirs(f"{self.base_dir}/archive", exist_ok=True)
    
    def collect_daily(self) -> DailyDigest:
        """收集每日新闻"""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        
        print(f"[{now.isoformat()}] 开始收集新闻...")
        
        all_signals = []
        
        for collector in self.collectors:
            print(f"  - {collector.__class__.__name__}...")
            signals = collector.fetch(hours_back=24)
            all_signals.extend(signals)
            print(f"    获取 {len(signals)} 条")
        
        print(f"  - MacroDataCollector...")
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
            elif signal.category == NewsCategory.MACRO.value:
                pass
            else:
                unclassified.append(signal)
        
        stats = {
            "total_signals": len(all_signals),
            "by_source": {},
            "by_category": {},
            "by_urgency": {},
        }
        
        for signal in all_signals:
            stats["by_source"][signal.source] = stats["by_source"].get(signal.source, 0) + 1
            stats["by_category"][signal.category] = stats["by_category"].get(signal.category, 0) + 1
            stats["by_urgency"][signal.urgency] = stats["by_urgency"].get(signal.urgency, 0) + 1
        
        digest = DailyDigest(
            date=date_str,
            generated_at=now.isoformat(),
            macro_signals=macro_signals,
            symbol_signals=symbol_signals,
            unclassified=unclassified,
            stats=stats
        )
        
        return digest
    
    def generate_digest_json(self, digest: DailyDigest) -> str:
        """生成 JSON 摘要（供人工审核）"""
        filepath = f"{self.base_dir}/daily/{digest.date}_digest.json"
        
        data = {
            "date": digest.date,
            "generated_at": digest.generated_at,
            "note": "此文件供人工审核，确认后写入 event_signals.json",
            "macro_signals": [asdict(s) for s in digest.macro_signals],
            "symbol_signals": {
                sym: [asdict(s) for s in signals]
                for sym, signals in digest.symbol_signals.items()
            },
            "unclassified": [asdict(s) for s in digest.unclassified],
            "stats": digest.stats,
            "action_required": {
                "step1": "人工审核每条信号",
                "step2": "确认后编辑 event_signals.json",
                "step3": "运行研究 Agent 生成最终决策"
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def generate_summary_md(self, digest: DailyDigest) -> str:
        """生成 Markdown 摘要（供复盘）"""
        filepath = f"{self.base_dir}/daily/{digest.date}_summary.md"
        
        lines = [
            f"# 每日新闻摘要 - {digest.date}",
            "",
            f"生成时间: {digest.generated_at}",
            "",
            "## 统计概览",
            "",
            f"- 总信号数: {digest.stats['total_signals']}",
            "",
            "### 按来源",
        ]
        
        for source, count in sorted(digest.stats['by_source'].items(), key=lambda x: -x[1]):
            lines.append(f"- {source}: {count}")
        
        lines.extend(["", "### 按分类"])
        for cat, count in sorted(digest.stats['by_category'].items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {count}")
        
        lines.extend(["", "### 按紧急程度"])
        for urg, count in sorted(digest.stats['by_urgency'].items(), key=lambda x: -x[1]):
            lines.append(f"- {urg}: {count}")
        
        lines.extend(["", "## 宏观信号", ""])
        
        if digest.macro_signals:
            for sig in digest.macro_signals:
                lines.extend([
                    f"### {sig.title}",
                    f"- 来源: {sig.source}",
                    f"- 紧急度: {sig.urgency}",
                    f"- 置信度: {sig.confidence}",
                    f"- [链接]({sig.source_url})",
                    "",
                ])
        else:
            lines.append("- 暂无即将发生的宏观事件")
        
        lines.extend(["", "## 币种信号"])
        
        for symbol in POOL_SYMBOLS_RAW:
            signals = digest.symbol_signals.get(symbol, [])
            lines.extend(["", f"### {symbol}", f"共 {len(signals)} 条信号", ""])
            
            if signals:
                sorted_signals = sorted(signals, key=lambda x: x.urgency)
                for sig in sorted_signals[:5]:
                    lines.extend([
                        f"**{sig.title}**",
                        f"- 分类: {sig.category} | 情感: {sig.sentiment:+.2f} | 置信: {sig.confidence}",
                        f"- 关键词: {', '.join(sig.keywords_matched)}",
                        f"- [链接]({sig.source_url})",
                        "",
                    ])
            else:
                lines.append("- 暂无相关信号")
        
        lines.extend([
            "",
            "## 待办事项",
            "",
            "1. [ ] 审核 `daily_news_digest.json`",
            "2. [ ] 确认重要信号并编辑 `event_signals.json`",
            "3. [ ] 运行研究 Agent 生成交易决策",
            "",
            "---",
            "",
            "**注意**: 此摘要仅供复盘参考，交易决策以人工确认后的 `event_signals.json` 为准。",
        ])
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return filepath
    
    def run(self):
        """运行完整流程"""
        print("=" * 50)
        print("新闻收集与摘要生成")
        print("=" * 50)
        
        digest = self.collect_daily()
        
        print("\n生成输出文件...")
        json_path = self.generate_digest_json(digest)
        md_path = self.generate_summary_md(digest)
        
        print(f"\n完成!")
        print(f"  - JSON (供审核): {json_path}")
        print(f"  - Markdown (供复盘): {md_path}")
        print(f"\n下一步: 人工审核后编辑 event_signals.json")


def main():
    agent = NewsDigestAgent()
    agent.run()


if __name__ == "__main__":
    main()