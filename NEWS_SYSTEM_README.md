# News & Macro Data Collection System

## 新闻与宏观数据收集系统

### 目标
实现"自动抓取 + 人工确认 + 结构化落地"，**不做全自动新闻交易**。

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│  数据源层                                                    │
│  ├── CryptoPanic (免费 API)                                 │
│  ├── Cointelegraph (RSS)                                    │
│  ├── Binance Announcements                                  │
│  └── Macro Calendar (NFP/CPI 等)                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  收集层: news_collector.py                                   │
│  ├── 自动抓取                                                │
│  ├── 关键词匹配 (SOL/XRP/DOGE/SUI/PEPE)                      │
│  ├── 分类 (TECH/REGULATION/MARKET/SECURITY/MACRO)            │
│  └── 结构化输出                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  输出层 (供人工审核)                                         │
│  ├── research/news/daily/YYYY-MM-DD_digest.json             │
│  └── research/news/daily/YYYY-MM-DD_summary.md              │
└─────────────────────────────────────────────────────────────┘
                              ↓ (人工确认)
┌─────────────────────────────────────────────────────────────┐
│  决策层: event_signals.json (人工编辑)                       │
│  ├── macro.block / reduce_risk                              │
│  └── symbols.XXX.news_signals                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  交易层: research_agent.py                                   │
│  └── 只读取 event_signals.json，不直接读新闻                │
└─────────────────────────────────────────────────────────────┘
```

---

## 信号字段

每条信号包含：

| 字段 | 说明 | 示例 |
|------|------|------|
| `category` | 分类 | "安全", "监管", "市场", "技术升级", "宏观" |
| `sentiment` | 情感分数 | -1.0 ~ +1.0 |
| `urgency` | 紧急程度 | immediate / today / week / bg |
| `source` | 来源 | CryptoPanic / Cointelegraph / Binance / BLS |
| `impact_scope` | 影响范围 | single / sector / market |
| `confidence` | 置信度 | 0.0 ~ 1.0 |
| `related_symbols` | 相关币种 | ["SOLUSDT", "XRPUSDT"] |

---

## 每日工作流程

### 1. 自动收集 (早上 7:00)
```bash
python3 run_news_collection.py
```

输出：
- `research/news/daily/2026-03-08_digest.json` (结构化数据)
- `research/news/daily/2026-03-08_summary.md` (可读摘要)

### 2. 人工审核 (早上 7:10)
阅读 `daily_summary.md`，确认重要信号：
- 是否需要 BLOCK 某个币？
- 是否需要 REDUCE_RISK？
- 是否有重大宏观事件？

### 3. 编辑 event_signals.json
```json
{
  "macro": {
    "upcoming_events": [
      {"date": "2026-03-12", "name": "CPI数据", "impact": "high"}
    ]
  },
  "symbols": {
    "SOLUSDT": {
      "news_signals": ["Solana网络升级完成，性能提升30%"]
    }
  }
}
```

### 4. 运行研究 Agent (早上 8:00)
```bash
python3 run_research_v2.py
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `news_collector.py` | 核心收集器 |
| `run_news_collection.py` | 每日运行脚本 |
| `research/news/daily/*_digest.json` | 原始信号（供审核） |
| `research/news/daily/*_summary.md` | 可读摘要（供复盘） |
| `event_signals.json` | 人工确认后的信号（交易系统读取） |

---

## 注意事项

1. **不直接交易**：新闻系统只生成摘要，不直接生成交易指令
2. **人工确认层**：所有交易相关信号必须经过 `event_signals.json` 人工确认
3. **复盘用**：`daily_summary.md` 用于后续策略复盘和优化
4. **关键词可扩展**：编辑 `SYMBOL_KEYWORDS` 添加更多币种或关键词