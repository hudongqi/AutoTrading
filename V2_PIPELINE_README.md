# V2 Pipeline (Unified)

## 主入口
- `run_daily_pipeline.py`（生产日报入口，推荐）
- `run_news_collection.py`（仅新闻采集）
- `run_research_v2.py`（仅研究与候选输出）

## 已废弃（Deprecated）
- `research_agent.py`
- `run_research.py`
- `event_signals_v2.json`

> 这些文件保留仅为兼容提示，生产流程不要再调用。

## 统一事件信号文件
仅保留：`event_signals.json`

### schema（最终）
```json
{
  "schema_version": "2.1",
  "last_updated_utc": "ISO-8601",
  "macro": {
    "block": false,
    "reduce_risk": false,
    "reason": "",
    "upcoming_events": [
      {"date": "YYYY-MM-DD", "name": "...", "impact": "low|medium|high"}
    ]
  },
  "whale": {
    "enabled": true,
    "mode": "auxiliary",
    "note": "Only for sizing/confidence/candidate filtering, never standalone trade trigger.",
    "last_updated_utc": "ISO-8601"
  },
  "symbols": {
    "SOLUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": [], "whale_score": 0.0, "whale_bias": "NEUTRAL", "whale_reason": ""},
    "XRPUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": [], "whale_score": 0.0, "whale_bias": "NEUTRAL", "whale_reason": ""},
    "DOGEUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": [], "whale_score": 0.0, "whale_bias": "NEUTRAL", "whale_reason": ""},
    "SUIUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": [], "whale_score": 0.0, "whale_bias": "NEUTRAL", "whale_reason": ""},
    "PEPEUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": [], "whale_score": 0.0, "whale_bias": "NEUTRAL", "whale_reason": ""}
  }
}
```

## 日报固定流程
```bash
python3 run_daily_pipeline.py
```

流程：
1. 新闻采集（生成 digest/summary）
2. Whale 辅助信号收集并合并到 `event_signals.json`
3. 人工确认并编辑 `event_signals.json`
4. 运行 `research_agent_v2.py`
5. 输出最终候选池结果
