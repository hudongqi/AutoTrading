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
  "schema_version": "2.0",
  "last_updated_utc": "ISO-8601",
  "macro": {
    "block": false,
    "reduce_risk": false,
    "reason": "",
    "upcoming_events": [
      {"date": "YYYY-MM-DD", "name": "...", "impact": "low|medium|high"}
    ]
  },
  "symbols": {
    "SOLUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": []},
    "XRPUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": []},
    "DOGEUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": []},
    "SUIUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": []},
    "PEPEUSDT": {"block": false, "reduce_risk": false, "reason": "", "news_signals": []}
  }
}
```

## 日报固定流程
```bash
python3 run_daily_pipeline.py
```

流程：
1. 新闻采集（生成 digest/summary）
2. 人工确认并编辑 `event_signals.json`
3. 运行 `research_agent_v2.py`
4. 输出最终候选池结果
