# Research Agent for High Volatility Pool

## Overview

This is a **Research & Event Filter Agent** for the high volatility altcoin pool. It provides structured research materials for the main strategy and execution layer without directly placing orders or modifying trading parameters.

## Responsibilities

### 1. Daily Data Collection
For each of the 5 coins in the pool:
- 1h / 4h trend direction
- ATR and ATR%
- 24h trading volume
- Bid-ask spread
- Open interest
- Funding rate
- Breakout continuation rate (6h / 12h)
- Fake breakout rate
- Recent event-driven movements or sector resonance

### 2. Event & Risk Filtering
Outputs structured labels based on macro and news:
- **ALLOW**: Aggressive sub-strategy can participate
- **REDUCE_RISK**: Can participate but with reduced risk
- **BLOCK**: Prohibited from participation

Key focus areas:
- CPI, NFP, FOMC, PCE macro data windows
- Coin-specific major news
- Sector hotspots and sentiment changes
- Funding / OI overheating

### 3. Candidate Pool Ranking
Daily scoring and ranking of 5 coins to find the best 1-2 coins for trading that day.

Scoring dimensions:
- volatility_score
- liquidity_score
- trend_score
- continuation_score
- fake_breakout_penalty
- funding_penalty
- oi_crowding_penalty

## Output Files

### JSON Report
`research/high_vol_pool/daily/YYYY-MM-DD_candidates.json`

```json
{
  "date": "2026-03-08",
  "macro_state": "ALLOW",
  "pool_status": "ACTIVE",
  "top_candidates": ["SOLUSDT", "XRPUSDT"],
  "overall_risk_recommendation": "NORMAL",
  "symbols": [
    {
      "symbol": "SOLUSDT",
      "score": 0.87,
      "decision": "ALLOW",
      "trend_4h": "UP",
      "atr_pct": 0.0052,
      "spread_score": 8.2,
      "funding_state": "NORMAL",
      "oi_state": "WARM",
      "continuation_score": 0.89,
      "fake_breakout_risk": 0.21,
      "reasons": ["4h趋势明确", "延续率较高", "流动性充足"]
    }
  ]
}
```

### Markdown Summary
`research/high_vol_pool/daily/YYYY-MM-DD_summary.md`

Contains:
- Today's macro risks
- Candidate pool ranking
- Recommended coins and reasons
- Filtered coins and reasons
- Overall risk recommendation

## Usage

```python
from research_agent import HighVolPoolResearchAgent

agent = HighVolPoolResearchAgent()
report = agent.run_daily_research()
```

## Event Signals File

Create `event_signals.json` to inject external signals:

```json
{
  "macro": {
    "block": false,
    "reduce_risk": true,
    "reason": "FOMC meeting today"
  },
  "symbols": {
    "SOLUSDT": {
      "block": false,
      "reduce_risk": false
    },
    "PEPEUSDT": {
      "block": true,
      "reason": "Major token unlock today"
    }
  }
}
```

## Pool Configuration

| Symbol | Risk Bucket | Category |
|--------|-------------|----------|
| SOLUSDT | 1.0 | Main pool |
| XRPUSDT | 1.0 | Main pool |
| DOGEUSDT | 0.7 | High elasticity |
| SUIUSDT | 0.7 | High elasticity |
| PEPEUSDT | 0.4 | High emotion, strictest rules |

## Scoring Weights

| Factor | Weight |
|--------|--------|
| Volatility | +22% |
| Liquidity | +20% |
| Trend | +20% |
| Continuation | +18% |
| Fake breakout penalty | -10% |
| Funding penalty | -5% |
| OI crowding penalty | -5% |

## Special Rules

1. **PEPE has extra restrictions**: continuation_score must be > 0.55 AND event_label must be ALLOW
2. **Max 2 coins per day**
3. **Minimum total score threshold**: 0.55
4. **NFP Fridays**: automatic REDUCE_RISK for macro