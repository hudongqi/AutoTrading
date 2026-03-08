#!/usr/bin/env python3
"""
每日运行脚本 - 高波动池研究 Agent
建议添加到 cron: 0 8 * * * cd /path/to/AutoTrading && python3 run_research.py
"""

from research_agent import HighVolPoolResearchAgent

def main():
    agent = HighVolPoolResearchAgent()
    report = agent.run_daily_research()
    
    print("\n" + "="*50)
    print("研究摘要")
    print("="*50)
    print(f"日期: {report.date}")
    print(f"宏观状态: {report.macro_state}")
    print(f"池状态: {report.pool_status}")
    print(f"候选币: {report.top_candidates if report.top_candidates else '无'}")
    print(f"整体建议: {report.overall_risk_recommendation}")
    print("="*50)

if __name__ == "__main__":
    main()