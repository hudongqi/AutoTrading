#!/usr/bin/env python3
"""
每日运行脚本 - 高波动池研究 Agent (增强版)
建议添加到 cron: 0 8 * * * cd /path/to/AutoTrading && python3 run_research_v2.py

增强功能:
1. 真实宏观/新闻输入
2. 滚动窗口数据 (默认120天)
3. 历史归档 CSV
"""

import sys
import argparse
from research_agent_v2 import HighVolPoolResearchAgent, ArchiveAgent

def main():
    parser = argparse.ArgumentParser(description='高波动池研究 Agent')
    parser.add_argument('--window', type=int, default=120, 
                        help='滚动窗口天数 (默认120, 范围60-180)')
    parser.add_argument('--stats', action='store_true',
                        help='显示历史统计')
    args = parser.parse_args()
    
    # 验证窗口范围
    window = max(60, min(180, args.window))
    if window != args.window:
        print(f"⚠️  窗口已调整至 {window} 天 (有效范围: 60-180)")
    
    if args.stats:
        # 显示历史统计
        archive = ArchiveAgent()
        stats = archive.generate_summary_stats()
        if stats:
            print("=" * 50)
            print("历史统计摘要")
            print("=" * 50)
            print(f"总记录数: {stats['total_records']}")
            print(f"日期范围: {stats['date_range']}")
            print("\n平均评分:")
            for sym, score in stats['avg_scores_by_symbol'].items():
                print(f"  {sym}: {score:.3f}")
            print("\n决策分布:")
            for dec, count in stats['decision_distribution'].items():
                print(f"  {dec}: {count}")
            print("\n候选率:")
            for sym, rate in stats['top_candidate_rate'].items():
                print(f"  {sym}: {rate:.1%}")
        else:
            print("暂无历史数据")
        return
    
    # 运行每日研究
    print("=" * 50)
    print(f"高波动池每日研究 (窗口: {window} 天)")
    print("=" * 50)
    
    agent = HighVolPoolResearchAgent(window_days=window)
    report = agent.run_daily_research()
    
    print("\n" + "=" * 50)
    print("研究摘要")
    print("=" * 50)
    print(f"日期: {report.date}")
    print(f"数据窗口: {report.data_window}")
    print(f"宏观状态: {report.macro_state}")
    print(f"池状态: {report.pool_status}")
    print(f"候选币: {report.top_candidates if report.top_candidates else '无'}")
    print(f"整体建议: {report.overall_risk_recommendation}")
    print("=" * 50)

if __name__ == "__main__":
    main()