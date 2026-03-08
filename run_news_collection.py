#!/usr/bin/env python3
"""
每日新闻收集脚本
建议添加到 cron: 0 7 * * * cd /path/to/AutoTrading && python3 run_news_collection.py
"""

from news_collector import NewsDigestAgent

def main():
    agent = NewsDigestAgent()
    agent.run()

if __name__ == "__main__":
    main()