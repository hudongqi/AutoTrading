#!/usr/bin/env python3
"""
每日新闻收集脚本
建议添加到 cron: 0 7 * * * cd /path/to/AutoTrading && python3 run_news_collection.py
"""

from news_collector import NewsDigestAgent


def main():
    try:
        agent = NewsDigestAgent()
        agent.run()
    except Exception as e:
        # 容错：单次新闻流程失败不应让整个 pipeline 崩溃
        print(f"[WARN] run_news_collection failed but pipeline can continue: {e}")


if __name__ == "__main__":
    main()
