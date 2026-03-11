#!/usr/bin/env python3
"""
Production daily pipeline (v2 only)

Flow:
  a) run_news_collection.py
  b) run_whale_collection.py (auxiliary smart-money signals)
  c) produce digest/summary files
  d) read unified event_signals.json (manual confirmation gate)
  e) run_research_v2.py
  f) output final candidate pool result
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EVENT_FILE = ROOT / "event_signals.json"
NEWS_DAILY_DIR = ROOT / "research" / "news" / "daily"

POOL_SYMBOLS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]


def run_cmd(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_event_signals(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    required_top = ["macro", "symbols", "whale"]
    for k in required_top:
        if k not in data:
            raise ValueError(f"event_signals.json missing top-level field: {k}")

    macro = data["macro"]
    for k in ["block", "reduce_risk", "reason", "upcoming_events"]:
        if k not in macro:
            raise ValueError(f"event_signals.json macro missing field: {k}")

    whale = data["whale"]
    for k in ["enabled", "mode", "note", "last_updated_utc"]:
        if k not in whale:
            raise ValueError(f"event_signals.json whale missing field: {k}")

    symbols = data["symbols"]
    for sym in POOL_SYMBOLS:
        if sym not in symbols:
            raise ValueError(f"event_signals.json symbols missing key: {sym}")
        for k in ["block", "reduce_risk", "reason", "news_signals", "whale_score", "whale_bias", "whale_reason"]:
            if k not in symbols[sym]:
                raise ValueError(f"event_signals.json symbols.{sym} missing field: {k}")

    return data


def latest_news_outputs():
    if not NEWS_DAILY_DIR.exists():
        return None, None
    digests = sorted(NEWS_DAILY_DIR.glob("*_digest.json"))
    summaries = sorted(NEWS_DAILY_DIR.glob("*_summary.md"))
    return (digests[-1] if digests else None, summaries[-1] if summaries else None)


def manual_gate(skip_confirm: bool):
    if skip_confirm:
        print("[WARN] --skip-confirm enabled, bypassing manual confirmation gate.")
        return

    print("\n[MANUAL CONFIRMATION REQUIRED]")
    print("1) Review latest news digest/summary")
    print("2) Update event_signals.json")
    answer = input("Type 'CONFIRM' to continue: ").strip()
    if answer != "CONFIRM":
        raise SystemExit("Manual confirmation not provided. Pipeline stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run daily production pipeline (v2)")
    parser.add_argument("--window", type=int, default=120, help="research window days for run_research_v2.py")
    parser.add_argument("--skip-confirm", action="store_true", help="skip manual confirmation gate")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    print(f"=== DAILY PIPELINE START {now} ===")

    # a) news collection
    run_cmd([sys.executable, "run_news_collection.py"])

    # b) whale auxiliary signals
    run_cmd([sys.executable, "run_whale_collection.py"])

    # c) locate digest/summary outputs
    digest, summary = latest_news_outputs()
    print("\nLatest news outputs:")
    print(f"- digest:  {digest}")
    print(f"- summary: {summary}")

    # c) read unified event_signals.json
    _ = validate_event_signals(EVENT_FILE)
    print(f"\nValidated unified signals schema: {EVENT_FILE}")

    # manual confirmation gate
    manual_gate(skip_confirm=args.skip_confirm)

    # d) run research v2
    run_cmd([sys.executable, "run_research_v2.py", "--window", str(args.window)])

    # e) final candidate output location
    research_daily_dir = ROOT / "research" / "high_vol_pool" / "daily"
    print("\nFinal candidate outputs are under:")
    print(f"- {research_daily_dir}")

    print("\n=== DAILY PIPELINE DONE ===")


if __name__ == "__main__":
    main()
