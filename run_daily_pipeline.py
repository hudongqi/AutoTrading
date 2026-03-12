#!/usr/bin/env python3
"""
Production daily pipeline (v2 only)

Flow:
  a) run_news_collection.py
  b) run_whale_collection.py (auxiliary smart-money signals)
  c) validate unified event_signals.json schema
  d) optional manual confirmation gate
  e) run_research_v2.py
  f) write final trading-oriented research fields back to event_signals.json
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
PYTHON_BIN = str((ROOT / ".venv" / "bin" / "python")) if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable

POOL_SYMBOLS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "PEPEUSDT"]
REQUIRE_MANUAL_CONFIRM = True


def run_cmd(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_event_signals(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    required_top = ["macro", "geopolitics", "sentiment", "whale", "symbols", "risk_mode", "market_bias"]
    for k in required_top:
        if k not in data:
            raise ValueError(f"event_signals.json missing top-level field: {k}")

    macro = data["macro"]
    for k in ["block", "reduce_risk", "reason", "upcoming_events"]:
        if k not in macro:
            raise ValueError(f"event_signals.json macro missing field: {k}")

    geo = data["geopolitics"]
    for k in ["reduce_risk", "block_new_entries", "alts_bias", "reason"]:
        if k not in geo:
            raise ValueError(f"event_signals.json geopolitics missing field: {k}")

    sent = data["sentiment"]
    for k in ["bias", "strength", "recommended_action", "reason"]:
        if k not in sent:
            raise ValueError(f"event_signals.json sentiment missing field: {k}")

    whale = data["whale"]
    for k in ["enabled", "mode", "note", "last_updated_utc", "whale_score", "whale_bias", "whale_reason"]:
        if k not in whale:
            raise ValueError(f"event_signals.json whale missing field: {k}")

    symbols = data["symbols"]
    for sym in POOL_SYMBOLS:
        if sym not in symbols:
            raise ValueError(f"event_signals.json symbols missing key: {sym}")
        for k in [
            "block", "reduce_risk", "reason", "news_signals",
            "whale_score", "whale_bias", "whale_reason",
            "bias", "strength", "recommended_action",
        ]:
            if k not in symbols[sym]:
                raise ValueError(f"event_signals.json symbols.{sym} missing field: {k}")

    return data


def latest_news_outputs():
    if not NEWS_DAILY_DIR.exists():
        return None, None, None
    digests = sorted(NEWS_DAILY_DIR.glob("*_digest.json"))
    summaries = sorted(NEWS_DAILY_DIR.glob("*_summary.md"))
    trade_research = sorted(NEWS_DAILY_DIR.glob("*_trade_research.json"))
    return (
        digests[-1] if digests else None,
        summaries[-1] if summaries else None,
        trade_research[-1] if trade_research else None,
    )


def manual_gate(require_manual: bool, skip_confirm: bool):
    if not require_manual or skip_confirm:
        print("[INFO] Manual gate bypassed (auto mode or --skip-confirm).")
        return

    print("\n[MANUAL CONFIRMATION REQUIRED]")
    print("1) Review latest news digest/trade_research/summary")
    print("2) Update event_signals.json")
    answer = input("Type 'CONFIRM' to continue: ").strip()
    if answer != "CONFIRM":
        raise SystemExit("Manual confirmation not provided. Pipeline stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run daily production pipeline (v2)")
    parser.add_argument("--window", type=int, default=120, help="research window days for run_research_v2.py")
    parser.add_argument("--skip-confirm", action="store_true", help="skip manual confirmation gate")
    parser.add_argument("--auto", action="store_true", help="force auto mode (no manual prompt)")
    args = parser.parse_args()

    require_manual = REQUIRE_MANUAL_CONFIRM and not args.auto

    now = datetime.now(timezone.utc).isoformat()
    print(f"=== DAILY PIPELINE START {now} ===")

    mode = "AUTO MODE" if args.auto else "MANUAL MODE"
    print(f"[MODE] {mode}")

    run_cmd([PYTHON_BIN, "run_news_collection.py"])
    run_cmd([PYTHON_BIN, "run_whale_collection.py"])

    digest, summary, trade_research = latest_news_outputs()
    print("\nLatest news outputs:")
    print(f"- digest:         {digest}")
    print(f"- trade_research: {trade_research}")
    print(f"- summary:        {summary}")

    _ = validate_event_signals(EVENT_FILE)
    print(f"\nValidated unified signals schema: {EVENT_FILE}")

    manual_gate(require_manual=require_manual, skip_confirm=args.skip_confirm)

    run_cmd([PYTHON_BIN, "run_research_v2.py", "--window", str(args.window)])

    research_daily_dir = ROOT / "research" / "high_vol_pool" / "daily"
    print("\nFinal candidate outputs are under:")
    print(f"- {research_daily_dir}")

    print("\n=== DAILY PIPELINE DONE ===")


if __name__ == "__main__":
    main()
