#!/usr/bin/env python3
from pathlib import Path
from whale_signal_collector import WhaleSignalCollector


def main():
    root = Path(__file__).resolve().parent
    collector = WhaleSignalCollector(root)
    signals = collector.collect()
    snap = collector.write_daily_snapshot(signals)
    event_file = collector.merge_into_event_signals(signals)

    print("[whale] done")
    print(f"[whale] snapshot: {snap}")
    print(f"[whale] merged into: {event_file}")


if __name__ == "__main__":
    main()
