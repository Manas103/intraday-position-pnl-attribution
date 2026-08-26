#!/usr/bin/env python
"""Demonstrate idempotent commodity deal ingest keyed on (deal_id,
sequence): replay the same message 1, 10 and 1000 times and show the
accepted count and resulting position are unaffected.

Usage:
    python scripts/run_commodity_idempotent_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import commodity_ingest as cingest
from pipeline import db
from pipeline.commodity_simulator import Deal


def main():
    deal = Deal("DEAL-WTI-320D-000001", 1, "new", "WTI", 320, 10.0, 70.0, "buy", 0, "2026-01-05T09:00:00")

    for n_repeats in (1, 10, 1000):
        conn = db.connect(":memory:")
        stream = [deal] * n_repeats
        stats = cingest.ingest_stream(conn, stream)
        cingest.rebuild_positions(conn)
        positions = cingest.final_positions(conn)
        print(f"n_repeats={n_repeats:5d}  accepted={stats['accepted']}  duplicates={stats['duplicates']}  "
              f"position={positions.get('WTI-320D')}")
        assert stats["accepted"] == 1
        assert stats["duplicates"] == n_repeats - 1
        assert positions.get("WTI-320D") == 10.0

    other = Deal("DEAL-WTI-320D-000002", 2, "new", "WTI", 320, -3.0, 71.0, "sell", 0, "2026-01-05T09:05:00")
    conn = db.connect(":memory:")
    stream = [deal, other, deal, deal, other]
    stats = cingest.ingest_stream(conn, stream)
    cingest.rebuild_positions(conn)
    positions = cingest.final_positions(conn)
    print(f"interleaved duplicates: accepted={stats['accepted']} duplicates={stats['duplicates']} "
          f"position={positions.get('WTI-320D')}")
    assert stats["accepted"] == 2
    assert stats["duplicates"] == 3

    print()
    print("claim: idempotent ingest keyed on deal ID and sequence -> MEETS")


if __name__ == "__main__":
    main()
