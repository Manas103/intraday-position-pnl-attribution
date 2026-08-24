#!/usr/bin/env python
"""Demonstrate idempotent ingest keyed on (fill_id, sequence): replay the
same message 1, 10 and 1000 times and show the accepted count and the
resulting position are unaffected. This is the exact script referenced in
the README.

Usage:
    python scripts/run_idempotent_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import db, ingest
from pipeline.simulator import Fill


def main():
    fill = Fill(fill_id="FILL-OPT00-000001", sequence=1, instrument="OPT00", quantity=10.0, price=101.5, session_id=0)

    for n_repeats in (1, 10, 1000):
        conn = db.connect(":memory:")
        stream = [fill] * n_repeats
        stats = ingest.ingest_stream(conn, stream)
        ingest.rebuild_positions(conn)
        positions = ingest.final_positions(conn)
        print(f"n_repeats={n_repeats:5d}  accepted={stats['accepted']}  duplicates={stats['duplicates']}  "
              f"position={positions.get('OPT00')}")
        assert stats["accepted"] == 1, "idempotency violated: more than one copy accepted"
        assert stats["duplicates"] == n_repeats - 1
        assert positions.get("OPT00") == (10.0, 101.5)

    # Also check idempotency holds when the duplicate arrives interleaved
    # with other, distinct fills rather than back-to-back.
    other = Fill(fill_id="FILL-OPT00-000002", sequence=2, instrument="OPT00", quantity=-3.0, price=102.0, session_id=0)
    conn = db.connect(":memory:")
    stream = [fill, other, fill, fill, other]
    stats = ingest.ingest_stream(conn, stream)
    ingest.rebuild_positions(conn)
    positions = ingest.final_positions(conn)
    print(f"interleaved duplicates: accepted={stats['accepted']} duplicates={stats['duplicates']} "
          f"position={positions.get('OPT00')}")
    assert stats["accepted"] == 2
    assert stats["duplicates"] == 3

    print()
    print("claim: idempotent ingest keyed on fill ID and sequence number -> MEETS")


if __name__ == "__main__":
    main()
