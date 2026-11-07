"""Measures: 2M+ fills ingested idempotently into PostgreSQL, keyed on
(fill_id, sequence). Generates the canonical stream once, ingests it, then
re-ingests the exact same stream a second time (a full-stream replay, not
just a handful of duplicates) and checks that the second pass accepts
zero new rows and the total row count is unchanged -- the strongest,
simplest idempotency demonstration: replaying the entire stream twice
must be indistinguishable in the database from replaying it once.

Usage: python3 scripts/run_risk_platform_ingest.py [n_instruments] [n_sessions]
"""
import sys
import time

sys.path.insert(0, ".")
from pipeline.risk_platform import pg_ingest, simulator

N_INSTRUMENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 420
N_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
SEED = 42


def main():
    conn = pg_ingest.connect()
    pg_ingest.ensure_schema(conn)
    pg_ingest.reset_tables(conn)

    t0 = time.time()
    n_canonical = 0

    def counted_stream():
        nonlocal n_canonical
        for f in simulator.generate_canonical_stream(SEED, N_INSTRUMENTS, N_SESSIONS):
            n_canonical += 1
            yield f

    accepted_first = pg_ingest.ingest_stream(conn, counted_stream())
    t1 = time.time()
    rows_after_first = pg_ingest.total_rows(conn)

    accepted_second = pg_ingest.ingest_stream(conn, simulator.generate_canonical_stream(SEED, N_INSTRUMENTS, N_SESSIONS))
    rows_after_second = pg_ingest.total_rows(conn)

    print(f"instruments={N_INSTRUMENTS} sessions={N_SESSIONS} seed={SEED}")
    print(f"canonical stream generated: {n_canonical} fills")
    print(f"first ingest pass: accepted={accepted_first} in {t1 - t0:.2f}s ({n_canonical / (t1 - t0):.0f} fills/sec)")
    print(f"rows in risk_fills after first pass: {rows_after_first}")
    print(f"second ingest pass (full-stream replay): accepted={accepted_second}")
    print(f"rows in risk_fills after second pass: {rows_after_second}")
    print()
    print(f"meets '2M+ fills ingested idempotently': {rows_after_first >= 2_000_000} "
          f"(measured {rows_after_first})")
    print(f"idempotent under full-stream replay: {accepted_second == 0 and rows_after_second == rows_after_first}")

    conn.close()


if __name__ == "__main__":
    main()
