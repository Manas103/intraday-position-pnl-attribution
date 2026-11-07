"""Measures: end-of-day reconciliation against an independent replay
catches 200 of 200 seeded breaks, with 0 false positives on everything
else. Builds the live EOD grid (SQL as-of queries against the PostgreSQL
ledger) and the independent EOD grid (a vectorized pandas fold over the
in-memory canonical stream, no SQL involved), corrupts 200 cells of the
independent grid, and reconciles.

Usage: python3 scripts/run_risk_platform_reconciliation.py [n_instruments] [n_sessions] [n_breaks]
"""
import sys

sys.path.insert(0, ".")
from pipeline.risk_platform import pg_ingest, reconciliation, simulator

N_INSTRUMENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 420
N_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
N_BREAKS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
SEED = 42


def main():
    conn = pg_ingest.connect()
    pg_ingest.ensure_schema(conn)
    pg_ingest.reset_tables(conn)

    canonical = list(simulator.generate_canonical_stream(SEED, N_INSTRUMENTS, N_SESSIONS))
    pg_ingest.ingest_stream(conn, canonical)

    instruments = simulator.instrument_ids(N_INSTRUMENTS)
    live = reconciliation.live_eod_grid(conn, instruments, N_SESSIONS)
    replay_raw = reconciliation.independent_eod_grid(canonical)
    replay_full = reconciliation.full_eod_grid(replay_raw, instruments, N_SESSIONS)

    # Sanity check before injecting anything: the two grids must already
    # agree everywhere, or "catches 200 of 200 seeded breaks" would be
    # meaningless (it would be impossible to tell a seeded break from a
    # pre-existing, unrelated disagreement).
    baseline = reconciliation.reconcile(live, replay_full)
    baseline_breaks = int(baseline["break"].sum())
    print(f"instruments={N_INSTRUMENTS} sessions={N_SESSIONS} grid cells={len(baseline)}")
    print(f"pre-injection disagreements (must be 0): {baseline_breaks}")

    corrupted, broken_keys = reconciliation.inject_seeded_breaks(replay_full, N_BREAKS, seed=SEED + 2)
    result = reconciliation.reconcile(live, corrupted)

    flagged = result[result["break"]]
    flagged_keys = set(zip(flagged["instrument"], flagged["session_id"].astype(int)))

    caught = broken_keys & flagged_keys
    missed = broken_keys - flagged_keys
    false_positives = flagged_keys - broken_keys

    print(f"seeded breaks: {len(broken_keys)}")
    print(f"caught: {len(caught)} of {len(broken_keys)}")
    print(f"missed: {len(missed)}")
    print(f"false positives (flagged but not seeded): {len(false_positives)}")
    print(f"clean cells (not seeded, not flagged): {len(result) - len(broken_keys) - len(false_positives)}")
    print()
    print(f"meets '200 of 200 seeded breaks caught': {len(caught) == N_BREAKS and len(missed) == 0}")
    print(f"meets '0 false positives': {len(false_positives) == 0}")

    conn.close()


if __name__ == "__main__":
    main()
