"""Tests for the PostgreSQL-backed risk platform extension.

Per the playbook: PostgreSQL is not installed as a standing service for
this repository. These tests try to import psycopg2 and connect to
127.0.0.1:5432/risk_platform; if either fails (no driver, no server, wrong
credentials), every test in this module skips cleanly rather than failing,
exactly like this repo's precedent (model-validation-alerting) for an
optional broker/DB dependency. On this project's own development machine
the driver and server are both present (see README: "Standing up
PostgreSQL for this extension"), so the tests run for real there.

Scale here is deliberately small (tens of instruments/sessions, not the
2M+ fills the benchmark scripts measure): these are correctness tests, run
on every `pytest` invocation, not a benchmark.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import psycopg2  # noqa: F401
    from pipeline.risk_platform import asof, pg_ingest, reconciliation, simulator

    def _can_connect():
        try:
            conn = pg_ingest.connect()
            conn.close()
            return True
        except Exception:
            return False

    HAVE_PG = _can_connect()
except ImportError:
    HAVE_PG = False

pytestmark = pytest.mark.skipif(not HAVE_PG, reason="psycopg2 not installed or PostgreSQL unreachable at 127.0.0.1:5432/risk_platform")


@pytest.fixture()
def conn():
    c = pg_ingest.connect()
    pg_ingest.ensure_schema(c)
    pg_ingest.reset_tables(c)
    yield c
    c.close()


def test_duplicate_exact_fill_accepted_once(conn):
    f = simulator.Fill("FILL-A-0000001", 1, "A", 10.0, 100.0, 0)
    accepted = pg_ingest.ingest_stream(conn, [f, f, f])
    assert accepted == 1
    assert pg_ingest.total_rows(conn) == 1


def test_full_stream_replay_is_idempotent(conn):
    canonical = list(simulator.generate_canonical_stream(seed=7, n_instruments=3, n_sessions=5))
    first = pg_ingest.ingest_stream(conn, canonical)
    rows_after_first = pg_ingest.total_rows(conn)
    second = pg_ingest.ingest_stream(conn, canonical)
    assert first == len(canonical)
    assert second == 0
    assert pg_ingest.total_rows(conn) == rows_after_first


def test_asof_position_matches_independent_fold_at_historical_cutoffs(conn):
    canonical = list(simulator.generate_canonical_stream(seed=11, n_instruments=4, n_sessions=8))
    pg_ingest.ingest_stream(conn, canonical)
    instruments = simulator.instrument_ids(4)
    result = asof.check_reconstruction(conn, canonical, instruments, cutoffs_per_instrument=6, seed=99)
    assert result["mismatches"] == []
    assert result["matched"] == result["checked"]
    assert result["checked"] > 0


def test_asof_at_final_sequence_equals_total_position(conn):
    canonical = list(simulator.generate_canonical_stream(seed=3, n_instruments=2, n_sessions=4))
    pg_ingest.ingest_stream(conn, canonical)
    inst = simulator.instrument_ids(2)[0]
    max_seq = max(f.sequence for f in canonical if f.instrument == inst)
    expected = sum(f.quantity for f in canonical if f.instrument == inst)
    assert asof.position_as_of(conn, inst, max_seq) == expected


def test_replay_with_duplicates_and_reordering_is_bit_identical(conn):
    canonical = list(simulator.generate_canonical_stream(seed=5, n_instruments=3, n_sessions=6))
    perturbed = simulator.perturb_stream(seed=5, canonical=canonical)
    assert len(perturbed) > len(canonical)  # duplicates really were added

    pg_ingest.reset_tables(conn)
    pg_ingest.ingest_stream(conn, canonical)
    with conn.cursor() as cur:
        cur.execute("SELECT instrument, SUM(quantity) FROM risk_fills GROUP BY instrument ORDER BY instrument")
        canonical_positions = dict(cur.fetchall())

    pg_ingest.reset_tables(conn)
    pg_ingest.ingest_stream(conn, perturbed)
    with conn.cursor() as cur:
        cur.execute("SELECT instrument, SUM(quantity) FROM risk_fills GROUP BY instrument ORDER BY instrument")
        perturbed_positions = dict(cur.fetchall())

    assert canonical_positions == perturbed_positions


def test_reconciliation_catches_seeded_breaks_with_no_false_positives(conn):
    n_instruments, n_sessions, n_breaks = 6, 10, 15
    canonical = list(simulator.generate_canonical_stream(seed=13, n_instruments=n_instruments, n_sessions=n_sessions))
    pg_ingest.ingest_stream(conn, canonical)

    instruments = simulator.instrument_ids(n_instruments)
    live = reconciliation.live_eod_grid(conn, instruments, n_sessions)
    replay = reconciliation.full_eod_grid(reconciliation.independent_eod_grid(canonical), instruments, n_sessions)

    baseline = reconciliation.reconcile(live, replay)
    assert baseline["break"].sum() == 0  # no pre-existing disagreement to confound the test

    corrupted, broken_keys = reconciliation.inject_seeded_breaks(replay, n_breaks, seed=17)
    result = reconciliation.reconcile(live, corrupted)
    flagged = result[result["break"]]
    flagged_keys = set(zip(flagged["instrument"], flagged["session_id"].astype(int)))

    assert flagged_keys == broken_keys
    assert len(broken_keys) == n_breaks
