from pipeline import db, ingest, replay, simulator


SEED = 42


def test_perturb_stream_preserves_the_message_multiset_of_originals():
    universe = simulator.build_universe(SEED, n_instruments=3)
    canonical = simulator.build_canonical_fill_stream(SEED, universe, n_sessions=20)
    perturbed = replay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    # every perturbed message is either an original or a verbatim duplicate
    # of one; nothing is invented or dropped.
    from collections import Counter

    canonical_keys = Counter((f.fill_id, f.sequence) for f in canonical)
    perturbed_keys = Counter((f.fill_id, f.sequence) for f in perturbed)

    assert set(perturbed_keys) == set(canonical_keys)
    for key, count in perturbed_keys.items():
        assert count in (1, 2), "duplication step should only ever create a single extra copy per key"


def test_measured_duplicate_fraction_is_close_to_five_percent():
    universe = simulator.build_universe(SEED, n_instruments=8)
    canonical = simulator.build_canonical_fill_stream(SEED, universe, n_sessions=250)
    perturbed = replay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    frac = replay.measured_duplicate_fraction(canonical, perturbed)
    assert abs(frac - 0.05) < 0.002


def test_perturbed_stream_is_actually_reordered_relative_to_canonical():
    universe = simulator.build_universe(SEED, n_instruments=3)
    canonical = simulator.build_canonical_fill_stream(SEED, universe, n_sessions=20)
    perturbed = replay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    dedup_perturbed_order = []
    seen = set()
    for f in perturbed:
        key = (f.fill_id, f.sequence)
        if key not in seen:
            seen.add(key)
            dedup_perturbed_order.append(key)
    canonical_order = [(f.fill_id, f.sequence) for f in canonical]

    assert dedup_perturbed_order != canonical_order


def test_full_book_replay_positions_are_bit_identical():
    """The headline claim: ingesting a stream with 5% duplicated and
    bounded-out-of-order messages produces exactly (==, not approximately)
    the same positions as ingesting the canonical, correctly ordered
    stream, across the full 250-session, 8-instrument book."""
    universe = simulator.build_universe(SEED, n_instruments=8)
    canonical = simulator.build_canonical_fill_stream(SEED, universe, n_sessions=250)
    perturbed = replay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    conn_canonical = db.connect(":memory:")
    stats_canonical = ingest.ingest_stream(conn_canonical, canonical)
    ingest.rebuild_positions(conn_canonical)
    positions_canonical = ingest.final_positions(conn_canonical)

    conn_perturbed = db.connect(":memory:")
    stats_perturbed = ingest.ingest_stream(conn_perturbed, perturbed)
    ingest.rebuild_positions(conn_perturbed)
    positions_perturbed = ingest.final_positions(conn_perturbed)

    assert stats_canonical["accepted"] == stats_perturbed["accepted"] == len(canonical)
    assert stats_perturbed["duplicates"] > 0
    assert positions_canonical == positions_perturbed
    assert len(positions_canonical) == 8
