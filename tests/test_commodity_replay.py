from collections import Counter

from pipeline import commodity_ingest as cingest
from pipeline import commodity_replay as creplay
from pipeline import commodity_simulator as csim
from pipeline import db

SEED = 42


def _build(n_sessions):
    universe = csim.build_commodity_universe()
    commodities = sorted({inst.commodity for inst in universe})
    curve_sessions = {c: csim.build_curve_sessions(SEED, c, n_sessions) for c in commodities}
    return csim.build_deal_blotter(SEED, universe, curve_sessions, n_sessions)


def test_perturb_stream_preserves_the_message_multiset_of_originals():
    canonical = _build(20)
    perturbed = creplay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    canonical_keys = Counter((d.deal_id, d.sequence) for d in canonical)
    perturbed_keys = Counter((d.deal_id, d.sequence) for d in perturbed)

    assert set(perturbed_keys) == set(canonical_keys)
    for key, count in perturbed_keys.items():
        assert count in (1, 2)


def test_measured_duplicate_fraction_is_close_to_five_percent():
    canonical = _build(250)
    perturbed = creplay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)
    frac = creplay.measured_duplicate_fraction(canonical, perturbed)
    assert abs(frac - 0.05) < 0.002


def test_perturbed_stream_is_actually_reordered_relative_to_canonical():
    canonical = _build(20)
    perturbed = creplay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    dedup_perturbed_order = []
    seen = set()
    for d in perturbed:
        key = (d.deal_id, d.sequence)
        if key not in seen:
            seen.add(key)
            dedup_perturbed_order.append(key)
    canonical_order = [(d.deal_id, d.sequence) for d in canonical]

    assert dedup_perturbed_order != canonical_order


def test_full_book_replay_positions_are_bit_identical():
    """The headline claim: ingesting a stream with 5% duplicated and
    bounded-out-of-order deal messages produces exactly the same positions
    as ingesting the canonical, correctly ordered stream, across the full
    250-session, 12-instrument commodity book."""
    canonical = _build(250)
    perturbed = creplay.perturb_stream(SEED, canonical, dup_fraction=0.05, window=10)

    conn_canonical = db.connect(":memory:")
    stats_canonical = cingest.ingest_stream(conn_canonical, canonical)
    cingest.rebuild_positions(conn_canonical)
    positions_canonical = cingest.final_positions(conn_canonical)

    conn_perturbed = db.connect(":memory:")
    stats_perturbed = cingest.ingest_stream(conn_perturbed, perturbed)
    cingest.rebuild_positions(conn_perturbed)
    positions_perturbed = cingest.final_positions(conn_perturbed)

    assert stats_canonical["accepted"] == stats_perturbed["accepted"] == len(canonical)
    assert stats_perturbed["duplicates"] > 0
    assert positions_canonical == positions_perturbed
    assert len(positions_canonical) == 12
