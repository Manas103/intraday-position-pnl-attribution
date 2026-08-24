from pipeline import db, ingest
from pipeline.simulator import Fill


def make_conn():
    return db.connect(":memory:")


def test_duplicate_exact_message_is_noop():
    conn = make_conn()
    fill = Fill("FILL-A-000001", 1, "A", 10.0, 100.0, 0)
    stats = ingest.ingest_stream(conn, [fill, fill, fill])
    assert stats == {"total_messages": 3, "accepted": 1, "duplicates": 2}
    ingest.rebuild_positions(conn)
    assert ingest.final_positions(conn) == {"A": (10.0, 100.0)}


def test_duplicate_many_times_is_still_a_noop():
    conn = make_conn()
    fill = Fill("FILL-A-000001", 1, "A", 5.0, 50.0, 0)
    stats = ingest.ingest_stream(conn, [fill] * 1000)
    assert stats["accepted"] == 1
    assert stats["duplicates"] == 999


def test_out_of_order_delivery_same_final_position():
    fills = [
        Fill("FILL-A-000001", 1, "A", 10.0, 100.0, 0),
        Fill("FILL-A-000002", 2, "A", 5.0, 110.0, 0),
        Fill("FILL-A-000003", 3, "A", -20.0, 90.0, 0),
    ]
    conn_forward = make_conn()
    ingest.ingest_stream(conn_forward, fills)
    ingest.rebuild_positions(conn_forward)

    conn_reversed = make_conn()
    ingest.ingest_stream(conn_reversed, list(reversed(fills)))
    ingest.rebuild_positions(conn_reversed)

    assert ingest.final_positions(conn_forward) == ingest.final_positions(conn_reversed)


def test_fully_reversed_large_stream_matches_forward():
    fills = [Fill(f"FILL-A-{i:06d}", i, "A", float((-1) ** i * (i % 7 + 1)), 100.0 + i, 0) for i in range(1, 201)]
    conn_forward = make_conn()
    ingest.ingest_stream(conn_forward, fills)
    ingest.rebuild_positions(conn_forward)

    conn_reversed = make_conn()
    ingest.ingest_stream(conn_reversed, list(reversed(fills)))
    ingest.rebuild_positions(conn_reversed)

    assert ingest.final_positions(conn_forward) == ingest.final_positions(conn_reversed)


def test_position_depends_only_on_accepted_set_not_on_arrival_order_or_duplication():
    fills = [
        Fill("FILL-A-000001", 1, "A", 10.0, 100.0, 0),
        Fill("FILL-A-000002", 2, "A", 5.0, 110.0, 0),
        Fill("FILL-A-000003", 3, "A", -20.0, 90.0, 0),
    ]
    noisy = [fills[2], fills[0], fills[2], fills[1], fills[0]]  # duplicated + reversed-ish

    conn_clean = make_conn()
    ingest.ingest_stream(conn_clean, fills)
    ingest.rebuild_positions(conn_clean)

    conn_noisy = make_conn()
    stats = ingest.ingest_stream(conn_noisy, noisy)
    ingest.rebuild_positions(conn_noisy)

    assert stats["accepted"] == 3
    assert stats["duplicates"] == 2
    assert ingest.final_positions(conn_clean) == ingest.final_positions(conn_noisy)


def _naive_arrival_order_fold(fills_in_arrival_order):
    """Reproduces the alternative, order-sensitive design we rejected:
    fold fills as they arrive instead of sorting by sequence first. Kept
    here only to prove, with a real example, why the production code in
    pipeline/ingest.py always sorts by sequence before folding. See
    README Findings for the numbers this test's assertion is based on."""
    qty, avg_price = 0.0, 0.0
    for f in fills_in_arrival_order:
        qty, avg_price = ingest._fold(qty, avg_price, f.quantity, f.price)
    return qty, avg_price


def test_naive_arrival_order_fold_diverges_from_sequence_sorted_fold():
    # Three fills whose flip point (long -> short) differs depending on
    # which arrives first, which is exactly the situation sequence-sorting
    # is designed to make irrelevant.
    seq1 = Fill("FILL-A-000001", 1, "A", 10.0, 100.0, 0)   # buy 10 @ 100
    seq2 = Fill("FILL-A-000002", 2, "A", -12.0, 90.0, 0)   # sell 12 @ 90
    seq3 = Fill("FILL-A-000003", 3, "A", -5.0, 80.0, 0)    # sell 5 @ 80

    correct_qty, correct_avg = _naive_arrival_order_fold([seq1, seq2, seq3])
    assert (correct_qty, round(correct_avg, 4)) == (-7.0, round(82.857142857142854, 4))

    # Deliver seq3 before seq2: an out-of-order arrival that a naive,
    # arrival-order fold applies directly, producing a different average
    # price for the same final position.
    out_of_order_qty, out_of_order_avg = _naive_arrival_order_fold([seq1, seq3, seq2])
    assert out_of_order_qty == correct_qty == -7.0
    assert out_of_order_avg != correct_avg
    assert out_of_order_avg == 90.0

    # The production rebuild_positions always sorts by sequence first, so
    # it gives the correct, order-independent answer for both deliveries.
    conn_a = db.connect(":memory:")
    ingest.ingest_stream(conn_a, [seq1, seq2, seq3])
    ingest.rebuild_positions(conn_a)

    conn_b = db.connect(":memory:")
    ingest.ingest_stream(conn_b, [seq1, seq3, seq2])
    ingest.rebuild_positions(conn_b)

    assert ingest.final_positions(conn_a) == ingest.final_positions(conn_b) == {"A": (-7.0, correct_avg)}
