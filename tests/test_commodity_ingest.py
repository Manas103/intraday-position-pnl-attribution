from pipeline import commodity_ingest as cingest
from pipeline import db
from pipeline.commodity_simulator import Deal


def make_conn():
    return db.connect(":memory:")


def d(deal_id, sequence, action, volume, session_id=0, trade_price=None):
    return Deal(deal_id, sequence, action, "WTI", 320, volume, trade_price, "buy" if volume >= 0 else "sell",
                session_id, f"S{session_id:04d}T{sequence:06d}")


def test_duplicate_exact_message_is_noop():
    conn = make_conn()
    deal = d("DEAL-1", 1, "new", 10.0, trade_price=70.0)
    stats = cingest.ingest_stream(conn, [deal, deal, deal])
    assert stats == {"total_messages": 3, "accepted": 1, "duplicates": 2}
    cingest.rebuild_positions(conn)
    assert cingest.final_positions(conn) == {"WTI-320D": 10.0}


def test_duplicate_many_times_is_still_a_noop():
    conn = make_conn()
    deal = d("DEAL-1", 1, "new", 5.0, trade_price=70.0)
    stats = cingest.ingest_stream(conn, [deal] * 1000)
    assert stats["accepted"] == 1
    assert stats["duplicates"] == 999


def test_amend_and_cancel_update_position():
    conn = make_conn()
    deals = [
        d("DEAL-1", 1, "new", 10.0, trade_price=70.0),
        d("DEAL-2", 2, "new", -4.0, trade_price=70.5),
        d("DEAL-1", 3, "amend", 6.0),   # resize DEAL-1 down to 6
        d("DEAL-2", 4, "cancel", 0.0),  # remove DEAL-2 entirely
    ]
    cingest.ingest_stream(conn, deals)
    cingest.rebuild_positions(conn)
    assert cingest.final_positions(conn) == {"WTI-320D": 6.0}


def test_out_of_order_delivery_same_final_position_when_no_conflicting_amends():
    deals = [
        d("DEAL-1", 1, "new", 10.0, trade_price=70.0),
        d("DEAL-2", 2, "new", 5.0, trade_price=71.0),
        d("DEAL-3", 3, "new", -3.0, trade_price=69.0),
    ]
    conn_forward = make_conn()
    cingest.ingest_stream(conn_forward, deals)
    cingest.rebuild_positions(conn_forward)

    conn_reversed = make_conn()
    cingest.ingest_stream(conn_reversed, list(reversed(deals)))
    cingest.rebuild_positions(conn_reversed)

    assert cingest.final_positions(conn_forward) == cingest.final_positions(conn_reversed)


def test_naive_arrival_order_fold_diverges_from_sequence_sorted_fold():
    """Two amends of the same deal, in sequence order: amend-to-4 (seq 2),
    amend-to-7 (seq 3). Sequence order says the deal ends at 7. Deliver
    seq 3 before seq 2 (a plausible reordering under the replay harness)
    and fold naively in ARRIVAL order, and the deal ends at 4 instead,
    because arrival-order folding applies seq 2's amend last."""
    new = d("DEAL-1", 1, "new", 10.0, trade_price=70.0)
    amend_to_4 = d("DEAL-1", 2, "amend", 4.0)
    amend_to_7 = d("DEAL-1", 3, "amend", 7.0)

    def naive_arrival_order_fold(deals_in_arrival_order):
        deal_volume: dict[str, float] = {}
        for msg in deals_in_arrival_order:
            cingest._fold_last_write_wins(deal_volume, msg.deal_id, msg.action, msg.volume)
        return sum(deal_volume.values())

    correct_final = naive_arrival_order_fold([new, amend_to_4, amend_to_7])
    assert correct_final == 7.0

    out_of_order_final = naive_arrival_order_fold([new, amend_to_7, amend_to_4])
    assert out_of_order_final == 4.0
    assert out_of_order_final != correct_final

    # The production rebuild_positions always sorts by sequence first, so
    # it gives the correct, order-independent answer for both deliveries.
    conn_a = db.connect(":memory:")
    cingest.ingest_stream(conn_a, [new, amend_to_4, amend_to_7])
    cingest.rebuild_positions(conn_a)

    conn_b = db.connect(":memory:")
    cingest.ingest_stream(conn_b, [new, amend_to_7, amend_to_4])
    cingest.rebuild_positions(conn_b)

    assert cingest.final_positions(conn_a) == cingest.final_positions(conn_b) == {"WTI-320D": 7.0}
