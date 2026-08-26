"""Idempotent deal ingest and deterministic forward-book position rebuild.

Mirrors `pipeline/ingest.py`'s two-part idempotency argument exactly, for a
different, order-dependence bug shape:

1. Idempotency at the storage layer: `commodity_deals` has a UNIQUE (in
   fact PRIMARY KEY) constraint on (deal_id, sequence). A deal_id is NOT
   unique on its own (an amend or cancel of the same economic deal reuses
   the original deal_id with a new, later sequence number); the composite
   key is what makes replaying the same message a no-op while still
   allowing legitimate follow-up messages for the same deal.

2. Order-independent position rebuild: positions are never accumulated
   incrementally as messages arrive. `rebuild_positions` always re-reads
   every accepted deal for an instrument sorted by `sequence` ascending.

Here the order-dependence is a "last write wins" bug, not an average-cost
fold bug: an "amend" carries the deal's new ABSOLUTE volume (not a delta).
Folding two amends of the same deal_id in the wrong order silently keeps
the wrong one. Concretely: amend-to-4 at sequence 2, amend-to-7 at sequence
3. Sequence order says the deal ends at 7 (3 is later). Deliver 3 before 2
(a perfectly plausible reordering under the replay harness's bounded-window
shuffle) and fold in ARRIVAL order instead, and the deal ends at 4, because
arrival-order folding applies sequence-2's amend last. Idempotency alone
(the UNIQUE constraint) does not catch this: both deliveries accept the
same *set* of (deal_id, sequence) messages, but naively folding that set in
arrival order gives a different, wrong answer than folding it in sequence
order. `tests/test_commodity_ingest.py::test_naive_arrival_order_fold_diverges_from_sequence_sorted_fold`
reproduces both folds side by side.
"""

from __future__ import annotations

import sqlite3

from .commodity_simulator import Deal


def ingest_deal(conn: sqlite3.Connection, deal: Deal, received_at: int) -> bool:
    """Insert one deal message. Returns True if newly accepted, False if
    this (deal_id, sequence) pair was already present (idempotent no-op)."""
    try:
        conn.execute(
            "INSERT INTO commodity_deals "
            "(deal_id, sequence, action, commodity, delivery_tenor_days, volume, trade_price, "
            " direction, session_id, timestamp, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (deal.deal_id, deal.sequence, deal.action, deal.commodity, deal.delivery_tenor_days,
             deal.volume, deal.trade_price, deal.direction, deal.session_id, deal.timestamp, received_at),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def ingest_stream(conn: sqlite3.Connection, deals: list[Deal]) -> dict:
    """Ingest a stream of deals (possibly containing duplicates and
    out-of-order deliveries) and commit. Returns counts for diagnostics."""
    accepted = 0
    duplicates = 0
    for i, deal in enumerate(deals):
        if ingest_deal(conn, deal, i):
            accepted += 1
        else:
            duplicates += 1
    conn.commit()
    return {"total_messages": len(deals), "accepted": accepted, "duplicates": duplicates}


def _fold_last_write_wins(deal_volume: dict[str, float], deal_id: str, action: str, volume: float) -> dict[str, float]:
    """Apply one deal message to a running {deal_id: current_volume} state,
    in place, and return it. "new" and "amend" set the deal's volume to the
    message's (absolute) volume; "cancel" sets it to zero. Whichever
    message is applied LAST wins, which is exactly why the caller must
    apply messages in sequence order, not arrival order."""
    deal_volume[deal_id] = 0.0 if action == "cancel" else volume
    return deal_volume


def rebuild_positions(conn: sqlite3.Connection) -> None:
    """Recompute the entire `commodity_positions` table from
    `commodity_deals`, folding each instrument's accepted deals strictly in
    ascending `sequence` order, and filling the position forward across any
    session with no activity for that instrument (a real position-keeping
    ledger carries yesterday's position forward unchanged)."""
    conn.execute("DELETE FROM commodity_positions")
    instruments = conn.execute(
        "SELECT DISTINCT commodity, delivery_tenor_days FROM commodity_deals"
    ).fetchall()

    for commodity, tenor in instruments:
        instrument_id = f"{commodity}-{tenor}D"
        rows = conn.execute(
            "SELECT deal_id, sequence, action, volume, session_id FROM commodity_deals "
            "WHERE commodity = ? AND delivery_tenor_days = ? ORDER BY sequence ASC",
            (commodity, tenor),
        ).fetchall()

        deal_volume: dict[str, float] = {}
        session_snapshot: dict[int, float] = {}
        for deal_id, sequence, action, volume, session_id in rows:
            _fold_last_write_wins(deal_volume, deal_id, action, volume)
            session_snapshot[session_id] = sum(deal_volume.values())

        if not session_snapshot:
            continue
        max_session = max(session_snapshot)
        running = 0.0
        for s in range(max_session + 1):
            if s in session_snapshot:
                running = session_snapshot[s]
            conn.execute(
                "INSERT INTO commodity_positions (instrument_id, session_id, quantity) VALUES (?, ?, ?)",
                (instrument_id, s, running),
            )
    conn.commit()


def final_positions(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        "SELECT instrument_id, quantity FROM commodity_positions p1 "
        "WHERE session_id = (SELECT MAX(session_id) FROM commodity_positions p2 "
        "                    WHERE p2.instrument_id = p1.instrument_id) "
        "ORDER BY instrument_id"
    ).fetchall()
    return {instrument_id: quantity for instrument_id, quantity in rows}
