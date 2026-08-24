"""Idempotent fill ingest and deterministic position rebuild.

Two separate ideas, both required for the "bit-identical under duplication
and reordering" claim:

1. Idempotency at the storage layer: the `fills` table has a UNIQUE (in
   fact PRIMARY KEY) constraint on (fill_id, sequence). Inserting the same
   pair twice raises sqlite3.IntegrityError, which we catch and treat as a
   no-op. This makes ingest safe to call with the same message any number
   of times, in any position in the stream.

2. Order-independent position rebuild: positions are NOT accumulated
   incrementally as messages arrive. They are recomputed by reading back
   every accepted fill for an instrument, sorted by `sequence` ascending,
   and folding them in that order. Because step 1 already guarantees the
   *set* of accepted (fill_id, sequence) pairs is independent of arrival
   order and duplication, and step 2 always folds that set in the same
   (sequence-sorted) order, the final position is bit-identical regardless
   of how the messages were delivered.

The first version of this module folded fills as they arrived (i.e. in
insertion/arrival order) and only deduplicated on fill_id. That passed the
"exact duplicate" test but silently produced a different average cost basis
under reordering, because weighted-average-cost is order-dependent: buying
10 then selling 4 gives a different intermediate (and, after a sign flip,
final) average price than selling 4 then buying 10 if the two ever land in
different relative order. See README Findings for the measurement that
caught this.
"""

from __future__ import annotations

import sqlite3

from .simulator import Fill


def ingest_fill(conn: sqlite3.Connection, fill: Fill, received_at: int) -> bool:
    """Insert one fill. Returns True if newly accepted, False if this
    (fill_id, sequence) pair was already present (idempotent no-op)."""
    try:
        conn.execute(
            "INSERT INTO fills (fill_id, sequence, instrument, quantity, price, session_id, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fill.fill_id, fill.sequence, fill.instrument, fill.quantity, fill.price, fill.session_id, received_at),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def ingest_stream(conn: sqlite3.Connection, fills: list[Fill]) -> dict:
    """Ingest a stream of fills (possibly containing duplicates and
    out-of-order deliveries) and commit. Returns counts for diagnostics."""
    accepted = 0
    duplicates = 0
    for i, fill in enumerate(fills):
        if ingest_fill(conn, fill, i):
            accepted += 1
        else:
            duplicates += 1
    conn.commit()
    return {"total_messages": len(fills), "accepted": accepted, "duplicates": duplicates}


def _fold(qty: float, avg_price: float, fill_qty: float, fill_price: float) -> tuple[float, float]:
    new_qty = qty + fill_qty
    same_direction_or_flat = qty == 0.0 or (qty > 0) == (fill_qty > 0)
    if same_direction_or_flat:
        if new_qty == 0.0:
            return 0.0, 0.0
        return new_qty, (qty * avg_price + fill_qty * fill_price) / new_qty
    # Reducing an existing position.
    flipped = (qty > 0 and new_qty < 0) or (qty < 0 and new_qty > 0)
    if flipped:
        # The excess beyond flat opens a fresh position at the fill price.
        return new_qty, fill_price
    if new_qty == 0.0:
        return 0.0, 0.0
    # Partial reduction: remaining quantity keeps the existing average price.
    return new_qty, avg_price


def rebuild_positions(conn: sqlite3.Connection) -> None:
    """Recompute the entire `positions` table from `fills`, folding each
    instrument's accepted fills strictly in ascending `sequence` order."""
    conn.execute("DELETE FROM positions")
    instruments = [row[0] for row in conn.execute("SELECT DISTINCT instrument FROM fills").fetchall()]
    for instrument in instruments:
        rows = conn.execute(
            "SELECT sequence, quantity, price, session_id FROM fills WHERE instrument = ? ORDER BY sequence ASC",
            (instrument,),
        ).fetchall()
        qty, avg_price, last_session = 0.0, 0.0, None
        for sequence, fill_qty, fill_price, session_id in rows:
            qty, avg_price = _fold(qty, avg_price, fill_qty, fill_price)
            last_session = session_id
        if last_session is not None:
            conn.execute(
                "INSERT INTO positions (instrument, session_id, quantity, avg_price) VALUES (?, ?, ?, ?)",
                (instrument, last_session, qty, avg_price),
            )
    conn.commit()


def final_positions(conn: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    rows = conn.execute("SELECT instrument, quantity, avg_price FROM positions ORDER BY instrument").fetchall()
    return {instrument: (qty, avg_price) for instrument, qty, avg_price in rows}
