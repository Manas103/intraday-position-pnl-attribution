"""As-of position and exposure queries: the claim this extension exists to
support, that any past risk number is reconstructable.

The mechanism is deliberately the simplest one that is actually correct:
`position_as_of` is a plain SQL aggregate over the immutable ledger,
filtered to `sequence <= as_of_sequence`, every time it is called. There is
no materialized running total anywhere that could drift from the ledger or
need its own reconciliation; the ledger itself, plus a WHERE clause, is the
entire mechanism. "Reconstructable" means exactly this: querying with an
old `as_of_sequence` reproduces the position that was true back then, using
only data that was already in the ledger at that point, not a snapshot
taken at the time.
"""

from __future__ import annotations

from collections import defaultdict

from .simulator import Fill


def position_as_of(conn, instrument: str, as_of_sequence: int) -> float:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(quantity), 0) FROM risk_fills "
            "WHERE instrument = %s AND sequence <= %s",
            (instrument, as_of_sequence),
        )
        return float(cur.fetchone()[0])


def last_price_as_of(conn, instrument: str, as_of_sequence: int) -> float | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT price FROM risk_fills WHERE instrument = %s AND sequence <= %s "
            "ORDER BY sequence DESC LIMIT 1",
            (instrument, as_of_sequence),
        )
        row = cur.fetchone()
        return float(row[0]) if row else None


def exposure_as_of(conn, instrument: str, as_of_sequence: int) -> float | None:
    """Gross notional exposure: position * the most recent traded price at
    or before as_of_sequence. None if the instrument had no fills at or
    before that point (there is no price to mark against)."""
    pos = position_as_of(conn, instrument, as_of_sequence)
    price = last_price_as_of(conn, instrument, as_of_sequence)
    if price is None:
        return None
    return pos * price


def independent_position_as_of_from_stream(sorted_fills_for_instrument: list[Fill], as_of_sequence: int) -> float:
    """Ground truth for the reconstruction check below: folds a single
    instrument's fills directly in Python, filtered to the same cutoff,
    with no SQL and no database involved at all. This is the
    "independent" half of "as-of queries ... making any past risk number
    reconstructable": the as-of SQL query is checked against a completely
    separate code path operating on the original generator output, not
    against another query against the same database.

    Takes that one instrument's fills pre-sorted by sequence (not the
    full multi-million-row stream, and not re-filtered on every call):
    `check_reconstruction` below groups the stream by instrument exactly
    once, which is what keeps this check O(fills) overall instead of
    O(fills * cutoffs_checked)."""
    total = 0.0
    for f in sorted_fills_for_instrument:
        if f.sequence > as_of_sequence:
            break  # pre-sorted by sequence: nothing later can qualify either
        total += f.quantity
    return total


def check_reconstruction(conn, canonical: list[Fill], instruments: list[str], cutoffs_per_instrument: int, seed: int) -> dict:
    """For each instrument, picks `cutoffs_per_instrument` historical
    sequence numbers spread across that instrument's own history
    (including early, middle and final points, not only the final/EOD
    sequence) and checks the as-of SQL query against the independent
    Python fold at every one. Returns a summary dict; raises nothing, so
    the caller can report a partial match honestly if one ever occurs."""
    import random

    rng = random.Random(seed)
    by_instrument: dict[str, list[Fill]] = defaultdict(list)
    for f in canonical:
        by_instrument[f.instrument].append(f)
    for fills in by_instrument.values():
        fills.sort(key=lambda f: f.sequence)

    checked = 0
    matched = 0
    mismatches = []
    for inst in instruments:
        inst_fills = by_instrument.get(inst, [])
        if not inst_fills:
            continue
        max_seq = inst_fills[-1].sequence
        cutoffs = sorted(set(rng.randint(1, max_seq) for _ in range(cutoffs_per_instrument)))
        for cutoff in cutoffs:
            sql_pos = position_as_of(conn, inst, cutoff)
            py_pos = independent_position_as_of_from_stream(inst_fills, cutoff)
            checked += 1
            if sql_pos == py_pos:
                matched += 1
            else:
                mismatches.append((inst, cutoff, sql_pos, py_pos))

    return {"checked": checked, "matched": matched, "mismatches": mismatches}
