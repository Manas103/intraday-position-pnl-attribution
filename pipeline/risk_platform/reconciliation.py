"""End-of-day reconciliation against an independent replay.

Two genuinely different computations of the same thing, end-of-day
position per instrument per session:

  - "live": one SQL query per (instrument, session) against the
    PostgreSQL ledger, using the same `position_as_of` aggregate the
    as-of module above exposes, evaluated at that instrument's own
    last sequence number at or before that session.
  - "independent replay": a vectorized pandas fold over the canonical
    in-memory fill stream (groupby + cumsum, not a SQL query and not a
    per-row Python loop), which never touches the database at all.

When the ledger and the independent replay are built from the same fill
stream, their EOD grids must match exactly: this is what "reconciliation"
checks. `inject_seeded_breaks` then corrupts a fixed number of cells in the
independent-replay side to prove the comparison actually catches a
divergence rather than passing by construction.
"""

from __future__ import annotations

import random

import pandas as pd

from .simulator import Fill


def live_eod_grid(conn, instruments: list[str], n_sessions: int) -> pd.DataFrame:
    """One row per (instrument, session_id), via the live SQL as-of query
    evaluated at each instrument's own last sequence number at or before
    that session -- computed inside a single SQL statement per instrument
    (not n_sessions separate round trips) for tractable runtime at scale."""
    rows = []
    with conn.cursor() as cur:
        for inst in instruments:
            cur.execute(
                """
                WITH session_cutoffs AS (
                    SELECT s AS session_id,
                           (SELECT MAX(sequence) FROM risk_fills
                            WHERE instrument = %(inst)s AND session_id <= s) AS cutoff_seq
                    FROM generate_series(0, %(max_session)s) AS s
                )
                SELECT session_id,
                       COALESCE((SELECT SUM(quantity) FROM risk_fills
                                 WHERE instrument = %(inst)s AND sequence <= session_cutoffs.cutoff_seq), 0)
                FROM session_cutoffs
                ORDER BY session_id
                """,
                {"inst": inst, "max_session": n_sessions - 1},
            )
            for session_id, position in cur.fetchall():
                rows.append((inst, session_id, float(position)))
    return pd.DataFrame(rows, columns=["instrument", "session_id", "position_live"])


def independent_eod_grid(canonical: list[Fill]) -> pd.DataFrame:
    """Vectorized pandas fold: cumulative signed quantity per instrument in
    sequence order, then the last cumulative value at or before each
    session boundary. No SQL, no database connection, no per-row Python
    loop -- a genuinely different computational method from the live
    grid above, not a second copy of the same query."""
    df = pd.DataFrame(canonical, columns=["fill_id", "sequence", "instrument", "quantity", "price", "session_id"])
    df = df.sort_values(["instrument", "sequence"])
    df["cum_qty"] = df.groupby("instrument")["quantity"].cumsum()
    # last cumulative quantity observed at or before each session, per instrument
    eod = df.groupby(["instrument", "session_id"], as_index=False)["cum_qty"].last()
    eod = eod.sort_values(["instrument", "session_id"])
    eod["position_replay"] = eod.groupby("instrument")["cum_qty"].ffill()
    return eod[["instrument", "session_id", "position_replay"]]


def full_eod_grid(independent: pd.DataFrame, instruments: list[str], n_sessions: int) -> pd.DataFrame:
    """Forward-fills every (instrument, session) combination, since an
    instrument with no fills in a given session still has an EOD position
    (unchanged from the prior session) that both sides must agree on."""
    full_index = pd.MultiIndex.from_product([instruments, range(n_sessions)], names=["instrument", "session_id"])
    full = independent.set_index(["instrument", "session_id"]).reindex(full_index)
    full["position_replay"] = full.groupby("instrument")["position_replay"].ffill().fillna(0.0)
    return full.reset_index()


def inject_seeded_breaks(grid: pd.DataFrame, n_breaks: int, seed: int) -> tuple[pd.DataFrame, set[tuple[str, int]]]:
    """Corrupts exactly n_breaks (instrument, session_id) cells of
    `grid['position_replay']` by adding a nonzero random delta, simulating
    an independently-maintained EOD snapshot that occasionally diverges
    from the live ledger (a missed update, a double-counted amendment).
    Returns the corrupted grid and the set of (instrument, session_id)
    keys that were actually corrupted, which is the ground truth the
    detection rate is scored against."""
    rng = random.Random(seed)
    grid = grid.copy()
    n = len(grid)
    if n_breaks > n:
        raise ValueError(f"cannot inject {n_breaks} breaks into a grid of {n} rows")
    break_positions = rng.sample(range(n), n_breaks)
    broken_keys = set()
    for pos in break_positions:
        delta = rng.choice([-1, 1]) * rng.randint(1, 500)
        row = grid.iloc[pos]
        grid.iloc[pos, grid.columns.get_loc("position_replay")] = row["position_replay"] + delta
        broken_keys.add((row["instrument"], int(row["session_id"])))
    return grid, broken_keys


def reconcile(live: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    merged = live.merge(replay, on=["instrument", "session_id"], how="outer")
    merged["break"] = merged["position_live"] != merged["position_replay"]
    return merged
