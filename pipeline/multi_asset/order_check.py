"""Demonstrates that a sequential ("waterfall") price-then-FX or
FX-then-price bump attribution hands the price/FX interaction to whichever
factor is repriced second, while this repository's simultaneous multi-asset
attribution (attribution.py) does not.

For one instrument-session, let dP = price_eod - price_sod (local currency)
and dFX = fx_eod - fx_sod. The two sequential orderings bucket the same
total USD move differently:

    Order A (price first):
        price_bucket_A = qty * dP * fx_sod
        fx_bucket_A    = qty * (price_sod + dP) * dFX
                        = qty * price_sod * dFX + qty * dP * dFX

    Order B (FX first):
        fx_bucket_B    = qty * price_sod * dFX
        price_bucket_B = qty * (fx_sod + dFX) * dP
                        = qty * fx_sod * dP + qty * dP * dFX

fx_bucket_A - fx_bucket_B (equivalently price_bucket_B - price_bucket_A) is
exactly qty * dP * dFX, the mixed cross term: nonzero whenever both price
and FX move in the same instrument-session, and assigned entirely to
whichever factor was bumped second, flipping sign and target bucket
depending on an arbitrary modeling choice. attribution.py has no such
choice: the fx_pnl and price_pnl/rate_pnl buckets are each one number,
computed once, from the same start-of-day state, and the cross term is
reported as residual rather than silently folded into whichever bucket
happened to go last.

USD-denominated instruments have dFX == 0 identically every session, so
both orderings trivially agree on them; this is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import simulator as masim


@dataclass(frozen=True)
class OrderDependenceRow:
    instrument_id: str
    session_id: int
    currency: str
    fx_bucket_a: float
    fx_bucket_b: float
    discrepancy: float  # fx_bucket_a - fx_bucket_b == qty * dP * dFX


def compare_sequential_orders(seed: int = 42, n_sessions: int = 250) -> list[OrderDependenceRow]:
    universe = masim.build_universe(seed)
    rows: list[OrderDependenceRow] = []

    for idx, inst in enumerate(universe):
        sessions = masim.build_sessions(seed, idx, inst, n_sessions)
        for st in sessions:
            d_price = st.price_eod - st.price_sod
            d_fx = st.fx_eod - st.fx_sod
            qty = inst.qty

            fx_bucket_a = qty * st.price_sod * d_fx + qty * d_price * d_fx
            fx_bucket_b = qty * st.price_sod * d_fx
            discrepancy = fx_bucket_a - fx_bucket_b

            rows.append(OrderDependenceRow(
                instrument_id=inst.instrument_id, session_id=st.session_id, currency=inst.currency,
                fx_bucket_a=fx_bucket_a, fx_bucket_b=fx_bucket_b, discrepancy=discrepancy,
            ))

    return rows


def summarize(rows: list[OrderDependenceRow]) -> dict:
    n = len(rows)
    disagree = sum(1 for r in rows if r.discrepancy != 0.0)
    mean_abs_discrepancy = sum(abs(r.discrepancy) for r in rows) / n if n else 0.0
    usd_rows = [r for r in rows if r.currency == "USD"]
    foreign_rows = [r for r in rows if r.currency != "USD"]
    return {
        "n_rows": n,
        "n_disagree": disagree,
        "pct_disagree": 100.0 * disagree / n if n else 0.0,
        "mean_abs_discrepancy": mean_abs_discrepancy,
        "n_usd_rows": len(usd_rows),
        "n_usd_disagree": sum(1 for r in usd_rows if r.discrepancy != 0.0),
        "n_foreign_rows": len(foreign_rows),
        "n_foreign_disagree": sum(1 for r in foreign_rows if r.discrepancy != 0.0),
    }
