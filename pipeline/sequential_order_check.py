"""Demonstrates why a sequential ("waterfall") bump-one-factor-at-a-time
attribution hands the spot/vol interaction to whichever factor is repriced
second, while this repository's simultaneous attribution (attribution.py,
cpp/src/attribution_main.cpp) does not.

Method: for each (instrument, session), the exact Black-Scholes reference
oracle (the same C++ engine used everywhere else in this repository, not a
separate implementation) is called three extra times per row, to price the
same contract at three intermediate market states, holding T fixed at T0 so
the comparison isolates the spot/vol interaction specifically:
    price0          = price(S0, sigma0)
    price_S_first    = price(S1, sigma0)   -- spot moved, vol not yet
    price_vol_first  = price(S0, sigma1)   -- vol moved, spot not yet
    price_final      = price(S1, sigma1)   -- both moved (T still T0)

Two sequential orderings bucket the same total move (price_final - price0)
differently:
    Order A (spot first):  spot_bucket_A = price_S_first - price0
                            vol_bucket_A  = price_final - price_S_first
    Order B (vol first):   vol_bucket_B  = price_vol_first - price0
                            spot_bucket_B = price_final - price_vol_first

vol_bucket_A - vol_bucket_B (equivalently spot_bucket_B - spot_bucket_A) is
the exact mixed second difference, a finite-difference stand-in for the
vanna cross term: it is nonzero whenever spot and vol both move, and it is
assigned entirely to whichever bucket was computed second, flipping sign
and target bucket depending on the arbitrary choice of order. This
repository's simultaneous attribution has no such choice: vanna_pnl is one
number, computed once, from the SOD Greeks.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cpp_bridge


@dataclass(frozen=True)
class OrderDependenceRow:
    instrument: str
    session_id: int
    spot_bucket_a: float
    vol_bucket_a: float
    spot_bucket_b: float
    vol_bucket_b: float
    bucket_discrepancy: float  # vol_bucket_a - vol_bucket_b
    vanna_pnl: float  # this repo's simultaneous cross-term bucket, for comparison


def compare_sequential_orders(sessions, binary_path=None) -> list[OrderDependenceRow]:
    """sessions: iterable of (instrument_id, session_id, S0, K, T0, r, sigma0,
    option_type, S1, sigma1, vanna_pnl), where vanna_pnl is already computed
    by the simultaneous engine (attribution.py) for the same row, passed in
    for side-by-side comparison rather than recomputed here.
    """
    requests = []
    meta = []
    for instrument, session_id, S0, K, T0, r, sigma0, opt_type, S1, sigma1, vanna_pnl in sessions:
        req_id = f"{instrument}:{session_id}"
        # Reuse the exact-reprice engine as a pure pricer: price1/price0 of
        # each synthetic request is an exact Black-Scholes price at whatever
        # (S, sigma, T) is given, regardless of what the request's own
        # Taylor-decomposition fields mean for that row.
        requests.append((f"{req_id}:S_first", S0, K, T0, r, sigma0, opt_type, S1, sigma0, T0))
        requests.append((f"{req_id}:vol_first", S0, K, T0, r, sigma0, opt_type, S0, sigma1, T0))
        meta.append((instrument, session_id, vanna_pnl))

    results = cpp_bridge.run_attribution_batch(requests, binary_path)
    # Each meta entry consumed exactly 2 results (S_first, vol_first).
    rows: list[OrderDependenceRow] = []
    for i, (instrument, session_id, vanna_pnl) in enumerate(meta):
        r_s_first = results[2 * i]
        r_vol_first = results[2 * i + 1]
        price0 = r_s_first.price_sod
        price_s_first = r_s_first.price_eod
        price_vol_first = r_vol_first.price_eod
        # price_final needs both moves applied; recover it as
        # price0 + (price_s_first - price0) + (price_vol_first - price0) is
        # WRONG in general (that is linear superposition, exactly the thing
        # a Taylor model assumes); instead issue one more honest request.
        rows.append((instrument, session_id, price0, price_s_first, price_vol_first, vanna_pnl))

    # Second pass: the true price_final, one request per row, reusing the
    # same S1/sigma1 pair but this time as the actual target state.
    final_requests = []
    for instrument, session_id, S0, K, T0, r, sigma0, opt_type, S1, sigma1, _ in sessions:
        final_requests.append((f"{instrument}:{session_id}:final", S0, K, T0, r, sigma0, opt_type, S1, sigma1, T0))
    final_results = cpp_bridge.run_attribution_batch(final_requests, binary_path)

    out: list[OrderDependenceRow] = []
    for (instrument, session_id, price0, price_s_first, price_vol_first, vanna_pnl), r_final in zip(
        rows, final_results
    ):
        price_final = r_final.price_eod
        spot_bucket_a = price_s_first - price0
        vol_bucket_a = price_final - price_s_first
        vol_bucket_b = price_vol_first - price0
        spot_bucket_b = price_final - price_vol_first
        discrepancy = vol_bucket_a - vol_bucket_b
        out.append(
            OrderDependenceRow(
                instrument=instrument,
                session_id=session_id,
                spot_bucket_a=spot_bucket_a,
                vol_bucket_a=vol_bucket_a,
                spot_bucket_b=spot_bucket_b,
                vol_bucket_b=vol_bucket_b,
                bucket_discrepancy=discrepancy,
                vanna_pnl=vanna_pnl,
            )
        )
    return out
