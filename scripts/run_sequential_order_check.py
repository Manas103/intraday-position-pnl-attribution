#!/usr/bin/env python
"""Demonstrates the claim 'interactions are attributed to their own
bucket, not to whichever factor was repriced second': builds the same 250
sessions x 8 instruments used everywhere else in this repository, and for
each row compares a naive sequential (spot-first vs. vol-first) bump
attribution against this repository's simultaneous vanna_pnl bucket (read
directly from the main engine's own output for the same row).

Usage:
    python scripts/run_sequential_order_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import cpp_bridge, simulator
from pipeline.sequential_order_check import compare_sequential_orders


def main():
    seed = 42
    n_sessions = 250
    n_instruments = 8
    universe = simulator.build_universe(seed, n_instruments)

    full_requests = []
    meta = []
    for idx, instrument in enumerate(universe):
        for s in simulator.build_sessions(seed, idx, instrument, n_sessions):
            req_id = f"{instrument.instrument_id}:{s.session_id}"
            full_requests.append(
                (req_id, s.s0, instrument.strike, s.t0, simulator.RATE, s.sigma0,
                 instrument.option_type, s.s1, s.sigma1, s.t1)
            )
            meta.append((instrument.instrument_id, s.session_id, s.s0, instrument.strike, s.t0,
                         simulator.RATE, s.sigma0, instrument.option_type, s.s1, s.sigma1))

    full_results = cpp_bridge.run_attribution_batch(full_requests)

    sessions = [
        m + (res.vanna_pnl,)
        for m, res in zip(meta, full_results)
    ]

    rows = compare_sequential_orders(sessions)

    n_nonzero_discrepancy = sum(1 for r in rows if abs(r.bucket_discrepancy) > 1e-12)
    abs_discrepancies = [abs(r.bucket_discrepancy) for r in rows]
    abs_vanna = [abs(r.vanna_pnl) for r in rows]
    mean_discrepancy = sum(abs_discrepancies) / len(abs_discrepancies)
    mean_vanna = sum(abs_vanna) / len(abs_vanna)

    print(f"rows checked: {len(rows)}")
    print(f"rows where spot-first and vol-first disagree on the vol bucket: {n_nonzero_discrepancy} "
          f"({100 * n_nonzero_discrepancy / len(rows):.2f}%)")
    print(f"mean |vol_bucket_a - vol_bucket_b| (the ambiguous sequential interaction): {mean_discrepancy:.8f}")
    print(f"mean |vanna_pnl| (this repo's unambiguous simultaneous cross term): {mean_vanna:.8f}")
    print()
    print("first 5 rows:")
    for r in rows[:5]:
        print(
            f"  {r.instrument}:{r.session_id}  spot_first(vol_bucket={r.vol_bucket_a:.6f})  "
            f"vol_first(vol_bucket={r.vol_bucket_b:.6f})  discrepancy={r.bucket_discrepancy:.6f}  "
            f"vanna_pnl={r.vanna_pnl:.6f}"
        )


if __name__ == "__main__":
    main()
