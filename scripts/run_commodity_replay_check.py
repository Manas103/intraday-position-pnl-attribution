#!/usr/bin/env python
"""Build the canonical commodity deal blotter, perturb it (5% duplicated,
bounded out-of-order), ingest both, and check the resulting positions are
bit-identical. This is the exact script referenced in the README's
"Extension" section.

Usage:
    python scripts/run_commodity_replay_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import commodity_ingest as cingest
from pipeline import commodity_replay as creplay
from pipeline import commodity_simulator as csim
from pipeline import db


def main():
    seed = 42
    n_sessions = 250
    universe = csim.build_commodity_universe()
    commodities = sorted({inst.commodity for inst in universe})
    curve_sessions = {c: csim.build_curve_sessions(seed, c, n_sessions) for c in commodities}
    canonical = csim.build_deal_blotter(seed, universe, curve_sessions, n_sessions)
    perturbed = creplay.perturb_stream(seed, canonical, dup_fraction=0.05, window=10)

    dup_frac = creplay.measured_duplicate_fraction(canonical, perturbed)
    print(f"canonical stream length: {len(canonical)}")
    print(f"perturbed stream length: {len(perturbed)}")
    print(f"measured duplicate fraction: {dup_frac * 100:.4f}%")

    conn_canonical = db.connect(":memory:")
    stats_canonical = cingest.ingest_stream(conn_canonical, canonical)
    cingest.rebuild_positions(conn_canonical)
    positions_canonical = cingest.final_positions(conn_canonical)

    conn_perturbed = db.connect(":memory:")
    stats_perturbed = cingest.ingest_stream(conn_perturbed, perturbed)
    cingest.rebuild_positions(conn_perturbed)
    positions_perturbed = cingest.final_positions(conn_perturbed)

    print(f"canonical ingest stats: {stats_canonical}")
    print(f"perturbed ingest stats: {stats_perturbed}")

    identical = positions_canonical == positions_perturbed
    print(f"accepted-message-count match: {stats_canonical['accepted'] == stats_perturbed['accepted']}")
    print(f"positions bit-identical (exact ==): {identical}")

    if not identical:
        for instrument in sorted(set(positions_canonical) | set(positions_perturbed)):
            c = positions_canonical.get(instrument)
            p = positions_perturbed.get(instrument)
            if c != p:
                print(f"  DIVERGENCE {instrument}: canonical={c} perturbed={p}")

    print()
    print(f"claim: bit-identical book rebuilt from a 5% duplicated, out-of-order replay -> "
          f"{'MEETS' if identical else 'DOES NOT MEET'}")


if __name__ == "__main__":
    main()
