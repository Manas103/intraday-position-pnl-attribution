#!/usr/bin/env python
"""Build the canonical fill stream, perturb it (5% duplicated, bounded
out-of-order), ingest both, and check the resulting positions are
bit-identical. This is the exact script referenced in the README.

Usage:
    python scripts/run_replay_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import db, ingest, replay, simulator


def main():
    seed = 42
    universe = simulator.build_universe(seed, n_instruments=8)
    canonical = simulator.build_canonical_fill_stream(seed, universe, n_sessions=250)
    perturbed = replay.perturb_stream(seed, canonical, dup_fraction=0.05, window=10)

    dup_frac = replay.measured_duplicate_fraction(canonical, perturbed)
    print(f"canonical stream length: {len(canonical)}")
    print(f"perturbed stream length: {len(perturbed)}")
    print(f"measured duplicate fraction: {dup_frac * 100:.4f}%")

    conn_canonical = db.connect(":memory:")
    stats_canonical = ingest.ingest_stream(conn_canonical, canonical)
    ingest.rebuild_positions(conn_canonical)
    positions_canonical = ingest.final_positions(conn_canonical)

    conn_perturbed = db.connect(":memory:")
    stats_perturbed = ingest.ingest_stream(conn_perturbed, perturbed)
    ingest.rebuild_positions(conn_perturbed)
    positions_perturbed = ingest.final_positions(conn_perturbed)

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
    print(f"claim: positions rebuilt bit-identically from a noisy replay -> {'MEETS' if identical else 'DOES NOT MEET'}")


if __name__ == "__main__":
    main()
