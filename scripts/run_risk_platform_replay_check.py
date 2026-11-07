"""Measures: positions bit-identical under a replay carrying 5% duplicated
and out-of-order messages, at the risk platform's own scale, into
PostgreSQL rather than SQLite. Ingests the canonical stream into the
ledger, records the final position per instrument, truncates, ingests the
perturbed stream, and compares.

Usage: python3 scripts/run_risk_platform_replay_check.py [n_instruments] [n_sessions]
"""
import sys

sys.path.insert(0, ".")
from pipeline.risk_platform import pg_ingest, simulator

N_INSTRUMENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 420
N_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
SEED = 42


def final_positions(conn, instruments):
    out = {}
    with conn.cursor() as cur:
        for inst in instruments:
            cur.execute("SELECT COALESCE(SUM(quantity), 0) FROM risk_fills WHERE instrument = %s", (inst,))
            out[inst] = float(cur.fetchone()[0])
    return out


def main():
    conn = pg_ingest.connect()
    pg_ingest.ensure_schema(conn)
    instruments = simulator.instrument_ids(N_INSTRUMENTS)

    canonical = list(simulator.generate_canonical_stream(SEED, N_INSTRUMENTS, N_SESSIONS))
    n_canonical = len(canonical)
    perturbed = simulator.perturb_stream(SEED, canonical)
    dup_fraction = simulator.measured_duplicate_fraction(n_canonical, perturbed)

    pg_ingest.reset_tables(conn)
    accepted_canonical = pg_ingest.ingest_stream(conn, canonical)
    positions_canonical = final_positions(conn, instruments)

    pg_ingest.reset_tables(conn)
    accepted_perturbed = pg_ingest.ingest_stream(conn, perturbed)
    positions_perturbed = final_positions(conn, instruments)

    identical = positions_canonical == positions_perturbed
    mismatches = [] if identical else [
        inst for inst in instruments if positions_canonical[inst] != positions_perturbed.get(inst)
    ]

    print(f"instruments={N_INSTRUMENTS} sessions={N_SESSIONS}")
    print(f"canonical stream: {n_canonical} messages")
    print(f"perturbed stream: {len(perturbed)} messages, measured duplicate fraction {dup_fraction * 100:.4f}%")
    print(f"accepted (canonical ingest): {accepted_canonical}")
    print(f"accepted (perturbed ingest): {accepted_perturbed}")
    print(f"distinct-fill counts match: {accepted_canonical == accepted_perturbed}")
    print(f"final positions bit-identical across all {len(instruments)} instruments: {identical}")
    if mismatches:
        print(f"  mismatched instruments: {mismatches[:20]}")

    conn.close()


if __name__ == "__main__":
    main()
