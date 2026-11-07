"""Measures: as-of position queries reconstruct historical risk numbers
exactly. Ingests the canonical stream, then for every instrument picks
several historical sequence cutoffs (not just the final one) and checks
the live SQL as-of query against an independent Python fold over the
same in-memory stream.

Usage: python3 scripts/run_risk_platform_asof_check.py [n_instruments] [n_sessions] [cutoffs_per_instrument]
"""
import sys

sys.path.insert(0, ".")
from pipeline.risk_platform import asof, pg_ingest, simulator

N_INSTRUMENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 420
N_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
CUTOFFS_PER_INSTRUMENT = int(sys.argv[3]) if len(sys.argv) > 3 else 20
SEED = 42


def main():
    conn = pg_ingest.connect()
    pg_ingest.ensure_schema(conn)
    pg_ingest.reset_tables(conn)

    canonical = list(simulator.generate_canonical_stream(SEED, N_INSTRUMENTS, N_SESSIONS))
    pg_ingest.ingest_stream(conn, canonical)

    instruments = simulator.instrument_ids(N_INSTRUMENTS)
    result = asof.check_reconstruction(conn, canonical, instruments, CUTOFFS_PER_INSTRUMENT, seed=SEED + 1)

    print(f"instruments={N_INSTRUMENTS} sessions={N_SESSIONS} cutoffs_per_instrument={CUTOFFS_PER_INSTRUMENT}")
    print(f"historical as-of cutoffs checked: {result['checked']}")
    print(f"matched independent Python fold exactly: {result['matched']}")
    if result["mismatches"]:
        print(f"MISMATCHES ({len(result['mismatches'])}):")
        for inst, cutoff, sql_pos, py_pos in result["mismatches"][:20]:
            print(f"  {inst} @ seq<={cutoff}: sql={sql_pos} python={py_pos}")
    print()
    print(f"meets 'any past risk number reconstructable': "
          f"{result['matched'] == result['checked']} "
          f"({result['matched']} of {result['checked']})")

    conn.close()


if __name__ == "__main__":
    main()
