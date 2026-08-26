#!/usr/bin/env python
"""Run the full 250-session commodity forward book attribution and print
the measured residual ratio. This is the exact script referenced in the
README's "Extension" section.

Usage:
    python scripts/run_commodity_attribution.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.commodity_attribution import run_full_commodity_attribution


def main():
    result = run_full_commodity_attribution(seed=42, n_sessions=250)

    print(f"sessions: {result['n_sessions']}")
    print(f"instruments: {result['n_instruments']}")
    print(f"total gross P&L (sum |actual_pnl|): {result['total_gross_pnl']:.6f}")
    print(f"total unexplained residual (sum |residual|): {result['total_unexplained_residual']:.10f}")
    print(f"overall residual ratio: {result['overall_residual_ratio'] * 100:.10f}%")
    print()
    claim_threshold = 0.01
    meets_claim = result["overall_residual_ratio"] < claim_threshold
    print(f"claim: unexplained residual under 1% of gross P&L -> {'MEETS' if meets_claim else 'DOES NOT MEET'}")


if __name__ == "__main__":
    main()
