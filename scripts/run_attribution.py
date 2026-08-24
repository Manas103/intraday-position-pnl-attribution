#!/usr/bin/env python
"""Run the full 250-session attribution and print/save the measured
residual ratio. This is the exact script referenced in the README.

Usage:
    python scripts/run_attribution.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.attribution import run_full_attribution


def main():
    result = run_full_attribution(seed=42, n_sessions=250, n_instruments=8)

    print(f"sessions: {result['n_sessions']}")
    print(f"instruments: {result['n_instruments']}")
    print(f"total gross P&L (sum |actual_pnl|): {result['total_gross_pnl']:.6f}")
    print(f"total unexplained residual (sum |residual|): {result['total_unexplained_residual']:.6f}")
    print(f"overall residual ratio: {result['overall_residual_ratio'] * 100:.4f}%")
    print()
    print(f"gap-session residual ratio, mean: {result['gap_session_ratio_mean'] * 100:.4f}%")
    print(f"gap-session residual ratio, max:  {result['gap_session_ratio_max'] * 100:.4f}%")
    print(f"non-gap-session residual ratio, mean: {result['non_gap_session_ratio_mean'] * 100:.4f}%")
    print(f"non-gap-session residual ratio, max:  {result['non_gap_session_ratio_max'] * 100:.4f}%")
    print()
    claim_threshold = 0.015
    meets_claim = result["overall_residual_ratio"] < claim_threshold
    print(f"claim: unexplained residual under 1.5% of gross P&L -> {'MEETS' if meets_claim else 'DOES NOT MEET'}")


if __name__ == "__main__":
    main()
