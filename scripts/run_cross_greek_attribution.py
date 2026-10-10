#!/usr/bin/env python
"""Run the full 250-session attribution with the cross-Greek (vanna/volga)
extension and print the baseline-vs-extended residual ratio comparison,
plus the sessions whose extended residual still exceeds 1% of that
session's gross P&L.

Usage:
    python scripts/run_cross_greek_attribution.py
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
    print()
    print(f"baseline (delta/gamma/vega/theta) residual ratio:     {result['overall_residual_ratio'] * 100:.4f}%")
    print(f"cross-Greek (+ vanna/volga) residual ratio:           {result['overall_residual_ratio_full'] * 100:.4f}%")
    reduction = result["overall_residual_ratio"] - result["overall_residual_ratio_full"]
    print(f"absolute reduction in residual ratio: {reduction * 100:.4f} percentage points")
    print()
    print(f"baseline gap-session ratio, mean:     {result['gap_session_ratio_mean'] * 100:.4f}%")
    print(f"cross-Greek gap-session ratio, mean:  {result['gap_session_ratio_mean_full'] * 100:.4f}%")
    print(f"baseline non-gap ratio, mean:         {result['non_gap_session_ratio_mean'] * 100:.4f}%")
    print(f"cross-Greek non-gap ratio, mean:      {result['non_gap_session_ratio_mean_full'] * 100:.4f}%")
    print()
    sessions_over = result["sessions_over_1pct_full"]
    print(f"sessions with cross-Greek residual ratio still over 1%: {len(sessions_over)}")
    for sid in sorted(sessions_over):
        print(f"  session {sid}: residual_full ratio = {result['per_session_ratio_full'][sid] * 100:.4f}%")
    print()
    claim_baseline = 0.0054
    claim_extended = 0.0012
    print(
        f"claim: baseline residual ~0.54% cut to ~0.12% by cross-Greek terms -> "
        f"measured {result['overall_residual_ratio'] * 100:.4f}% cut to "
        f"{result['overall_residual_ratio_full'] * 100:.4f}%"
    )


if __name__ == "__main__":
    main()
