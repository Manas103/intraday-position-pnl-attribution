"""Measures the multi-asset extension's residual ratio over 250 sessions.
Run from the repo root: python scripts/run_multi_asset_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.multi_asset.attribution import run_full_multi_asset_attribution  # noqa: E402


def main() -> None:
    result = run_full_multi_asset_attribution(seed=42, n_sessions=250)
    df = result["df"]

    print(f"instruments: {result['n_instruments']}  sessions: {result['n_sessions']}")
    by_class = df.groupby("asset_class").agg(
        gross=("gross_pnl", "sum"), residual=("abs_residual", "sum"), n=("instrument_id", "nunique"),
    )
    print("\nper asset class:")
    for asset_class, row in by_class.iterrows():
        ratio = (row["residual"] / row["gross"] * 100.0) if row["gross"] else 0.0
        print(f"  {asset_class}: n_instruments={int(row['n'])} gross={row['gross']:.2f} "
              f"residual={row['residual']:.4f} ratio={ratio:.4f}%")

    total_gross = result["total_gross_pnl"]
    total_residual = result["total_unexplained_residual"]
    ratio_pct = result["overall_residual_ratio"] * 100.0

    print(f"\ntotal gross P&L: {total_gross:.4f}")
    print(f"total unexplained residual: {total_residual:.4f}")
    print(f"OVERALL RESIDUAL RATIO: {ratio_pct:.4f}%  (claimed 0.20%)")


if __name__ == "__main__":
    main()
