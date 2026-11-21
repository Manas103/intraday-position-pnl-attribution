"""Measures how often a price-first vs. FX-first sequential bump ordering
disagrees on the multi-asset book. Run from the repo root:
python scripts/run_multi_asset_order_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.multi_asset.order_check import compare_sequential_orders, summarize  # noqa: E402


def main() -> None:
    rows = compare_sequential_orders(seed=42, n_sessions=250)
    summary = summarize(rows)

    print(f"total instrument-sessions: {summary['n_rows']}")
    print(f"USD-denominated rows: {summary['n_usd_rows']} (disagree: {summary['n_usd_disagree']})")
    print(f"foreign-currency rows: {summary['n_foreign_rows']} (disagree: {summary['n_foreign_disagree']})")
    print(f"mean abs discrepancy: {summary['mean_abs_discrepancy']:.6f}")
    print(f"\nROWS WHERE PRICE-FIRST AND FX-FIRST DISAGREE: {summary['n_disagree']} of {summary['n_rows']} "
          f"({summary['pct_disagree']:.2f}%)  (claimed 98.6%)")


if __name__ == "__main__":
    main()
