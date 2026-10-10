"""Run full-book P&L attribution across all simulated sessions and measure
the unexplained residual as a fraction of gross P&L.

Definitions (used consistently everywhere in this repo and the README):
  actual_pnl (per instrument, per session) = full reprice at EOD market
    state minus full reprice at SOD market state, both via the exact
    Black-Scholes formula (the reference oracle). This is ground truth.
  taylor_sum = delta_pnl + gamma_pnl + vega_pnl + theta_pnl, the first- and
    second-order Taylor approximation to actual_pnl using SOD Greeks.
  residual = actual_pnl - taylor_sum. By construction,
    delta_pnl + gamma_pnl + vega_pnl + theta_pnl + residual == actual_pnl
    exactly (see tests/test_attribution.py).
  gross P&L (per session) = sum over instruments of abs(actual_pnl). This
    is the denominator: total absolute mark-to-market movement in the book
    that day, not net P&L (net P&L can be near zero while gross is large
    if positions offset, which would make a ratio to net meaningless).
  unexplained residual (per session) = sum over instruments of
    abs(residual).
  residual ratio = sum(unexplained residual) / sum(gross P&L), aggregated
    over all sessions and instruments (not averaged per-session ratios,
    which would over-weight low-volume sessions).

Cross-Greek extension: everything above is untouched and still computed
exactly as before. taylor_sum_full = taylor_sum + vanna_pnl + volga_pnl,
computed simultaneously from the same SOD Greeks and the same (dS, dVol)
deltas as the original four terms, not as a second sequential bump on top
of them (see scripts/run_sequential_order_check.py for why that
distinction matters). residual_full = actual_pnl - taylor_sum_full, and
residual_full_ratio is the same ratio definition as residual ratio above,
applied to residual_full instead of residual.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import cpp_bridge, simulator


@dataclass(frozen=True)
class SessionRow:
    instrument: str
    session_id: int
    is_gap: bool
    price_sod: float
    price_eod: float
    actual_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    residual: float
    vanna_pnl: float
    volga_pnl: float
    taylor_sum_full: float
    residual_full: float


def run_full_attribution(seed: int = 42, n_sessions: int = 250, n_instruments: int = 8, binary_path=None):
    universe = simulator.build_universe(seed, n_instruments)

    requests = []
    meta = []
    for idx, instrument in enumerate(universe):
        sessions = simulator.build_sessions(seed, idx, instrument, n_sessions)
        for s in sessions:
            req_id = f"{instrument.instrument_id}:{s.session_id}"
            requests.append(
                (
                    req_id,
                    s.s0,
                    instrument.strike,
                    s.t0,
                    simulator.RATE,
                    s.sigma0,
                    instrument.option_type,
                    s.s1,
                    s.sigma1,
                    s.t1,
                )
            )
            meta.append((instrument.instrument_id, s.session_id, s.is_gap))

    engine_results = cpp_bridge.run_attribution_batch(requests, binary_path)

    rows: list[SessionRow] = []
    for (instrument_id, session_id, is_gap), r in zip(meta, engine_results):
        rows.append(
            SessionRow(
                instrument=instrument_id,
                session_id=session_id,
                is_gap=is_gap,
                price_sod=r.price_sod,
                price_eod=r.price_eod,
                actual_pnl=r.actual_pnl,
                delta_pnl=r.delta_pnl,
                gamma_pnl=r.gamma_pnl,
                vega_pnl=r.vega_pnl,
                theta_pnl=r.theta_pnl,
                residual=r.residual,
                vanna_pnl=r.vanna_pnl,
                volga_pnl=r.volga_pnl,
                taylor_sum_full=r.taylor_sum_full,
                residual_full=r.residual_full,
            )
        )

    per_session_gross: dict[int, float] = defaultdict(float)
    per_session_residual: dict[int, float] = defaultdict(float)
    per_session_residual_full: dict[int, float] = defaultdict(float)
    for row in rows:
        per_session_gross[row.session_id] += abs(row.actual_pnl)
        per_session_residual[row.session_id] += abs(row.residual)
        per_session_residual_full[row.session_id] += abs(row.residual_full)

    total_gross = sum(per_session_gross.values())
    total_residual = sum(per_session_residual.values())
    total_residual_full = sum(per_session_residual_full.values())
    overall_ratio = (total_residual / total_gross) if total_gross else 0.0
    overall_ratio_full = (total_residual_full / total_gross) if total_gross else 0.0

    per_session_ratio = {
        sid: (per_session_residual[sid] / per_session_gross[sid] if per_session_gross[sid] else 0.0)
        for sid in per_session_gross
    }
    per_session_ratio_full = {
        sid: (per_session_residual_full[sid] / per_session_gross[sid] if per_session_gross[sid] else 0.0)
        for sid in per_session_gross
    }

    gap_session_ids = {row.session_id for row in rows if row.is_gap}
    gap_ratios = [per_session_ratio[sid] for sid in gap_session_ids]
    non_gap_ratios = [per_session_ratio[sid] for sid in per_session_ratio if sid not in gap_session_ids]
    gap_ratios_full = [per_session_ratio_full[sid] for sid in gap_session_ids]
    non_gap_ratios_full = [
        per_session_ratio_full[sid] for sid in per_session_ratio_full if sid not in gap_session_ids
    ]

    worst_sessions_full = sorted(
        per_session_ratio_full.items(), key=lambda kv: kv[1], reverse=True
    )
    sessions_over_1pct_full = [sid for sid, ratio in worst_sessions_full if ratio > 0.01]

    return {
        "rows": rows,
        "n_sessions": n_sessions,
        "n_instruments": n_instruments,
        "total_gross_pnl": total_gross,
        "total_unexplained_residual": total_residual,
        "overall_residual_ratio": overall_ratio,
        "per_session_ratio": per_session_ratio,
        "gap_session_ratio_mean": (sum(gap_ratios) / len(gap_ratios)) if gap_ratios else 0.0,
        "non_gap_session_ratio_mean": (sum(non_gap_ratios) / len(non_gap_ratios)) if non_gap_ratios else 0.0,
        "gap_session_ratio_max": max(gap_ratios) if gap_ratios else 0.0,
        "non_gap_session_ratio_max": max(non_gap_ratios) if non_gap_ratios else 0.0,
        # Cross-Greek extension.
        "total_unexplained_residual_full": total_residual_full,
        "overall_residual_ratio_full": overall_ratio_full,
        "per_session_ratio_full": per_session_ratio_full,
        "gap_session_ratio_mean_full": (sum(gap_ratios_full) / len(gap_ratios_full)) if gap_ratios_full else 0.0,
        "non_gap_session_ratio_mean_full": (
            (sum(non_gap_ratios_full) / len(non_gap_ratios_full)) if non_gap_ratios_full else 0.0
        ),
        "sessions_over_1pct_full": sessions_over_1pct_full,
    }
