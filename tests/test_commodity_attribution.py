import math

from pipeline import commodity_simulator as csim
from pipeline.commodity_attribution import (
    SessionDealAction,
    attribute_session,
    run_full_commodity_attribution,
)

SEED = 42


def _make_curve_session(level_delta, shape_delta, base=70.0):
    grid_sod = tuple(base for _ in csim.CURVE_TENOR_GRID_DAYS)
    grid_eod = tuple(g + level_delta + s for g, s in zip(grid_sod, shape_delta))
    return csim.CurveSession(0, grid_sod, grid_eod, level_delta, tuple(shape_delta))


def test_component_invariant_holds_across_the_full_run():
    result = run_full_commodity_attribution(seed=SEED, n_sessions=250)
    for row in result["rows"]:
        reconstructed = (
            row.price_pnl + row.curve_shift_pnl + row.time_pnl
            + row.new_deals_pnl + row.volume_pnl + row.residual
        )
        assert math.isclose(reconstructed, row.actual_pnl, rel_tol=1e-9, abs_tol=1e-9)


def test_residual_is_floating_point_noise_not_a_meaningful_gap():
    """The headline claim: mean unexplained residual under 1% of gross P&L
    over 250 simulated sessions. This is the second, current attempt (see
    pipeline/commodity_attribution.py module docstring); it measures far
    below the 1% threshold, at floating-point-noise scale."""
    result = run_full_commodity_attribution(seed=SEED, n_sessions=250)
    assert result["overall_residual_ratio"] < 0.01
    assert result["overall_residual_ratio"] < 1e-6  # documents how far below the threshold it actually is


def test_full_run_produces_expected_row_count():
    result = run_full_commodity_attribution(seed=SEED, n_sessions=250)
    assert result["n_sessions"] == 250
    assert result["n_instruments"] == 12
    assert len(result["rows"]) == 250 * 12
    assert result["total_gross_pnl"] > 0.0


def test_deterministic_given_same_seed():
    r1 = run_full_commodity_attribution(seed=7, n_sessions=30)
    r2 = run_full_commodity_attribution(seed=7, n_sessions=30)
    assert r1["overall_residual_ratio"] == r2["overall_residual_ratio"]
    assert [row.actual_pnl for row in r1["rows"]] == [row.actual_pnl for row in r2["rows"]]


def test_attribute_session_single_new_deal_zero_move_gives_zero_pnl():
    cs = _make_curve_session(level_delta=0.0, shape_delta=[0.0] * len(csim.CURVE_TENOR_GRID_DAYS))
    actions = [SessionDealAction("D1", "new", 10.0, 70.0)]  # struck exactly at the flat curve, no gap
    step = attribute_session(v0=0.0, deal_volume_prev={}, tenor_sod=300, cs=cs, actions=actions)
    assert math.isclose(step.actual_pnl, 0.0, abs_tol=1e-9)
    assert math.isclose(step.residual, 0.0, abs_tol=1e-9)
    assert step.v1 == 10.0


def test_attribute_session_hand_checked_scenario():
    """A small, human-checkable scenario: existing position of 5, curve
    moves up by a level of 2.0 (no shape change), plus one new deal of
    +3 struck 1.0 below the session-open mark."""
    n = len(csim.CURVE_TENOR_GRID_DAYS)
    cs = _make_curve_session(level_delta=2.0, shape_delta=[0.0] * n, base=70.0)
    actions = [SessionDealAction("D1", "new", 3.0, 69.0)]
    step = attribute_session(v0=5.0, deal_volume_prev={}, tenor_sod=300, cs=cs, actions=actions)

    # price: 5 * 2.0 = 10.0 (flat curve, so mark_sod == mark_eod == 70/72 everywhere -> no shape/time)
    assert math.isclose(step.price_pnl, 10.0, abs_tol=1e-9)
    assert math.isclose(step.curve_shift_pnl, 0.0, abs_tol=1e-9)
    assert math.isclose(step.time_pnl, 0.0, abs_tol=1e-9)
    # new_deals: 3 * (mark_eod=72.0) - 3*69.0 = 216 - 207 = 9.0
    assert math.isclose(step.new_deals_pnl, 9.0, abs_tol=1e-9)
    assert math.isclose(step.volume_pnl, 0.0, abs_tol=1e-9)
    # actual_pnl = v1*mark_eod - v0*mark_sod - cash = 8*72 - 5*70 - 3*69 = 576-350-207=19
    assert math.isclose(step.actual_pnl, 19.0, abs_tol=1e-9)
    assert math.isclose(step.residual, 0.0, abs_tol=1e-9)


def test_reference_oracle_independent_per_deal_loop_matches_engine():
    """Reference-oracle diff: actual_pnl recomputed by a plain, separately
    coded per-deal loop (not the engine's aggregate dv_new_total/cash_new
    sums) must match the engine's actual_pnl to tight tolerance, for a
    real slice of the full run."""
    result = run_full_commodity_attribution(seed=SEED, n_sessions=40)
    universe = csim.build_commodity_universe()
    commodities = sorted({inst.commodity for inst in universe})
    curve_sessions = {c: csim.build_curve_sessions(SEED, c, 40) for c in commodities}
    deals = csim.build_deal_blotter(SEED, universe, curve_sessions, 40)

    deals_by_instrument = {}
    for d in deals:
        deals_by_instrument.setdefault(csim.instrument_id(d), []).append(d)

    checked = 0
    for row in result["rows"]:
        inst_deals = [d for d in deals_by_instrument.get(row.instrument_id, []) if d.session_id == row.session_id]
        if not inst_deals or not all(d.action == "new" for d in inst_deals):
            continue  # only check sessions with pure "new" activity, where the independent formula is simplest
        # independent per-deal loop: accumulate cash flow one message at a time, not via the
        # engine's aggregate dv_new_total/cash_new running sums.
        cash = 0.0
        for d in sorted(inst_deals, key=lambda x: x.sequence):
            cash += d.volume * d.trade_price
        oracle_actual_pnl = row.v1 * row.mark_eod - row.v0 * row.mark_sod - cash
        assert math.isclose(oracle_actual_pnl, row.actual_pnl, rel_tol=1e-9, abs_tol=1e-9)
        checked += 1
    assert checked > 0
