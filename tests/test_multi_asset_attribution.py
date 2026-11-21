import math

from pipeline.multi_asset import simulator as masim
from pipeline.multi_asset.attribution import attribute_session, run_full_multi_asset_attribution

SEED = 42


def test_component_invariant_holds_across_the_full_run():
    result = run_full_multi_asset_attribution(seed=SEED, n_sessions=250)
    for row in result["rows"]:
        reconstructed = row.price_pnl + row.rate_pnl + row.fx_pnl + row.carry_pnl + row.residual
        assert math.isclose(reconstructed, row.actual_pnl, rel_tol=1e-9, abs_tol=1e-6)


def test_full_run_produces_expected_universe():
    result = run_full_multi_asset_attribution(seed=SEED, n_sessions=250)
    assert result["n_sessions"] == 250
    assert result["n_instruments"] == 40
    assert len(result["rows"]) == 250 * 40
    assert result["total_gross_pnl"] > 0.0
    classes = {inst.asset_class for inst in result["universe"]}
    assert classes == {"EQ", "RATE", "CREDIT"}


def test_deterministic_given_same_seed():
    r1 = run_full_multi_asset_attribution(seed=7, n_sessions=30)
    r2 = run_full_multi_asset_attribution(seed=7, n_sessions=30)
    assert r1["overall_residual_ratio"] == r2["overall_residual_ratio"]
    assert [row.actual_pnl for row in r1["rows"]] == [row.actual_pnl for row in r2["rows"]]


def test_residual_is_small_relative_to_gross_pnl():
    """The headline claim: unexplained residual under some small threshold
    of gross P&L over 250 sessions. Three genuine attempts were made (see
    pipeline/multi_asset/attribution.py module docstring and the README
    Findings): first-order duration only, then adding bond convexity, then
    switching to the average-rate FX convention. This is the measured
    result of the third and final attempt."""
    result = run_full_multi_asset_attribution(seed=SEED, n_sessions=250)
    assert result["overall_residual_ratio"] < 0.01  # well under the 1% order of magnitude


def test_eq_bucket_has_exactly_zero_residual():
    """EQ instruments have no rate/credit factor and the average-rate FX
    convention makes price_bucket + fx_bucket telescope exactly to the true
    USD P&L move, so EQ rows should carry zero residual, not just a small
    one."""
    result = run_full_multi_asset_attribution(seed=SEED, n_sessions=250)
    eq_rows = [r for r in result["rows"] if r.asset_class == "EQ"]
    assert len(eq_rows) > 0
    for row in eq_rows:
        assert math.isclose(row.residual, 0.0, abs_tol=1e-6)


def test_usd_domestic_instrument_has_zero_fx_pnl_every_session():
    universe = masim.build_universe(SEED)
    domestic = [inst for inst in universe if inst.currency == "USD"]
    assert len(domestic) == masim.N_DOMESTIC_USD
    inst = domestic[0]
    idx = universe.index(inst)
    sessions = masim.build_sessions(SEED, idx, inst, n_sessions=20)
    for st in sessions:
        row = attribute_session(inst, st)
        assert row.fx_pnl == 0.0
        assert st.fx_sod == 1.0 and st.fx_eod == 1.0


def test_hand_checked_equity_scenario():
    """A small, human-checkable scenario: 10 shares, price moves from 100 to
    105, fx moves from 1.2 to 1.25, no dividend accrual this session."""
    inst = masim.Instrument(
        instrument_id="EQ_TEST", asset_class="EQ", currency="EUR", qty=10.0,
        s0=100.0, eq_vol=0.2, div_yield=0.0,
    )
    st = masim.SessionState(
        session_id=0, price_sod=100.0, price_eod=105.0, y_sod=0.0, y_eod=0.0,
        spread_sod=0.0, spread_eod=0.0, carry_local=0.0, fx_sod=1.2, fx_eod=1.25,
    )
    row = attribute_session(inst, st)
    # average-rate price bucket: 10 * 5 * ((1.2+1.25)/2) = 10*5*1.225 = 61.25
    assert math.isclose(row.price_pnl, 61.25, abs_tol=1e-9)
    # fx bucket: 10 * ((100+105)/2) * (1.25-1.2) = 10*102.5*0.05 = 51.25
    assert math.isclose(row.fx_pnl, 51.25, abs_tol=1e-9)
    # true pnl: 10*105*1.25 - 10*100*1.2 = 1312.5 - 1200 = 112.5
    assert math.isclose(row.actual_pnl, 112.5, abs_tol=1e-9)
    assert math.isclose(row.residual, 0.0, abs_tol=1e-9)
