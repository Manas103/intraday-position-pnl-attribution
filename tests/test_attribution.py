import math

from pipeline import cpp_bridge
from pipeline.attribution import run_full_attribution


def test_single_position_components_sum_to_actual_pnl(cpp_engine_path):
    requests = [("t1", 100.0, 100.0, 1.0, 0.05, 0.2, "C", 102.0, 0.21, 0.996)]
    result = cpp_bridge.run_attribution_batch(requests, cpp_engine_path)[0]

    reconstructed = result.delta_pnl + result.gamma_pnl + result.vega_pnl + result.theta_pnl + result.residual
    assert math.isclose(reconstructed, result.actual_pnl, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result.taylor_sum, result.delta_pnl + result.gamma_pnl + result.vega_pnl + result.theta_pnl)


def test_reference_oracle_matches_direct_repricing(cpp_engine_path):
    """actual_pnl must equal price(EOD state) - price(SOD state) exactly,
    which is the reference-oracle definition of ground truth here."""
    requests = [
        ("a", 150.0, 140.0, 0.5, 0.03, 0.25, "P", 148.0, 0.26, 0.496),
        ("b", 60.0, 65.0, 2.0, 0.02, 0.4, "C", 55.0, 0.5, 1.996),
    ]
    results = cpp_bridge.run_attribution_batch(requests, cpp_engine_path)
    for r in results:
        assert math.isclose(r.actual_pnl, r.price_eod - r.price_sod, rel_tol=1e-12)


def test_zero_market_move_gives_zero_pnl_and_zero_residual(cpp_engine_path):
    requests = [("noop", 100.0, 100.0, 1.0, 0.05, 0.2, "C", 100.0, 0.2, 1.0)]
    r = cpp_bridge.run_attribution_batch(requests, cpp_engine_path)[0]
    assert math.isclose(r.actual_pnl, 0.0, abs_tol=1e-12)
    assert math.isclose(r.residual, 0.0, abs_tol=1e-12)


def test_small_move_residual_is_small_relative_to_pnl(cpp_engine_path):
    """For a small, smooth market move, the second-order Taylor expansion
    should explain almost all of the P&L (sanity check on the attribution
    math itself, independent of the full 250-session measurement)."""
    requests = [("small", 100.0, 100.0, 1.0, 0.05, 0.2, "C", 100.3, 0.201, 0.996)]
    r = cpp_bridge.run_attribution_batch(requests, cpp_engine_path)[0]
    assert abs(r.residual) < 0.02 * abs(r.actual_pnl)


def test_full_run_produces_250_sessions_across_8_instruments(cpp_engine_path):
    result = run_full_attribution(seed=42, n_sessions=250, n_instruments=8, binary_path=cpp_engine_path)
    assert result["n_sessions"] == 250
    assert len(result["rows"]) == 250 * 8
    assert 0.0 <= result["overall_residual_ratio"] < 1.0  # sanity bound, not the claim threshold
    assert result["total_gross_pnl"] > 0.0


def test_gap_sessions_have_worse_residual_ratio_than_normal_sessions(cpp_engine_path):
    """Documents the known limitation: large gap moves are where the
    second-order Taylor attribution is expected to do worst, because a
    single point-estimate delta/gamma poorly approximates a large jump."""
    result = run_full_attribution(seed=42, n_sessions=250, n_instruments=8, binary_path=cpp_engine_path)
    assert result["gap_session_ratio_mean"] > result["non_gap_session_ratio_mean"]


def test_component_invariant_holds_across_the_full_run(cpp_engine_path):
    result = run_full_attribution(seed=42, n_sessions=250, n_instruments=8, binary_path=cpp_engine_path)
    for row in result["rows"]:
        reconstructed = row.delta_pnl + row.gamma_pnl + row.vega_pnl + row.theta_pnl + row.residual
        assert math.isclose(reconstructed, row.actual_pnl, rel_tol=1e-9, abs_tol=1e-9)


def test_deterministic_given_same_seed(cpp_engine_path):
    r1 = run_full_attribution(seed=7, n_sessions=30, n_instruments=4, binary_path=cpp_engine_path)
    r2 = run_full_attribution(seed=7, n_sessions=30, n_instruments=4, binary_path=cpp_engine_path)
    assert r1["overall_residual_ratio"] == r2["overall_residual_ratio"]
    assert [row.actual_pnl for row in r1["rows"]] == [row.actual_pnl for row in r2["rows"]]
