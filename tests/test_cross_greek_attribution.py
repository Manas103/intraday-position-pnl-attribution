import math

from pipeline import cpp_bridge
from pipeline.attribution import run_full_attribution
from pipeline.sequential_order_check import compare_sequential_orders


def test_cross_greek_component_invariant_holds(cpp_engine_path):
    requests = [
        ("t1", 100.0, 100.0, 1.0, 0.05, 0.2, "C", 102.0, 0.21, 0.996),
        ("t2", 150.0, 140.0, 0.5, 0.03, 0.25, "P", 148.0, 0.26, 0.496),
    ]
    results = cpp_bridge.run_attribution_batch(requests, cpp_engine_path)
    for r in results:
        reconstructed = (
            r.delta_pnl + r.gamma_pnl + r.vega_pnl + r.theta_pnl + r.vanna_pnl + r.volga_pnl + r.residual_full
        )
        assert math.isclose(reconstructed, r.actual_pnl, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(r.taylor_sum_full, r.taylor_sum + r.vanna_pnl + r.volga_pnl)


def test_vanna_and_volga_are_zero_at_zero_market_move(cpp_engine_path):
    requests = [("noop", 100.0, 100.0, 1.0, 0.05, 0.2, "C", 100.0, 0.2, 1.0)]
    r = cpp_bridge.run_attribution_batch(requests, cpp_engine_path)[0]
    assert math.isclose(r.vanna_pnl, 0.0, abs_tol=1e-12)
    assert math.isclose(r.volga_pnl, 0.0, abs_tol=1e-12)
    assert math.isclose(r.residual_full, 0.0, abs_tol=1e-12)


def test_cross_greek_residual_ratio_is_no_worse_than_baseline(cpp_engine_path):
    """Adding real terms to a Taylor expansion cannot increase the exact
    reconstruction error for the same move; it can only re-explain some of
    what the baseline left in the residual."""
    result = run_full_attribution(seed=42, n_sessions=250, n_instruments=8, binary_path=cpp_engine_path)
    assert result["overall_residual_ratio_full"] <= result["overall_residual_ratio"] + 1e-12


def test_full_run_tracks_sessions_whose_cross_greek_residual_still_exceeds_1pct(cpp_engine_path):
    result = run_full_attribution(seed=42, n_sessions=250, n_instruments=8, binary_path=cpp_engine_path)
    sessions_over = result["sessions_over_1pct_full"]
    assert isinstance(sessions_over, list)
    for sid in sessions_over:
        assert result["per_session_ratio_full"][sid] > 0.01


def test_sequential_order_disagrees_on_the_vol_bucket(cpp_engine_path):
    """The whole point of the simultaneous attribution: two equally
    reasonable sequential orderings of the same two bumps (spot then vol,
    vol then spot) assign the interaction to different buckets. If this
    ever stopped being true, the resume claim this extension exists to
    support would be false."""
    sessions = [
        ("t1", 1, 100.0, 100.0, 1.0, 0.05, 0.2, "C", 108.0, 0.26, 0.0),
        ("t2", 2, 150.0, 140.0, 0.5, 0.03, 0.25, "P", 138.0, 0.33, 0.0),
        ("t3", 3, 60.0, 65.0, 2.0, 0.02, 0.4, "C", 50.0, 0.55, 0.0),
    ]
    rows = compare_sequential_orders(sessions, cpp_engine_path)
    for r in rows:
        assert abs(r.bucket_discrepancy) > 1e-6, (
            f"{r.instrument}:{r.session_id} expected spot-first and vol-first orderings "
            "to disagree on the vol bucket for a two-way market move"
        )
