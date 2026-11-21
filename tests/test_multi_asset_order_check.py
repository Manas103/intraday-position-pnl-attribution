from pipeline.multi_asset.order_check import compare_sequential_orders, summarize

SEED = 42


def test_usd_rows_always_agree():
    rows = compare_sequential_orders(seed=SEED, n_sessions=30)
    usd_rows = [r for r in rows if r.currency == "USD"]
    assert len(usd_rows) > 0
    for r in usd_rows:
        assert r.discrepancy == 0.0


def test_most_foreign_rows_disagree():
    rows = compare_sequential_orders(seed=SEED, n_sessions=250)
    summary = summarize(rows)
    assert summary["n_rows"] == 250 * 40
    assert summary["n_foreign_disagree"] == summary["n_foreign_rows"]
    assert summary["pct_disagree"] > 90.0


def test_discrepancy_equals_qty_times_dprice_times_dfx():
    rows = compare_sequential_orders(seed=SEED, n_sessions=5)
    for r in rows[:50]:
        assert r.fx_bucket_a - r.fx_bucket_b == r.discrepancy


def test_deterministic_given_same_seed():
    rows1 = compare_sequential_orders(seed=7, n_sessions=10)
    rows2 = compare_sequential_orders(seed=7, n_sessions=10)
    assert [r.discrepancy for r in rows1] == [r.discrepancy for r in rows2]
