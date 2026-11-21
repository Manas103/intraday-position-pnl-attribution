from pipeline.multi_asset.publish_gate import Mark, evaluate


def test_fresh_in_sequence_mark_publishes():
    mark = Mark(instrument_id="EQ00", sequence=5, observed_at=100.0, price=50.0)
    decision = evaluate(mark, last_published_sequence=4, now=100.5, max_staleness_seconds=2.0)
    assert decision.publish is True


def test_stale_mark_withheld():
    mark = Mark(instrument_id="EQ00", sequence=5, observed_at=100.0, price=50.0)
    decision = evaluate(mark, last_published_sequence=4, now=103.0, max_staleness_seconds=2.0)
    assert decision.publish is False
    assert "stale" in decision.reason


def test_out_of_sequence_mark_withheld():
    mark = Mark(instrument_id="EQ00", sequence=3, observed_at=100.0, price=50.0)
    decision = evaluate(mark, last_published_sequence=5, now=100.1, max_staleness_seconds=2.0)
    assert decision.publish is False
    assert "out_of_sequence" in decision.reason


def test_first_mark_for_instrument_has_no_prior_sequence_to_compare():
    mark = Mark(instrument_id="EQ00", sequence=1, observed_at=100.0, price=50.0)
    decision = evaluate(mark, last_published_sequence=None, now=100.1, max_staleness_seconds=2.0)
    assert decision.publish is True


def test_duplicate_same_sequence_withheld_as_out_of_sequence():
    mark = Mark(instrument_id="EQ00", sequence=5, observed_at=100.0, price=50.0)
    decision = evaluate(mark, last_published_sequence=5, now=100.1, max_staleness_seconds=2.0)
    assert decision.publish is False
