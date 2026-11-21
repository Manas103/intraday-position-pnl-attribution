"""A mark too old or too far out of sequence is withheld rather than priced.

This is the behavior a live intraday P&L screen needs that the end-of-day
reconciliation (risk_platform/reconciliation.py) does not provide on its
own: reconciliation catches a break after the fact, at end of day; this
gate stops a stale or out-of-order mark from being published to the P&L
screen in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mark:
    instrument_id: str
    sequence: int
    observed_at: float  # simulated monotonic clock, seconds
    price: float


@dataclass(frozen=True)
class GateDecision:
    publish: bool
    reason: str


def evaluate(
    mark: Mark,
    last_published_sequence: int | None,
    now: float,
    max_staleness_seconds: float = 2.0,
    max_sequence_regression: int = 0,
) -> GateDecision:
    """Withhold a mark if it is older than `max_staleness_seconds` by the
    time it would be published, or if its sequence number is behind the
    last published sequence by more than `max_sequence_regression` (0 means
    any out-of-order mark is withheld, not just a late duplicate)."""
    age = now - mark.observed_at
    if age > max_staleness_seconds:
        return GateDecision(publish=False, reason=f"stale: age={age:.3f}s > {max_staleness_seconds}s")

    if last_published_sequence is not None:
        if mark.sequence <= last_published_sequence - max_sequence_regression:
            return GateDecision(
                publish=False,
                reason=f"out_of_sequence: mark.sequence={mark.sequence} <= "
                       f"last_published={last_published_sequence} - {max_sequence_regression}",
            )

    return GateDecision(publish=True, reason="ok")
