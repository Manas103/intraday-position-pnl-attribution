"""Deterministic, seeded fill-stream generator for the risk platform
extension, scaled for a 2M+ fill run.

Reuses the exact two conventions the original options book's
`pipeline/simulator.py` establishes: `sequence` increases strictly
monotonically per instrument (not globally), and `fill_id` encodes the
instrument and sequence so it is unique on its own even though the
database's idempotency key is the pair `(fill_id, sequence)`. Fills are
plain tuples here, not dataclass instances, to keep the memory footprint
of a multi-million-row in-process list manageable (resource courtesy; see
playbook section 3z).
"""

from __future__ import annotations

import random
from typing import Iterator, NamedTuple

TRADING_DAYS_PER_YEAR = 252


def derive_seed(*parts) -> str:
    """Same construction as pipeline.simulator.derive_seed, duplicated
    rather than imported so this extension has no import-time dependency
    on the SQLite options book's module (see README: the two subsystems
    are not wired together)."""
    return "|".join(str(p) for p in parts)


class Fill(NamedTuple):
    fill_id: str
    sequence: int
    instrument: str
    quantity: float   # signed: positive = buy, negative = sell
    price: float
    session_id: int


def instrument_ids(n_instruments: int) -> list[str]:
    return [f"RISK-{i:04d}" for i in range(n_instruments)]


def generate_canonical_stream(
    seed: int,
    n_instruments: int,
    n_sessions: int,
    min_fills_per_instrument_session: int = 16,
    max_fills_per_instrument_session: int = 26,
) -> Iterator[Fill]:
    """Yields the canonical fill stream in session-then-instrument order
    (the order a real feed would plausibly deliver it in before any
    perturbation). A generator, not a list, so the caller can stream
    straight into a COPY buffer without ever materializing the full
    stream twice.

    Price follows an independent geometric random walk per instrument
    (mean-reverting around a per-instrument base price), since this
    module only needs a plausible, deterministic price series to support
    exposure = position * last_traded_price, not a realistic options
    book like the original simulator's.
    """
    instruments = instrument_ids(n_instruments)
    rng = random.Random(derive_seed(seed, "risk_platform_canonical"))
    base_price = {inst: 20.0 + rng.uniform(0, 480.0) for inst in instruments}
    price = dict(base_price)
    per_instrument_sequence = {inst: 0 for inst in instruments}

    for session_id in range(n_sessions):
        for inst in instruments:
            price[inst] = max(0.5, price[inst] * (1.0 + rng.gauss(0.0, 0.01)))
            n_fills = rng.randint(min_fills_per_instrument_session, max_fills_per_instrument_session)
            for _ in range(n_fills):
                per_instrument_sequence[inst] += 1
                sequence = per_instrument_sequence[inst]
                side = 1.0 if rng.random() < 0.5 else -1.0
                qty = side * rng.randint(1, 50)
                trade_price = round(price[inst] * (1.0 + rng.gauss(0.0, 0.0015)), 4)
                fill_id = f"FILL-{inst}-{sequence:07d}"
                yield Fill(fill_id, sequence, inst, float(qty), trade_price, session_id)


def perturb_stream(seed: int, canonical: list[Fill], dup_fraction: float = 0.05, window: int = 10) -> list[Fill]:
    """Identical algorithm to pipeline.replay.perturb_stream (duplicate a
    seeded fraction of messages, then shuffle the whole stream in fixed
    non-overlapping windows), reimplemented here on plain tuples rather
    than imported, for the same reason derive_seed is duplicated above."""
    rng = random.Random(derive_seed(seed, "risk_platform_perturb"))
    n = len(canonical)
    n_dup = round(n * dup_fraction)
    dup_indices = set(rng.sample(range(n), n_dup)) if n_dup else set()

    with_dups: list[Fill] = []
    for i, msg in enumerate(canonical):
        with_dups.append(msg)
        if i in dup_indices:
            with_dups.append(msg)

    perturbed: list[Fill] = []
    for start in range(0, len(with_dups), window):
        block = list(with_dups[start:start + window])
        rng.shuffle(block)
        perturbed.extend(block)
    return perturbed


def measured_duplicate_fraction(n_canonical: int, perturbed: list[Fill]) -> float:
    from collections import Counter

    counts = Counter((f.fill_id, f.sequence) for f in perturbed)
    n_dup_messages = sum(c - 1 for c in counts.values() if c > 1)
    return n_dup_messages / n_canonical if n_canonical else 0.0
