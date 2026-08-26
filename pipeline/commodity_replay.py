"""Replay harness for the commodity deal blotter. Same construction as
`pipeline/replay.py` (duplicate a random 5% of messages, then shuffle in
fixed non-overlapping windows), applied to `commodity_simulator.Deal`'s
(deal_id, sequence) key instead of Fill's (fill_id, sequence) key.
"""

from __future__ import annotations

import random
from collections import Counter

from .commodity_simulator import Deal
from .simulator import derive_seed


def perturb_stream(seed: int, canonical: list[Deal], dup_fraction: float = 0.05, window: int = 10) -> list[Deal]:
    rng = random.Random(derive_seed(seed, "commodity_perturb"))
    n = len(canonical)
    n_dup = round(n * dup_fraction)
    dup_indices = set(rng.sample(range(n), n_dup)) if n_dup else set()

    with_dups: list[Deal] = []
    for i, msg in enumerate(canonical):
        with_dups.append(msg)
        if i in dup_indices:
            with_dups.append(msg)

    perturbed: list[Deal] = []
    for start in range(0, len(with_dups), window):
        block = list(with_dups[start:start + window])
        rng.shuffle(block)
        perturbed.extend(block)

    return perturbed


def measured_duplicate_fraction(canonical: list[Deal], perturbed: list[Deal]) -> float:
    counts = Counter((d.deal_id, d.sequence) for d in perturbed)
    n_dup_messages = sum(c - 1 for c in counts.values() if c > 1)
    return n_dup_messages / len(canonical) if canonical else 0.0
