"""Replay harness: perturb a canonical fill stream with duplicates and
bounded out-of-order delivery, then check that ingesting the perturbed
stream produces bit-identical positions to ingesting the canonical stream.

Exact construction (documented here, not just in the README):

1. Start from the canonical, correctly sequenced stream of N messages.
2. Duplication: choose round(N * dup_fraction) distinct source indices at
   random (seeded). For each chosen message, insert a verbatim copy of it
   immediately after the original in the working list. This yields exactly
   the requested duplicate fraction (5% by default) before the reordering
   step below scrambles their positions too.
3. Reordering: walk the resulting (N + n_dup)-length list in fixed,
   non-overlapping windows of `window` consecutive messages and randomly
   shuffle each window in place. This bounds how far out of order any
   message can arrive (at most `window - 1` slots) while guaranteeing the
   whole stream, not just a sampled fraction of it, experiences reordering,
   which is a harder test than perturbing only part of the stream.

Both steps are driven by a single seeded `random.Random`, so the perturbed
stream is itself fully deterministic given the seed.
"""

from __future__ import annotations

import random

from .simulator import Fill, derive_seed


def perturb_stream(seed: int, canonical: list[Fill], dup_fraction: float = 0.05, window: int = 10) -> list[Fill]:
    rng = random.Random(derive_seed(seed, "perturb"))
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


def measured_duplicate_fraction(canonical: list[Fill], perturbed: list[Fill]) -> float:
    from collections import Counter

    counts = Counter((f.fill_id, f.sequence) for f in perturbed)
    n_dup_messages = sum(c - 1 for c in counts.values() if c > 1)
    return n_dup_messages / len(canonical) if canonical else 0.0
