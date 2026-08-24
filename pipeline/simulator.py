"""Deterministic, seeded simulator for a small options book.

Everything produced here is synthetic. There is no real market data or real
fill anywhere in this module or this repository; see the README's honest
framing section.

Two independent scenario streams are generated from the same seed:

1. Market-state sessions per instrument (for P&L attribution): a
   start-of-day and end-of-day (spot, implied vol, time-to-maturity) for
   each of `n_sessions` simulated trading days, evolved as a geometric
   random walk in spot with a mean-reverting-ish random walk in vol, plus
   occasional injected "gap" sessions with an enlarged shock to stress the
   Taylor attribution (see README Findings).

2. A canonical fill stream (for the ingest/idempotency/replay subsystem):
   a handful of synthetic fills per instrument per session, with a strictly
   increasing `sequence` per instrument and a unique `fill_id`.

Both streams are pure functions of the seed: same seed, same output, every
time, on any machine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252
DT_YEARS = 1.0 / TRADING_DAYS_PER_YEAR
RATE = 0.03  # constant continuously-compounded risk-free rate across the book
ANNUAL_DRIFT = 0.05  # real-world (not risk-neutral) drift used to evolve spot
VOL_OF_VOL = 0.012  # per-session gaussian step size for implied vol
GAP_EVERY = 25  # inject an enlarged shock roughly every this many sessions
GAP_SHOCK_MULTIPLIER = 5.0


def derive_seed(*parts) -> str:
    """Combine a seed with arbitrary extra parts into a single deterministic
    string usable with random.Random(). random.Random only special-cases
    str/bytes/bytearray for seeding (via a fixed sha512-based expansion);
    tuples fall through to the general "int, float, str, bytes, bytearray
    only" path and raise TypeError, which is what an early version of this
    module hit immediately on the first test run. Joining parts into a
    single string sidesteps that entirely and is still fully deterministic."""
    return "|".join(str(p) for p in parts)


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    option_type: str  # "C" (call) or "P" (put)
    strike: float
    s0: float
    sigma0: float
    maturity_years: float  # time to maturity, in years, at session 0


@dataclass(frozen=True)
class SessionState:
    session_id: int
    s0: float
    sigma0: float
    t0: float
    s1: float
    sigma1: float
    t1: float
    is_gap: bool


def build_universe(seed: int, n_instruments: int = 8) -> list[Instrument]:
    rng = random.Random(derive_seed(seed, "universe"))
    instruments = []
    for i in range(n_instruments):
        option_type = "C" if i % 2 == 0 else "P"
        s0 = round(rng.uniform(50.0, 250.0), 4)
        moneyness = rng.uniform(0.85, 1.15)
        strike = round(s0 * moneyness, 2)
        sigma0 = round(rng.uniform(0.15, 0.45), 4)
        maturity_years = round(rng.uniform(1.3, 2.6), 4)
        instruments.append(
            Instrument(
                instrument_id=f"OPT{i:02d}",
                option_type=option_type,
                strike=strike,
                s0=s0,
                sigma0=sigma0,
                maturity_years=maturity_years,
            )
        )
    return instruments


def build_sessions(seed: int, index: int, instrument: Instrument, n_sessions: int = 250) -> list[SessionState]:
    """Build the SOD/EOD market-state path for one instrument.

    `index` is the instrument's position in the universe list and is mixed
    into the per-instrument RNG seed so each instrument gets an
    independent, still-fully-deterministic path.
    """
    rng = random.Random(derive_seed(seed, index, "sessions"))
    sessions: list[SessionState] = []

    s = instrument.s0
    sigma = instrument.sigma0
    t = instrument.maturity_years
    # Offset which sessions gap per instrument so not every instrument gaps
    # on the same simulated day.
    gap_offset = index % GAP_EVERY

    for session_id in range(n_sessions):
        s0, sigma0, t0 = s, sigma, t

        is_gap = (session_id % GAP_EVERY) == gap_offset
        shock_mult = GAP_SHOCK_MULTIPLIER if is_gap else 1.0

        z = rng.gauss(0.0, 1.0)
        s1 = s0 * pow(
            2.718281828459045,
            (ANNUAL_DRIFT - 0.5 * sigma0 * sigma0) * DT_YEARS + sigma0 * (DT_YEARS ** 0.5) * z * shock_mult,
        )

        vol_step = rng.gauss(0.0, VOL_OF_VOL)
        sigma1 = min(max(sigma0 + vol_step, 0.05), 0.90)

        t1 = max(t0 - DT_YEARS, 1e-6)

        sessions.append(SessionState(session_id, s0, sigma0, t0, s1, sigma1, t1, is_gap))

        s, sigma, t = s1, sigma1, t1

    return sessions


@dataclass(frozen=True)
class Fill:
    fill_id: str
    sequence: int
    instrument: str
    quantity: float
    price: float
    session_id: int


def build_canonical_fill_stream(seed: int, universe: list[Instrument], n_sessions: int = 250) -> list[Fill]:
    """Build the canonical, correctly-ordered fill stream across the whole
    book: for each session, each instrument gets 1-3 fills, interleaved
    round-robin across instruments the way a real feed would arrive.
    `sequence` increases strictly monotonically per instrument across the
    entire stream (never reset per session).
    """
    rng = random.Random(derive_seed(seed, "fills"))
    per_instrument_sequence = {inst.instrument_id: 0 for inst in universe}
    fills: list[Fill] = []

    for session_id in range(n_sessions):
        for inst in universe:
            n_fills = rng.randint(1, 3)
            base_price = inst.s0  # arbitrary but deterministic reference price
            for _ in range(n_fills):
                per_instrument_sequence[inst.instrument_id] += 1
                sequence = per_instrument_sequence[inst.instrument_id]
                quantity = rng.choice([-1, 1]) * rng.randint(1, 10)
                price = round(base_price * (1.0 + rng.uniform(-0.01, 0.01)), 4)
                fill_id = f"FILL-{inst.instrument_id}-{sequence:06d}"
                fills.append(Fill(fill_id, sequence, inst.instrument_id, float(quantity), price, session_id))

    return fills
