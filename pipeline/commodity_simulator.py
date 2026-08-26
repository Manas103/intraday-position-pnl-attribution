"""Deterministic, seeded simulator for a small commodity forward book.

Everything produced here is synthetic. There is no real market data or real
deal anywhere in this module; see the README's honest framing section. This
is a second, parallel simulator to `pipeline/simulator.py` (the options
book), reusing the same "pure function of an integer seed" discipline but
new domain logic: forward curves and a deal blotter instead of an option
universe and a fill stream.

Two independent pieces are generated per commodity, from the same seed:

1. A forward curve, sampled at a small grid of tenor points
   (`CURVE_TENOR_GRID_DAYS`) and linearly interpolated for any delivery
   month in between. Each session the curve evolves by a *level* move (one
   scalar added to every grid point, a parallel shift) and a *shape* move
   (one independent draw per grid point, forced to have exactly zero mean
   across the grid before being applied). Forcing the shape draw to zero
   mean is what makes the two genuinely, exactly separable: the level delta
   is provably the entire parallel component and the shape delta is
   provably the entire non-parallel component, not an approximation.

2. A deal blotter: a stream of new/amend/cancel actions on forward
   contracts for that commodity, with a strictly increasing `sequence` per
   instrument (mirroring the fill stream's sequencing convention) and a
   `deal_id` that repeats across an amend or cancel of the same economic
   deal (unlike a fill_id, which is always unique per message).

Both are pure functions of the seed: same seed, same output, every time.
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass

from .simulator import derive_seed

# Tenor grid the forward curve is actually sampled at, in days to delivery.
# Any delivery month's mark is linearly interpolated between the two
# bracketing grid points, and clamped flat beyond the outer two (a
# disclosed simplification; see README Limitations).
CURVE_TENOR_GRID_DAYS = [30, 90, 180, 270, 365, 540, 730]

# Plausible, not calibrated, starting curves: WTI in mild backwardation,
# the two refined products in mild contango. Units are abstract price per
# contract unit (not intended to map to real $/bbl or $/gal quoting).
COMMODITY_BASE_CURVE: dict[str, list[float]] = {
    "WTI": [72.0, 71.5, 71.0, 70.5, 70.0, 69.5, 69.0],
    "RBOB": [2.10, 2.15, 2.20, 2.22, 2.24, 2.26, 2.28],
    "ULSD": [2.40, 2.42, 2.45, 2.47, 2.50, 2.53, 2.55],
}

# Per-session gaussian standard deviations for the level and shape moves.
# Refined products are quoted at a much smaller absolute price level than
# crude, so their vols are scaled down to stay a plausible fraction of
# price, not a plausible fraction of WTI's price.
COMMODITY_LEVEL_VOL: dict[str, float] = {"WTI": 0.6, "RBOB": 0.03, "ULSD": 0.03}
COMMODITY_SHAPE_VOL: dict[str, float] = {"WTI": 0.15, "RBOB": 0.008, "ULSD": 0.008}

# Four delivery-month buckets per commodity, chosen so that even after
# rolling down by one day per session for the full 250-session run, every
# instrument's tenor stays comfortably positive (no simulated expiry to
# handle) and mostly inside the interpolation grid.
INITIAL_TENOR_DAYS = [320, 470, 620, 770]

# Negotiated execution spread around the session-open curve mark for a
# brand new deal, expressed as a fraction of that mark. This is what gives
# the "new deals" attribution bucket a genuine, nonzero day-one gap.
TRADE_SPREAD_FRACTION = 0.003

SESSION_ZERO_DATE = datetime.date(2026, 1, 5)


@dataclass(frozen=True)
class ForwardInstrument:
    instrument_id: str  # e.g. "WTI-320D"
    commodity: str
    initial_tenor_days: int


def build_commodity_universe() -> list[ForwardInstrument]:
    instruments = []
    for commodity in COMMODITY_BASE_CURVE:
        for tenor in INITIAL_TENOR_DAYS:
            instruments.append(ForwardInstrument(f"{commodity}-{tenor}D", commodity, tenor))
    return instruments


def _interp(grid_days: list[int], grid_values: tuple[float, ...], tenor_days: float) -> float:
    """Piecewise-linear interpolation, clamped flat outside the grid. This
    is a linear operator in `grid_values` for a fixed `tenor_days` and fixed
    `grid_days`, which is the property the attribution engine's exact
    level/shape/roll decomposition depends on (see commodity_attribution.py
    module docstring)."""
    if tenor_days <= grid_days[0]:
        return grid_values[0]
    if tenor_days >= grid_days[-1]:
        return grid_values[-1]
    for i in range(len(grid_days) - 1):
        d0, d1 = grid_days[i], grid_days[i + 1]
        if d0 <= tenor_days <= d1:
            w = (tenor_days - d0) / (d1 - d0)
            return grid_values[i] * (1.0 - w) + grid_values[i + 1] * w
    return grid_values[-1]  # unreachable given the bounds checks above


@dataclass(frozen=True)
class CurveSession:
    session_id: int
    grid_sod: tuple[float, ...]
    grid_eod: tuple[float, ...]
    level_delta: float          # the session's parallel move, exact by construction
    shape_delta: tuple[float, ...]  # the session's per-grid-point shape move, exactly zero-mean

    def mark_sod(self, tenor_days: float) -> float:
        return _interp(CURVE_TENOR_GRID_DAYS, self.grid_sod, tenor_days)

    def mark_eod(self, tenor_days: float) -> float:
        return _interp(CURVE_TENOR_GRID_DAYS, self.grid_eod, tenor_days)

    def shape_at(self, tenor_days: float) -> float:
        return _interp(CURVE_TENOR_GRID_DAYS, self.shape_delta, tenor_days)


def build_curve_sessions(seed: int, commodity: str, n_sessions: int = 250) -> list[CurveSession]:
    rng = random.Random(derive_seed(seed, commodity, "curve"))
    grid = list(COMMODITY_BASE_CURVE[commodity])
    level_vol = COMMODITY_LEVEL_VOL[commodity]
    shape_vol = COMMODITY_SHAPE_VOL[commodity]
    n_points = len(CURVE_TENOR_GRID_DAYS)

    sessions: list[CurveSession] = []
    for session_id in range(n_sessions):
        grid_sod = tuple(grid)
        level_delta = rng.gauss(0.0, level_vol)
        raw_shape = [rng.gauss(0.0, shape_vol) for _ in range(n_points)]
        mean_shape = sum(raw_shape) / n_points
        shape_delta = tuple(x - mean_shape for x in raw_shape)  # exactly zero-mean by construction
        grid_eod = tuple(g + level_delta + s for g, s in zip(grid_sod, shape_delta))
        sessions.append(CurveSession(session_id, grid_sod, grid_eod, level_delta, shape_delta))
        grid = list(grid_eod)
    return sessions


@dataclass(frozen=True)
class Deal:
    deal_id: str
    sequence: int
    action: str  # "new" | "amend" | "cancel"
    commodity: str
    delivery_tenor_days: int  # identifies the instrument: initial_tenor_days of its ForwardInstrument
    volume: float  # signed. "new": the deal's volume. "amend": the new ABSOLUTE volume (not a delta).
                    # "cancel": always 0.0 (informational; the instrument's ingest logic treats any
                    # cancel as setting this deal_id's contribution to zero regardless of this field).
    trade_price: float | None  # negotiated price, "new" deals only; None for amend/cancel
    direction: str  # "buy" if volume/new-volume >= 0 else "sell"; derived, kept for blotter realism
    session_id: int
    timestamp: str


def instrument_id_of(commodity: str, tenor_days: int) -> str:
    return f"{commodity}-{tenor_days}D"


def instrument_id(deal: Deal) -> str:
    return instrument_id_of(deal.commodity, deal.delivery_tenor_days)


def _timestamp(session_id: int, sequence: int) -> str:
    day = SESSION_ZERO_DATE + datetime.timedelta(days=session_id)
    hour = 8 + (sequence % 8)
    minute = (sequence * 7) % 60
    return f"{day.isoformat()}T{hour:02d}:{minute:02d}:00"


def build_deal_blotter(
    seed: int,
    universe: list[ForwardInstrument],
    curve_sessions: dict[str, list["CurveSession"]],
    n_sessions: int = 250,
) -> list[Deal]:
    """Build the canonical, correctly-ordered deal stream across the whole
    book: for each session, each instrument gets 1-3 actions. `sequence`
    increases strictly monotonically per instrument across the entire
    stream (never reset per session), exactly mirroring
    `simulator.build_canonical_fill_stream`'s sequencing convention.

    Roughly 65% of actions are "new" (or all actions, before any deal
    exists yet to amend or cancel); of the remainder, about 70% are amend
    and 30% are cancel, targeting a uniformly random currently-open deal_id
    for that instrument.
    """
    rng = random.Random(derive_seed(seed, "commodity_deals"))
    per_instrument_sequence = {inst.instrument_id: 0 for inst in universe}
    open_deals: dict[str, dict[str, float]] = {inst.instrument_id: {} for inst in universe}

    deals: list[Deal] = []

    for session_id in range(n_sessions):
        for inst in universe:
            cs = curve_sessions[inst.commodity][session_id]
            tenor_sod = inst.initial_tenor_days - session_id
            mark_sod = cs.mark_sod(tenor_sod)

            n_actions = rng.randint(1, 3)
            for _ in range(n_actions):
                per_instrument_sequence[inst.instrument_id] += 1
                sequence = per_instrument_sequence[inst.instrument_id]
                timestamp = _timestamp(session_id, sequence)
                existing_ids = list(open_deals[inst.instrument_id].keys())

                if existing_ids and rng.random() < 0.35:
                    action = "cancel" if rng.random() < 0.3 else "amend"
                else:
                    action = "new"

                if action == "new":
                    deal_id = f"DEAL-{inst.instrument_id}-{sequence:06d}"
                    magnitude = float(rng.randint(1, 10))
                    signed_volume = magnitude if rng.random() < 0.5 else -magnitude
                    spread = rng.uniform(-TRADE_SPREAD_FRACTION, TRADE_SPREAD_FRACTION)
                    trade_price = round(mark_sod * (1.0 + spread), 4)
                    direction = "buy" if signed_volume >= 0 else "sell"
                    deals.append(Deal(deal_id, sequence, "new", inst.commodity, inst.initial_tenor_days,
                                       signed_volume, trade_price, direction, session_id, timestamp))
                    open_deals[inst.instrument_id][deal_id] = signed_volume

                elif action == "amend":
                    deal_id = rng.choice(existing_ids)
                    current = open_deals[inst.instrument_id][deal_id]
                    change = rng.uniform(-0.5, 0.5) * max(abs(current), 3.0)
                    new_volume = round(current + change, 4)
                    direction = "buy" if new_volume >= 0 else "sell"
                    deals.append(Deal(deal_id, sequence, "amend", inst.commodity, inst.initial_tenor_days,
                                       new_volume, None, direction, session_id, timestamp))
                    open_deals[inst.instrument_id][deal_id] = new_volume

                else:  # cancel
                    deal_id = rng.choice(existing_ids)
                    deals.append(Deal(deal_id, sequence, "cancel", inst.commodity, inst.initial_tenor_days,
                                       0.0, None, "buy", session_id, timestamp))
                    del open_deals[inst.instrument_id][deal_id]

    return deals
