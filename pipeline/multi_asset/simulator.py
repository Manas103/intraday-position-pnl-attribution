"""Deterministic, seeded simulator for a small cross-asset book: equity,
government-rate and credit instruments, most of them denominated in a
foreign currency against a USD base, each with its own daily carry.

Everything here is synthetic, exactly like the sibling simulators in this
repository (``pipeline/simulator.py``, ``pipeline/commodity_simulator.py``):
no real market data, no real position, no real FX rate. Same seed, same
output, every time, on any machine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252
DT_YEARS = 1.0 / TRADING_DAYS_PER_YEAR

CURRENCIES = ["EUR", "JPY", "GBP"]  # foreign currencies quoted as USD per 1 unit
N_EQUITY = 15
N_RATE = 10
N_CREDIT = 15
N_DOMESTIC_USD = 1  # how many instruments (across all classes) stay USD-only


def derive_seed(*parts) -> str:
    """Same technique as pipeline.simulator.derive_seed: random.Random only
    special-cases str/bytes/bytearray/int/float for seeding, so a tuple of
    parts is joined into one string first."""
    return "|".join(str(p) for p in parts)


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    asset_class: str  # "EQ" | "RATE" | "CREDIT"
    currency: str  # "USD" or one of CURRENCIES
    qty: float
    # EQ fields
    s0: float = 0.0
    eq_vol: float = 0.0
    div_yield: float = 0.0
    # RATE / CREDIT fields
    notional: float = 0.0
    y0: float = 0.0
    spread0: float = 0.0
    duration: float = 0.0
    coupon_rate: float = 0.0


@dataclass(frozen=True)
class SessionState:
    session_id: int
    # local-currency price/yield state
    price_sod: float
    price_eod: float
    y_sod: float
    y_eod: float
    spread_sod: float
    spread_eod: float
    carry_local: float  # accrued coupon/dividend this session, local currency
    # fx: USD per 1 unit of local currency (1.0 for USD instruments)
    fx_sod: float
    fx_eod: float


def build_universe(seed: int) -> list[Instrument]:
    rng = random.Random(derive_seed(seed, "multi_asset_universe"))
    instruments: list[Instrument] = []
    all_ids = list(range(N_EQUITY + N_RATE + N_CREDIT))
    domestic_idx = set(rng.sample(all_ids, N_DOMESTIC_USD))
    idx = 0

    for i in range(N_EQUITY):
        currency = "USD" if idx in domestic_idx else rng.choice(CURRENCIES)
        s0 = round(rng.uniform(20.0, 400.0), 4)
        eq_vol = round(rng.uniform(0.15, 0.45), 4)
        div_yield = round(rng.uniform(0.005, 0.035), 5)
        qty = float(rng.randint(50, 500))
        instruments.append(Instrument(
            instrument_id=f"EQ{i:02d}", asset_class="EQ", currency=currency, qty=qty,
            s0=s0, eq_vol=eq_vol, div_yield=div_yield,
        ))
        idx += 1

    for i in range(N_RATE):
        currency = "USD" if idx in domestic_idx else rng.choice(CURRENCIES)
        notional = round(rng.uniform(500_000.0, 5_000_000.0), 2)
        y0 = round(rng.uniform(0.02, 0.055), 5)
        duration = round(rng.uniform(2.5, 9.0), 3)
        coupon_rate = round(y0 - rng.uniform(-0.002, 0.002), 5)
        qty = 1.0
        instruments.append(Instrument(
            instrument_id=f"RT{i:02d}", asset_class="RATE", currency=currency, qty=qty,
            notional=notional, y0=y0, duration=duration, coupon_rate=coupon_rate,
        ))
        idx += 1

    for i in range(N_CREDIT):
        currency = "USD" if idx in domestic_idx else rng.choice(CURRENCIES)
        notional = round(rng.uniform(500_000.0, 5_000_000.0), 2)
        y0 = round(rng.uniform(0.02, 0.05), 5)
        spread0 = round(rng.uniform(0.008, 0.045), 5)
        duration = round(rng.uniform(2.0, 7.0), 3)
        coupon_rate = round(y0 + spread0 - rng.uniform(-0.002, 0.002), 5)
        qty = 1.0
        instruments.append(Instrument(
            instrument_id=f"CR{i:02d}", asset_class="CREDIT", currency=currency, qty=qty,
            notional=notional, y0=y0, spread0=spread0, duration=duration, coupon_rate=coupon_rate,
        ))
        idx += 1

    return instruments


def bond_price(notional: float, y_eff: float, duration: float) -> float:
    """Closed-form, exactly-repriced bond value: notional * exp(-y_eff *
    duration). This is the TRUE nonlinear reval function the duration-linear
    attribution bucket below is only a first-order approximation of; the gap
    between them is genuine, honestly measured residual, the same role
    gap/convexity residual plays everywhere else P&L is Taylor-decomposed in
    this repository."""
    return notional * math.exp(-y_eff * duration)


def build_sessions(seed: int, index: int, instrument: Instrument, n_sessions: int = 250) -> list[SessionState]:
    rng = random.Random(derive_seed(seed, index, "multi_asset_sessions"))

    fx_vol = 0.0 if instrument.currency == "USD" else round(rng.uniform(0.07, 0.13), 4)
    fx_drift = 0.0 if instrument.currency == "USD" else rng.uniform(-0.01, 0.01)
    fx = 1.0 if instrument.currency == "USD" else round(rng.uniform(0.8, 1.6), 4)

    sessions: list[SessionState] = []

    if instrument.asset_class == "EQ":
        price = instrument.s0
        for session_id in range(n_sessions):
            price_sod = price
            fx_sod = fx
            z = rng.gauss(0.0, 1.0)
            price_eod = price_sod * math.exp(
                (0.04 - 0.5 * instrument.eq_vol ** 2) * DT_YEARS
                + instrument.eq_vol * math.sqrt(DT_YEARS) * z
            )
            if instrument.currency == "USD":
                fx_eod = 1.0
            else:
                zf = rng.gauss(0.0, 1.0)
                fx_eod = fx_sod * math.exp(
                    (fx_drift - 0.5 * fx_vol ** 2) * DT_YEARS + fx_vol * math.sqrt(DT_YEARS) * zf
                )
            carry_local = instrument.div_yield / TRADING_DAYS_PER_YEAR * price_sod
            sessions.append(SessionState(
                session_id=session_id, price_sod=price_sod, price_eod=price_eod,
                y_sod=0.0, y_eod=0.0, spread_sod=0.0, spread_eod=0.0,
                carry_local=carry_local, fx_sod=fx_sod, fx_eod=fx_eod,
            ))
            price, fx = price_eod, fx_eod
        return sessions

    # RATE and CREDIT: evolve yield (and spread, for CREDIT) as bounded random walks.
    y = instrument.y0
    spread = instrument.spread0 if instrument.asset_class == "CREDIT" else 0.0
    yield_vol_daily = 0.0009
    spread_vol_daily = 0.0006

    for session_id in range(n_sessions):
        y_sod, spread_sod, fx_sod = y, spread, fx
        y_eod = min(max(y_sod + rng.gauss(0.0, yield_vol_daily), 0.001), 0.15)
        if instrument.asset_class == "CREDIT":
            spread_eod = min(max(spread_sod + rng.gauss(0.0, spread_vol_daily), 0.0005), 0.20)
        else:
            spread_eod = 0.0

        if instrument.currency == "USD":
            fx_eod = 1.0
        else:
            zf = rng.gauss(0.0, 1.0)
            fx_eod = fx_sod * math.exp(
                (fx_drift - 0.5 * fx_vol ** 2) * DT_YEARS + fx_vol * math.sqrt(DT_YEARS) * zf
            )

        price_sod = bond_price(instrument.notional, y_sod + spread_sod, instrument.duration)
        price_eod = bond_price(instrument.notional, y_eod + spread_eod, instrument.duration)
        carry_local = instrument.coupon_rate / TRADING_DAYS_PER_YEAR * instrument.notional

        sessions.append(SessionState(
            session_id=session_id, price_sod=price_sod, price_eod=price_eod,
            y_sod=y_sod, y_eod=y_eod, spread_sod=spread_sod, spread_eod=spread_eod,
            carry_local=carry_local, fx_sod=fx_sod, fx_eod=fx_eod,
        ))
        y, spread, fx = y_eod, spread_eod, fx_eod

    return sessions
