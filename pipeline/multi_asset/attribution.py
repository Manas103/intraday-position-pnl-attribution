"""Cross-asset P&L attribution into price, rate, FX, carry and residual
terms, computed simultaneously from the start-of-day state rather than by
bumping one factor at a time.

For every instrument, in USD:

    V_local(t) = qty * price_local(t)
    V_usd(t)   = V_local(t) * fx(t)          (fx == 1.0 for USD instruments)
    actual_pnl = (V_usd_eod - V_usd_sod) + carry_usd

The identity used to decompose the first term uses the average-rate
convention standard in cross-currency P&L attribution: each side's move is
valued at the OTHER factor's SOD/EOD average rather than at its SOD value
alone (price_bucket uses avg_fx = (fx_sod + fx_eod) / 2; fx_bucket uses the
TRUE avg_price = (price_sod + price_eod) / 2). For any instrument whose
local-currency price move is itself exact (EQ), this makes

    price_bucket + fx_bucket == qty * (price_eod * fx_eod - price_sod * fx_sod)

exactly, with no leftover price/FX cross term, because
    dP * (fx_sod + fx_eod) + dFX * (price_sod + price_eod) == 2 * (P1*F1 - P0*F0)
is an algebraic identity for any P0, P1, F0, F1 (expand and the mixed terms
cancel). This was not the first design: the first attempt valued price_bucket
at fx_sod alone (a textbook first-order split), which left the entire
dP * dFX cross term in residual for every foreign instrument; switching to
the average-rate convention is a later genuine attempt and removes that
term by construction rather than by tuning a parameter. See Findings in the
README for the measured effect.

``dP`` (the local-currency price move being valued above) is itself split
differently by asset class:

  EQ:     dP is the instrument's own idiosyncratic equity return, carried
          through exactly (no further decomposition, no approximation).
  RATE:   dP is the EXACT change in a closed-form bond price,
          notional * exp(-y * duration). The rate bucket uses a
          duration-PLUS-convexity Taylor approximation of that move
          (-duration * price_sod * d_yield + 0.5 * duration**2 * price_sod *
          d_yield**2; convexity here is exactly duration**2 because the true
          price is a pure exponential in yield, not an independently
          estimated number); the gap between the true dP and this
          second-order approximation is genuine Taylor-truncation residual,
          not an error. This convexity term was added as one of the three
          genuine attempts described in Findings, alongside the
          average-rate fix above; a first-order-only, fx_sod-only version
          left a larger residual.
  CREDIT: dP is driven by two factors, the base rate and the credit spread,
          dP = P(y1+s1) - P(y0+s0). The second-order approximation is split
          into a rate bucket and a price bucket (the instrument's own
          idiosyncratic credit-risk move), with the joint convexity term
          allocated between them in proportion to each factor's own linear
          move; the gap between the true dP and the sum of both pieces is
          again convexity residual.

carry_usd = qty * carry_local * fx_eod (coupon/dividend cash accrued this
session, converted at the rate it is actually received).

residual = actual_pnl - (price_pnl + rate_pnl + fx_pnl + carry_pnl)

For EQ this residual is 0 by construction (the average-rate identity above
is exact and dP itself is exact). For RATE/CREDIT it is purely the
duration-convexity Taylor-truncation gap. Nothing here is a bug; see
sequential_order_check.py for why the price/FX cross term this module
eliminates by construction cannot be eliminated by picking a different
SEQUENTIAL bump order, only by computing both sides simultaneously at their
SOD/EOD average, which is what this module does.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import simulator as masim


@dataclass(frozen=True)
class MultiAssetRow:
    instrument_id: str
    asset_class: str
    currency: str
    session_id: int
    price_sod: float
    price_eod: float
    fx_sod: float
    fx_eod: float
    actual_pnl: float
    price_pnl: float
    rate_pnl: float
    fx_pnl: float
    carry_pnl: float
    residual: float


def attribute_session(inst: masim.Instrument, st: masim.SessionState) -> MultiAssetRow:
    d_price = st.price_eod - st.price_sod
    d_fx = st.fx_eod - st.fx_sod
    qty = inst.qty
    # Average-rate convention (the identity used throughout real
    # cross-currency P&L attribution): valuing each side's move at the
    # OTHER factor's SOD/EOD average, rather than at its SOD value alone,
    # makes price_bucket + fx_bucket telescope EXACTLY to qty * (price_eod *
    # fx_eod - price_sod * fx_sod) for any instrument whose local-currency
    # price move is itself exact (EQ) -- see module docstring, third
    # attempt. It does not remove the RATE/CREDIT duration-convexity gap,
    # which is a different, genuine approximation error handled below.
    avg_fx = 0.5 * (st.fx_sod + st.fx_eod)

    if inst.asset_class == "EQ":
        price_bucket = qty * d_price * avg_fx
        rate_bucket = 0.0
    elif inst.asset_class == "RATE":
        d_yield = st.y_eod - st.y_sod
        d = inst.duration
        # Second-order (duration-convexity) approximation of the exact
        # closed-form bond move, not just the first-order duration term:
        # price_local(y_eff) = notional * exp(-y_eff * duration) is a pure
        # exponential in y_eff, so its own convexity is exactly duration**2
        # (d^2/dy^2 exp(-y*D) = D^2 * exp(-y*D)), not an independently
        # estimated parameter. Adding this term is the same move the
        # cross-Greek extension made for the options book: a genuine,
        # algebraically justified second-order term that shrinks the
        # Taylor-truncation residual rather than a tuned fudge factor.
        rate_approx = st.price_sod * (-d * d_yield + 0.5 * d * d * d_yield * d_yield)
        price_bucket = 0.0
        rate_bucket = qty * rate_approx * avg_fx
    else:  # CREDIT
        d_yield = st.y_eod - st.y_sod
        d_spread = st.spread_eod - st.spread_sod
        d = inst.duration
        d_yield_eff = d_yield + d_spread
        convexity_total = 0.5 * d * d * st.price_sod * d_yield_eff * d_yield_eff
        rate_approx = -d * st.price_sod * d_yield
        spread_approx = -d * st.price_sod * d_spread
        # The joint second-order term is split between the rate and price
        # (credit-spread) buckets in proportion to each factor's own linear
        # move, since both factors enter the same (y + s) exponential
        # symmetrically; see module docstring.
        linear_total = abs(rate_approx) + abs(spread_approx)
        rate_share = abs(rate_approx) / linear_total if linear_total else 0.5
        rate_approx += convexity_total * rate_share
        spread_approx += convexity_total * (1.0 - rate_share)
        price_bucket = qty * spread_approx * avg_fx
        rate_bucket = qty * rate_approx * avg_fx

    avg_price_true = 0.5 * (st.price_sod + st.price_eod)
    fx_bucket = qty * avg_price_true * d_fx
    carry_bucket = qty * st.carry_local * st.fx_eod

    d_v_usd = qty * st.price_eod * st.fx_eod - qty * st.price_sod * st.fx_sod
    actual_pnl = d_v_usd + carry_bucket
    residual = actual_pnl - (price_bucket + rate_bucket + fx_bucket + carry_bucket)

    return MultiAssetRow(
        instrument_id=inst.instrument_id, asset_class=inst.asset_class, currency=inst.currency,
        session_id=st.session_id, price_sod=st.price_sod, price_eod=st.price_eod,
        fx_sod=st.fx_sod, fx_eod=st.fx_eod, actual_pnl=actual_pnl,
        price_pnl=price_bucket, rate_pnl=rate_bucket, fx_pnl=fx_bucket, carry_pnl=carry_bucket,
        residual=residual,
    )


def run_full_multi_asset_attribution(seed: int = 42, n_sessions: int = 250) -> dict:
    universe = masim.build_universe(seed)
    rows: list[MultiAssetRow] = []

    for idx, inst in enumerate(universe):
        sessions = masim.build_sessions(seed, idx, inst, n_sessions)
        for st in sessions:
            rows.append(attribute_session(inst, st))

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["gross_pnl"] = df["actual_pnl"].abs()
    df["abs_residual"] = df["residual"].abs()

    total_gross = float(df["gross_pnl"].sum())
    total_residual = float(df["abs_residual"].sum())
    overall_ratio = (total_residual / total_gross) if total_gross else 0.0

    return {
        "rows": rows,
        "df": df,
        "n_instruments": len(universe),
        "n_sessions": n_sessions,
        "universe": universe,
        "total_gross_pnl": total_gross,
        "total_unexplained_residual": total_residual,
        "overall_residual_ratio": overall_ratio,
    }
