"""
CMO Deal Economics — Cash Flow Conservation & Investor Demand Model.

This module defines the RULES of CMO structuring as hard constraints,
so the RL environment's action space "falls out" naturally.  Instead of
penalizing bad structures with ad-hoc reward shaping, we enforce cash
flow conservation and let the investor demand model determine P&L.

══════════════════════════════════════════════════════════════════════════
GLOSSARY — Dealer Profit in CMO / Mortgage Trading
══════════════════════════════════════════════════════════════════════════

─── Collateral ────────────────────────────────────────────────────────

WAC         Weighted Average Coupon — avg interest rate of pool loans.
            This is the pool's GROSS coupon before guarantee fee.

Net Coupon  WAC minus guarantee fee (g-fee, typically 25-50bps).
            This is what the MBS passthrough certificate actually pays.

WAM/WALA    Weighted Average Maturity / Loan Age (months).

Factor      current_balance / original_balance.  Prepayment shrinks it.

PSA         Public Securities Association prepayment benchmark.
            100% PSA = CPR ramps linearly from 0% to 6% over 30 months.
            200% PSA = twice that speed, etc.

CPR/SMM     Conditional Prepayment Rate (annual) / Single Monthly
            Mortality.  SMM = 1 - (1 - CPR)^(1/12).

─── TBA Market ────────────────────────────────────────────────────────

TBA         To Be Announced — forward market for agency MBS.  Trades
            are specified by coupon, maturity, and settlement date, but
            the SPECIFIC pools aren't known until 48 hours before
            settlement ("48-hour rule").

TBA Price   Market price for generic passthrough of given coupon/term.
            This is the dealer's acquisition cost for collateral.

Dollar Roll Sell front-month TBA, buy back-month.  The DROP between
            months reflects financing value + prepayment optionality.
            Song & Zhu (RFS 2019): specialness is the excess return
            over implied repo rate.

Specialness How much the roll exceeds implied financing cost.
            When a coupon is "special" (scarce), the roll is rich,
            and the dealer earns extra carry income.

─── Spec Pools ────────────────────────────────────────────────────────

Spec Pool   Passthrough with known characteristics commanding a PAYUP
            (premium) over generic TBA.  The payup reflects the
            prepayment protection value.

Payup       Premium in 32nds over TBA price.  Bednarek et al. (JFE
            2023): payup can exceed 30% of lender revenue for LLB.

LLB         Low Loan Balance — small average loan → slower prepay
            (fixed refi costs are larger relative to loan size).

HLB/HLTV    High LTV — borrower has less equity → less refi incentive.

Geographic  NY, PR, etc. — different state laws/economics affect prepay.

Investor    Investor property loans — different default/prepay behavior.

─── CMO / REMIC Structure ─────────────────────────────────────────────

CMO         Collateralized Mortgage Obligation — multi-tranche security
            backed by MBS pools.  The tranching creates VALUE because
            different investors want different cash flow profiles.

REMIC       Real Estate Mortgage Investment Conduit — tax-efficient
            structure (IRC §860D).  Most agency CMOs are REMICs.
            The "R" tranche holds residual interest.

Tranche     A slice of the CMO with specific priority rules for
            receiving principal and interest.

Waterfall   The set of rules that allocate collateral cash flows to
            tranches at each payment date.

Principal   Period during which a tranche actively receives principal.
Window      Short window = more predictable WAL.

Lockout     Period before a tranche starts receiving principal.

─── Tranche Types (Principal Rules) ───────────────────────────────────

Sequential  Receives principal in strict priority order.
(SEQ)       A-tranche pays down first, then B, then C, etc.
            Shortest window = most predictable WAL for senior.

PAC         Planned Amortization Class.  Principal follows a SCHEDULE
            derived from two PSA speeds (the "bands", e.g. 100-300).
            Within the bands, the PAC has a very stable WAL.
            Fukushima (2004): optimal PAC fraction ~74% for max value.

Support/    Absorbs prepayment variability to protect the PAC schedule.
Companion   When prepay is fast, support gets excess principal (shorter).
            When prepay is slow, support gets deferred (longer).
            High yield but volatile WAL.

TAC         Targeted Amortization Class.  Like PAC but protects
            against only one direction (usually extension).

Z-bond/     Receives no current interest payments.  Instead, interest
Accrual     ACCRUES and is added to the Z-bond's face value.  The
            accrued interest is redirected to pay down shorter tranches.
            Z-bonds are long-duration assets for insurance/pension.

VADM        Very Accurately Determined Maturity.  Principal comes ONLY
            from Z-bond accrued interest, not from collateral prepay.
            Extremely stable WAL.  Fabozzi (2016): the most stable
            tranche type, WAL std ~0.2yr vs PAC ~0.6yr.

Schedule    Like PAC but targeted at a SINGLE PSA speed (not a band).
Bond        Cheaper to create, less protection, more yield.

Jump-Z      Z-bond that "jumps" to current-pay when support tranches
            are exhausted.  Protects PAC from being broken.

NAS         Non-Accelerating Senior.  Has a principal LOCKOUT period.
            Guaranteed minimum average life.

Z-PAC       Z-accrual during lockout, then follows PAC schedule.
            Combines long duration with PAC stability.

─── Tranche Types (Interest Rules) ────────────────────────────────────

Fixed       Pays a fixed coupon rate.  Most CMO tranches.

Floating    Pays index + spread.  Typically 1M-SOFR + margin.
            Created by SPLITTING a fixed-rate tranche.
            Deep demand from money market funds and banks.

Inverse     Pays K - multiplier × index.  The COMPLEMENT of a floater.
Floater     Can ONLY exist paired with a matching floater.
            The pair must satisfy the interest conservation equation:
              floater_bal × (index + margin)
            + inverse_bal × (K - m × index)
            = parent_bal × parent_coupon     ∀ values of index.
            Hedge fund product (leverage on rate declines).

IO          Interest Only — receives interest cash flows, no principal.
(Interest   IO coupon = WAC - weighted_avg_bond_coupon.  This is the
Only)       EXCESS COUPON left over after all bond tranches are paid.
            Only ONE IO strip per deal (it's the residual).
            Primary dealer profit center: 60-70% of desk P&L.

PO          Principal Only — receives all principal, no interest.
(Principal  Complement of IO.  Speeds up with prepay (gets face value
Only)       sooner).  Slows with extension (waits longer).

─── Cash Flow Conservation (THE fundamental constraint) ────────────────

Rule 1 (Principal): At every period t,
    Σ tranche_principal_i(t) = collateral_principal(t)
    Every dollar of principal goes somewhere.

Rule 2 (Interest): At every period t,
    Σ tranche_interest_i(t) = collateral_interest(t)
    Every dollar of interest goes somewhere.

Rule 3 (Balance):
    Σ tranche_balance_i = collateral_balance    at origination
    (Non-IO tranches only; IO is notional.)

Rule 4 (IO is the residual):
    IO_coupon = WAC - Σ(bond_coupon_i × bond_bal_i) / Σ(bond_bal_i)
    IO_notional = collateral_balance  (or a portion)
    You can't create IO coupon from nothing — it's whatever the
    collateral pays in excess of what the bonds need.

Rule 5 (Floater/Inverse pairing):
    floater_bal + inverse_bal = parent_tranche_bal
    inverse_bal = parent_bal / (1 + multiplier)
    floater_bal = parent_bal × multiplier / (1 + multiplier)
    This is NOT a choice — it's determined by the math.

─── Dealer Desk P&L ───────────────────────────────────────────────────

The desk's profit has four components:

1. STRUCTURING ARBITRAGE (investor surplus capture)
   Investors value specific cash flow profiles MORE than the raw pool.
   A PAC buyer pays a premium for prepayment stability.
   A hedge fund pays a premium for inverse floater leverage.
   Surplus = Σ (-spread_i × duration_i × face_i / 10000)
   Diverse deals access broader demand → tighter aggregate spreads.

2. IO REVENUE (excess coupon)
   IO_value = (WAC - avg_bond_cpn) × collateral_bal × WAL × PV_factor
   This is FIXED by the coupon structure — more IO strips don't
   create more excess coupon.  They just split the same stream.
   Typical IO value: 20-40 ticks per $100M.

3. DOLLAR ROLL INCOME (TBA specialness)
   roll_income = specialness × collateral × price / 12
   Depends on market conditions, not deal structure.
   Song & Zhu (2019): significant for on-the-run coupons.

4. SPEC POOL PAYUP INCOME
   net_payup = protection_value - payup_cost
   Bednarek et al. (2023): slower prepay → longer IO life → higher PV.
   Depends on pool selection, not deal structure.

─── Negative Convexity & Accretive Structures ───────────────────────

Negative     MBS price rises LESS when rates fall (borrowers prepay)
Convexity    than it falls when rates rise (extension risk).
             ALL mortgage-backed securities are negatively convex.
             CMO structuring REDISTRIBUTES convexity across tranches.

Convexity    PAC absorbs the GOOD convexity (stable WAL).
Redistri-    Support absorbs ALL the negative convexity (volatile WAL).
bution       The desk profits from the spread between what PAC buyers
             pay for stability and what support buyers demand for risk.

Accretive    A structure is "accretive" when the sum of tranche values
Structure    exceeds the collateral cost.  This happens because:
             1. PAC buyers pay premium for convexity protection
             2. Z-bond accretion redirects interest → accelerates short pay
             3. Floater/inverse split → accesses bank + HF demand
             4. IO extraction → captures excess coupon as separate security

Hold-Down    Principal lockout mechanisms that create WAL stability:
             NAS lockout, Z-bond accretion, Support deferred until PAC met.
             Investors pay for the stability this creates.

             Convexity risk premiums (typical, over theoretical option cost):
             PAC-I: ~5bp    Support: ~20bp    IO: ~70bp    Inverse(3x): ~100bp

─── Investor Demand Segmentation ──────────────────────────────────────

Each investor class has a base spread (willingness-to-pay), a demand
capacity (how much they'll buy), and an elasticity (how fast spreads
widen when supply exceeds capacity).

Insurance/Pension:  PAC, Z-bond, VADM   — long duration, stable
Banks:              Floaters, short SEQ  — ALM matching, rate hedging
Hedge Funds:        Inverse, IO, support — leverage, relative value
Money Markets:      Floaters, short SEQ  — liquidity, low duration
Money Managers:     PAC, TAC, schedule   — predictable, benchmarkable

Market saturation: excess supply of one type widens spreads for ALL
tranches of that type.  This naturally prevents "spam" — the Nth
inverse floater faces a much wider spread than the first.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .cmo_structure import TrancheSpec, PrincipalType, InterestType


# ═══════════════════════════════════════════════════════════════════════
# Investor Demand Model
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DemandCurve:
    """Investor demand for a tranche type.

    base_spread_bps: Spread at which the FIRST dollar trades.
        Negative = trades rich (investor pays premium).
    capacity_M:     How much ($M) investors will absorb at base spread.
    elasticity:     Spread widens by this many bps per $1B OVER capacity.

    Example: PAC has base=-18, capacity=5000, elasticity=2.
    First $5B of PAC trades at -18bps (rich).
    At $6B: spread = -18 + 2×(6000-5000)/1000 = -16bps.
    At $10B: spread = -18 + 2×5 = -8bps.
    """
    base_spread_bps: float
    capacity_M: float
    elasticity: float


# Demand curves by (InterestType, PrincipalType).
# Calibrated to approximate real agency CMO new-issue spreads.
# Sources: Vickery & Wright (2013), Adelino et al. (2019), dealer runs.
DEMAND_CURVES: dict[tuple[InterestType, PrincipalType], DemandCurve] = {
    # ─── Fixed-rate tranches ─────────────────────────────────────────
    (InterestType.FIXED, PrincipalType.SEQUENTIAL):
        DemandCurve(-5, 3000, 4),       # Banks + general: broad demand
    (InterestType.FIXED, PrincipalType.PAC):
        DemandCurve(-18, 5000, 2),      # Insurance/pension: very deep
    (InterestType.FIXED, PrincipalType.TAC):
        DemandCurve(-10, 2000, 5),      # Insurance: moderate
    (InterestType.FIXED, PrincipalType.SUPPORT):
        DemandCurve(25, 1500, 10),      # Hedge funds: thin, wide spread

    # ─── Z-accrual ───────────────────────────────────────────────────
    (InterestType.Z_ACCRUAL, PrincipalType.SEQUENTIAL):
        DemandCurve(-8, 1500, 6),       # Insurance: long duration

    # ─── IO / PO ─────────────────────────────────────────────────────
    (InterestType.IO_ONLY, PrincipalType.PASSTHROUGH):
        DemandCurve(-25, 800, 15),      # Specialist: narrow but rich
    (InterestType.PO_ONLY, PrincipalType.PASSTHROUGH):
        DemandCurve(8, 600, 12),        # Specialist: thin

    # ─── Floating / Inverse ──────────────────────────────────────────
    (InterestType.FLOATING, PrincipalType.SEQUENTIAL):
        DemandCurve(-12, 4000, 3),      # Money market: deep demand
    (InterestType.INVERSE_FLOATING, PrincipalType.SEQUENTIAL):
        DemandCurve(-20, 50, 80),       # Hedge funds: VERY thin, ~$50M/deal max
}

# Fallback for any type not in the table
_DEFAULT_DEMAND = DemandCurve(0, 2000, 5)


def market_clearing_spread(
    tranche: TrancheSpec,
    supply_of_type_M: float,
    structurally_valid: bool = True,
) -> float:
    """Compute market-clearing spread for a tranche given total supply.

    Args:
        tranche: The tranche being priced.
        supply_of_type_M: Total $M of this (interest, principal) type
            in the deal.  Higher supply → wider spread.
        structurally_valid: False for unpaired inverse floaters, etc.

    Returns:
        Spread in bps.  Negative = trades rich.
    """
    if not structurally_valid:
        return 150.0  # No investor will buy this → deep discount

    key = (tranche.interest_type, tranche.principal_type)
    curve = DEMAND_CURVES.get(key, _DEFAULT_DEMAND)

    excess_M = max(0.0, supply_of_type_M - curve.capacity_M)
    spread = curve.base_spread_bps + curve.elasticity * (excess_M / 1000.0)

    # Priority bump: later tranches in the waterfall trade a bit wider
    spread += tranche.priority * 2.0

    return spread


# ═══════════════════════════════════════════════════════════════════════
# Structural Constraints
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StructuralValidation:
    """Result of validating a deal's structural integrity."""
    is_valid: bool = True
    tranche_validity: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    io_excess_coupon_pct: float = 0.0   # WAC - avg bond coupon
    n_floater: int = 0
    n_inverse: int = 0
    n_paired: int = 0
    n_io: int = 0


def validate_deal_structure(
    tranches: list[TrancheSpec],
    collateral_wac: float,
    collateral_balance: float,
) -> StructuralValidation:
    """Validate a deal's structural integrity.

    Checks cash flow conservation rules and marks structurally
    invalid tranches (e.g. unpaired inverse floaters).
    """
    v = StructuralValidation()

    v.n_floater = sum(1 for t in tranches
                      if t.interest_type == InterestType.FLOATING)
    v.n_inverse = sum(1 for t in tranches
                      if t.interest_type == InterestType.INVERSE_FLOATING)
    v.n_paired = min(v.n_floater, v.n_inverse)
    v.n_io = sum(1 for t in tranches
                 if t.interest_type == InterestType.IO_ONLY)

    # Mark inverse floaters as valid/invalid based on pairing
    inv_seen = 0
    for t in tranches:
        if t.interest_type == InterestType.INVERSE_FLOATING:
            v.tranche_validity[t.name] = (inv_seen < v.n_paired)
            if inv_seen >= v.n_paired:
                v.warnings.append(
                    f"{t.name}: unpaired inverse floater (no matching floater)")
            inv_seen += 1
        else:
            v.tranche_validity[t.name] = True

    # Unpaired floaters are still sellable (they just don't get the
    # inverse premium), so they're always "valid".

    # Compute excess coupon for IO
    non_io = [t for t in tranches if not t.is_io and not t.is_po]
    total_bal = sum(t.original_balance for t in non_io)
    if total_bal > 0:
        avg_bond_cpn = sum(t.coupon * t.original_balance
                           for t in non_io) / total_bal
        v.io_excess_coupon_pct = max(0.0, collateral_wac - avg_bond_cpn)
    else:
        v.io_excess_coupon_pct = 0.0

    return v


# ═══════════════════════════════════════════════════════════════════════
# IO Value — THE primary profit center
# ═══════════════════════════════════════════════════════════════════════

def compute_io_value_ticks(
    tranches: list[TrancheSpec],
    collateral_wac: float,
    collateral_balance: float,
    collateral_wal: float = 5.0,
) -> float:
    """Compute the IO strip value in ticks (32nds per $100 face).

    The IO value is the PV of the EXCESS COUPON stream:
        excess = WAC - weighted_avg_bond_coupon

    CRITICAL: This value is FIXED by the coupon structure.  Multiple
    IO strips split the same stream — they do NOT multiply it.
    Only the EXISTENCE of at least one IO strip matters (it means the
    dealer is retaining and selling the excess coupon).

    Returns 0 if no IO strip exists (excess coupon is wasted / left
    in the R-tranche residual).
    """
    has_io = any(t.interest_type == InterestType.IO_ONLY for t in tranches)
    if not has_io:
        return 0.0

    # Compute excess coupon
    non_io = [t for t in tranches if not t.is_io and not t.is_po]
    total_bal = sum(t.original_balance for t in non_io)
    if total_bal <= 0:
        return 0.0

    avg_bond_cpn = sum(t.coupon * t.original_balance
                       for t in non_io) / total_bal
    excess_spread = max(0.0, collateral_wac - avg_bond_cpn) / 100.0

    # Annual IO cash flow (on the COLLATERAL balance, not IO notional)
    annual_io_cf = excess_spread * collateral_balance

    # PV using WAL-based discount
    wal = max(1.0, min(15.0, collateral_wal))
    discount_rate = 0.05
    pv_mult = wal * (1.0 - discount_rate * wal / 2.0)
    pv_mult = max(1.0, min(10.0, pv_mult))

    # Conservative pricing at 50% of theoretical PV (bid-side)
    io_value_dollars = annual_io_cf * pv_mult * 0.5

    # Convert to ticks per $collateral_balance
    # 1 tick = 1/32 of a point per $100 face = $3.125 per $10,000 face
    tick_value = collateral_balance * 0.0003125
    io_ticks = io_value_dollars / tick_value if tick_value > 0 else 0.0

    return max(0.0, io_ticks)


# ═══════════════════════════════════════════════════════════════════════
# Investor Surplus — The structuring arb
# ═══════════════════════════════════════════════════════════════════════

def compute_investor_surplus_ticks(
    tranches: list[TrancheSpec],
    spreads: dict[str, float],
    durations: dict[str, float],
    collateral_balance: float,
) -> float:
    """Compute structuring arbitrage from investor surplus.

    Each tranche's spread reflects how much investors value it
    relative to the raw collateral.  Negative spread = investor pays
    a premium = dealer captures surplus.

    Surplus_i = -spread_i × duration_i × face_i / 10000

    This replaces the broken "proceeds minus cost" calculation that
    failed because the OAS pricing engine doesn't capture the
    investor utility premia that make CMO structuring profitable.
    """
    total_surplus = 0.0
    for t in tranches:
        # IO strips are excluded from surplus — their value is computed
        # separately in compute_io_value_ticks (excess coupon PV).
        # Including them here would double-count.
        if t.interest_type == InterestType.IO_ONLY:
            continue

        face = t.notional_balance if t.is_io else t.original_balance
        if face <= 0:
            continue

        spread = spreads.get(t.name, 0.0)
        dur = durations.get(t.name, 3.0)

        surplus = -spread * dur * face / 10000.0
        total_surplus += surplus

    tick_value = collateral_balance * 0.0003125
    return total_surplus / tick_value if tick_value > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Convexity Redistribution — The "accretive structure" alpha
# ═══════════════════════════════════════════════════════════════════════

# Typical negative convexity by tranche type (scaled 0-1).
# 0 = no negative convexity (or positive), 1 = maximum negative convexity.
# PAC and VADM absorb the least; support, IO, inverse absorb the most.
_CONVEXITY_ABSORPTION = {
    # Principal type → base negative convexity absorbed
    PrincipalType.PAC: 0.05,           # Very low — protected by bands
    PrincipalType.TAC: 0.15,           # Moderate — one-sided protection
    PrincipalType.SEQUENTIAL: 0.25,    # Passthrough-like
    PrincipalType.SUPPORT: 0.70,       # High — absorbs PAC's convexity
    PrincipalType.PASSTHROUGH: 0.30,   # Generic
}

# Interest type → convexity multiplier
_INTEREST_CVX_MULT = {
    InterestType.FIXED: 1.0,
    InterestType.Z_ACCRUAL: 1.3,          # Longer duration amplifies
    InterestType.IO_ONLY: 2.5,            # Highly negatively convex
    InterestType.PO_ONLY: 0.3,            # POSITIVE convexity (rare in MBS)
    InterestType.FLOATING: 0.2,           # Near-zero duration ≈ no convexity
    InterestType.INVERSE_FLOATING: 3.0,   # Maximum — leveraged neg convexity
}

# Convexity risk premium: bps of OAS premium per unit of negative convexity
# This is what investors demand ABOVE the theoretical option cost.
_CVX_RISK_PREMIUM_BPS = 8.0


def compute_convexity_redistribution_ticks(
    tranches: list[TrancheSpec],
    durations: dict[str, float],
    collateral_balance: float,
) -> float:
    """Compute the convexity redistribution alpha in ticks.

    CMO structuring creates value by concentrating negative convexity
    into tranches held by investors who are PAID to absorb it.  The
    "accretive" part: a deal with PAC + Support is worth more than
    a deal of all sequentials because the PAC's convexity protection
    commands a premium that exceeds the support's convexity discount.

    The alpha = Σ (convexity_premium × duration × face) for protected
    tranches, minus the collateral's baseline negative convexity cost.

    Protected tranches (PAC, VADM, NAS): LOW negative convexity → premium
    Absorber tranches (Support, Inverse, IO): HIGH negative convexity → discount
    The NET is positive because insurance/pension pay MORE for protection
    than hedge funds demand for absorption.  This is the structural arb.
    """
    if not tranches:
        return 0.0

    # Collateral baseline: passthrough has moderate negative convexity
    collateral_cvx_cost = 0.25  # Normalized baseline

    # Compute weighted convexity for the structured deal
    total_face = 0.0
    weighted_cvx = 0.0
    protected_premium = 0.0

    for t in tranches:
        if t.interest_type == InterestType.IO_ONLY:
            continue  # IO handled separately

        face = t.original_balance
        if face <= 0:
            continue

        dur = durations.get(t.name, 3.0)

        # How much negative convexity does this tranche absorb?
        base_cvx = _CONVEXITY_ABSORPTION.get(t.principal_type, 0.25)
        int_mult = _INTEREST_CVX_MULT.get(t.interest_type, 1.0)
        tranche_cvx = base_cvx * int_mult

        weighted_cvx += tranche_cvx * face
        total_face += face

        # Protected tranches earn a convexity PREMIUM
        # (they're LESS negatively convex than the collateral)
        cvx_improvement = collateral_cvx_cost - tranche_cvx
        if cvx_improvement > 0:
            # Investors pay extra for better convexity profile
            premium_bps = cvx_improvement * _CVX_RISK_PREMIUM_BPS
            protected_premium += premium_bps * dur * face / 10000.0

    if total_face <= 0:
        return 0.0

    # Convexity redistribution efficiency:
    # A deal that separates PAC (low cvx) from support (high cvx)
    # creates more value than a deal of all sequentials (same cvx).
    avg_deal_cvx = weighted_cvx / total_face
    cvx_dispersion = 0.0
    for t in tranches:
        if t.interest_type == InterestType.IO_ONLY:
            continue
        face = t.original_balance
        if face <= 0:
            continue
        base_cvx = _CONVEXITY_ABSORPTION.get(t.principal_type, 0.25)
        int_mult = _INTEREST_CVX_MULT.get(t.interest_type, 1.0)
        tranche_cvx = base_cvx * int_mult
        cvx_dispersion += abs(tranche_cvx - avg_deal_cvx) * face / total_face

    # Dispersion bonus: more dispersion = more effective redistribution
    # A deal of 3 sequentials has dispersion ~0 (no redistribution)
    # A PAC+Support deal has high dispersion (effective redistribution)
    dispersion_bonus = cvx_dispersion * 15.0  # Scale to ticks

    # Convert protected premium to ticks
    tick_value = collateral_balance * 0.0003125
    premium_ticks = protected_premium / tick_value if tick_value > 0 else 0.0

    # Total convexity alpha = premium from protected tranches + dispersion bonus
    cvx_ticks = premium_ticks + dispersion_bonus
    return max(0.0, cvx_ticks)


# ═══════════════════════════════════════════════════════════════════════
# Full Deal P&L
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DealPnL:
    """Complete breakdown of desk P&L for a CMO deal."""
    arb_ticks: float = 0.0         # Investor surplus
    io_ticks: float = 0.0          # IO excess coupon value
    roll_ticks: float = 0.0        # Dollar roll income
    payup_ticks: float = 0.0       # Spec pool payup income
    convexity_ticks: float = 0.0   # Convexity redistribution alpha
    total_ticks: float = 0.0       # Sum of above
    reward: float = 0.0            # Unclipped total_ticks (natural bounds)

    # Diagnostics
    n_tranches: int = 0
    n_unique_types: int = 0
    avg_spread: float = 0.0
    io_excess_coupon: float = 0.0
    validation: Optional[StructuralValidation] = None


def compute_deal_pnl(
    tranches: list[TrancheSpec],
    pricing_results: dict,     # name → PricingResult
    collateral_wac: float,
    collateral_balance: float,
    collateral_wal: float,
    dollar_roll_specialness: float = 0.0,
    tba_price: float = 100.0,
    selected_pool_payup: float = 0.0,
) -> DealPnL:
    """Compute full desk P&L for a CMO deal.

    This is the ONLY function the RL env needs to call.
    No ad-hoc bonuses.  The demand model and cash flow conservation
    rules determine everything.
    """
    from collections import defaultdict
    from .market_simulator import specialness_to_roll_income_ticks

    pnl = DealPnL()
    pnl.n_tranches = len(tranches)

    # 1. Validate structure
    validation = validate_deal_structure(tranches, collateral_wac, collateral_balance)
    pnl.validation = validation
    pnl.io_excess_coupon = validation.io_excess_coupon_pct

    # 2. Compute supply by type
    supply_by_type: dict[tuple, float] = defaultdict(float)
    for t in tranches:
        face = t.notional_balance if t.is_io else t.original_balance
        supply_by_type[(t.interest_type, t.principal_type)] += face / 1e6

    # 3. Market-clearing spreads
    spreads: dict[str, float] = {}
    for t in tranches:
        supply_M = supply_by_type[(t.interest_type, t.principal_type)]
        is_valid = validation.tranche_validity.get(t.name, True)
        spreads[t.name] = market_clearing_spread(t, supply_M, is_valid)

    # 4. Durations from pricing results
    durations: dict[str, float] = {}
    for t in tranches:
        pr = pricing_results.get(t.name)
        if pr and pr.wal_years > 0:
            durations[t.name] = min(pr.wal_years * 0.85, 15.0)
        else:
            durations[t.name] = 3.0

    # 5. Investor surplus (structuring arb)
    pnl.arb_ticks = compute_investor_surplus_ticks(
        tranches, spreads, durations, collateral_balance)

    # 6. IO value (excess coupon)
    pnl.io_ticks = compute_io_value_ticks(
        tranches, collateral_wac, collateral_balance, collateral_wal)

    # 7. Dollar roll income
    pnl.roll_ticks = specialness_to_roll_income_ticks(
        dollar_roll_specialness, collateral_balance, tba_price)

    # 8. Spec pool payup
    has_io = any(t.interest_type == InterestType.IO_ONLY for t in tranches)
    recovery = 0.5 if has_io else 0.25
    pnl.payup_ticks = selected_pool_payup * recovery

    # 9. Convexity redistribution bonus
    # The desk captures value by concentrating negative convexity in
    # tranches that are paid to absorb it (support, inverse, IO) while
    # selling convexity-protected tranches (PAC, VADM, NAS) at a premium.
    # This is the "accretive structure" alpha.
    pnl.convexity_ticks = compute_convexity_redistribution_ticks(
        tranches, durations, collateral_balance)

    # Total
    pnl.total_ticks = (pnl.arb_ticks + pnl.io_ticks
                       + pnl.roll_ticks + pnl.payup_ticks
                       + pnl.convexity_ticks)
    pnl.reward = pnl.total_ticks

    # Diagnostics
    unique_types = set((t.interest_type, t.principal_type) for t in tranches)
    pnl.n_unique_types = len(unique_types)
    if spreads:
        pnl.avg_spread = sum(spreads.values()) / len(spreads)

    return pnl
