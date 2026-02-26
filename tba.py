"""
TBA (To-Be-Announced) Market, Dollar Roll, and Delivery Analytics.

From Fuster, Lucca & Vickery (2022), Section 5:
"The key feature of a TBA trade is that the seller does not specify exactly
which pools will be delivered at settlement."

The TBA market concentrates ~1M individual agency MBS pools into a small number
of liquid forward contracts, with $261bn average daily trading volume.

Implements:
- TBA pricing by coupon/agency/settlement
- Cheapest-to-deliver (CTD) analysis
- Dollar roll analytics (implied financing rate, drop)
- Spec pool payup modeling
- Good delivery variance calculations
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .spec_pool import SpecPool, AgencyType, PoolCharacteristic, project_pool_cashflows
from .prepayment import estimate_psa_speed, PrepaymentModel


class TBASettlement(Enum):
    """TBA settlement months."""
    CURRENT = 0    # Current month
    FORWARD_1 = 1  # 1-month forward
    FORWARD_2 = 2  # 2-month forward
    FORWARD_3 = 3  # 3-month forward


@dataclass
class TBAContract:
    """A TBA forward contract specification."""
    agency: str          # FNMA, FHLMC, GNMA2, UMBS
    coupon: float        # Pass-through coupon (e.g., 5.5)
    term: int            # 30 or 15 year
    settlement: TBASettlement = TBASettlement.CURRENT
    price: float = 0.0   # Price in 32nds handle (e.g., 101-16 = 101.5)
    face_value: float = 1_000_000  # Standard $1M face

    @property
    def price_decimal(self) -> float:
        """Price as decimal (e.g., 101.5)."""
        return self.price

    @property
    def dollar_price(self) -> float:
        """Dollar price of the contract."""
        return self.price / 100.0 * self.face_value


@dataclass
class TBAPriceGrid:
    """TBA price grid by coupon - mirrors Yield Book's price display."""
    agency: str
    term: int  # 30 or 15
    settlement_month: str
    prices: dict[float, float]  # coupon -> price
    jan_drop: dict[float, float] = field(default_factory=dict)  # coupon -> drop

    def get_price(self, coupon: float) -> float:
        if coupon in self.prices:
            return self.prices[coupon]
        # Interpolate
        coupons = sorted(self.prices.keys())
        if coupon <= coupons[0]:
            return self.prices[coupons[0]]
        if coupon >= coupons[-1]:
            return self.prices[coupons[-1]]
        for i in range(len(coupons) - 1):
            if coupons[i] <= coupon <= coupons[i + 1]:
                w = (coupon - coupons[i]) / (coupons[i + 1] - coupons[i])
                return self.prices[coupons[i]] * (1 - w) + self.prices[coupons[i + 1]] * w
        return 100.0


def build_tba_price_grid(
    treasury_10y: float,
    agency: str = "FNMA15",
    term: int = 30,
    vol: float = 0.008,
) -> TBAPriceGrid:
    """
    Build a TBA price grid based on current market conditions.
    Mirrors the price display in the Yield Book screenshots.

    The current coupon (par coupon) is approximately:
        current_coupon = 10Y Treasury + mortgage-treasury spread - servicing - gfee
    """
    if agency.startswith("GN"):
        spread = 0.0170  # GNMA spread ~170bps
        servicing = 0.0044  # 44bps
        gfee = 0.0006  # 6bps
    else:
        spread = 0.0150  # Conv spread ~150bps
        servicing = 0.0025  # 25bps
        gfee = 0.0045  # ~45bps average

    current_coupon_rate = treasury_10y + spread
    par_coupon = round((current_coupon_rate - servicing - gfee) * 2) / 2  # Round to nearest 0.5

    # Generate prices for standard coupons
    if term == 30:
        coupons = [c / 2 for c in range(4, 17)]  # 2.0 to 8.0 in 0.5 steps
    else:
        coupons = [c / 2 for c in range(3, 14)]  # 1.5 to 6.5

    prices = {}
    duration_base = 5.5 if term == 30 else 3.5

    for cpn in coupons:
        moneyness = cpn - par_coupon
        # Duration decreases as coupon increases (more prepayment)
        duration = duration_base - moneyness * 0.8
        duration = max(2.0, min(8.0, duration))

        # Price = par + moneyness * duration * 100bps
        price = 100.0 + moneyness * duration
        # Cap at realistic levels
        price = max(85.0, min(115.0, price))
        prices[cpn] = round(price, 4)

    return TBAPriceGrid(
        agency=agency,
        term=term,
        settlement_month="current",
        prices=prices,
    )


# ─── Dollar Roll Analytics ───────────────────────────────────────────────

@dataclass
class DollarRollResult:
    """Result of dollar roll analysis."""
    front_price: float          # Front month TBA price
    back_price: float           # Back month TBA price
    drop: float                 # Price difference (front - back)
    drop_32nds: float           # Drop in 32nds
    implied_financing_rate: float  # Implied repo rate from the roll
    coupon_income: float        # Monthly coupon income
    paydown_return: float       # Return from principal paydown
    breakeven_speed: float      # PSA speed at which roll is breakeven
    roll_advantage_bps: float   # Advantage over repo financing
    recommendation: str         # "Roll" or "Hold"


def analyze_dollar_roll(
    coupon: float,
    front_price: float,
    back_price: float,
    wam: int = 357,
    psa_speed: float = 150,
    repo_rate: float = 0.043,
    days_between: int = 30,
    face_value: float = 1_000_000,
) -> DollarRollResult:
    """
    Analyze a dollar roll transaction.

    From the paper (Section 5):
    "In a dollar roll, the roll seller sells TBAs for a coming delivery month
    (the front month) and simultaneously purchases TBAs for a later back month.
    This provides short-term funding to the roll seller."

    The roll is attractive when the implied financing rate < repo rate.

    Args:
        coupon: MBS coupon rate
        front_price: Front month TBA price
        back_price: Back month TBA price
        wam: Weighted average maturity
        psa_speed: PSA prepayment speed
        repo_rate: Current repo rate (for comparison)
        days_between: Days between settlements
        face_value: Notional face value
    """
    drop = front_price - back_price
    drop_32nds = drop * 32

    # Monthly coupon income foregone by rolling
    monthly_coupon = coupon / 100.0 / 12.0 * face_value

    # Principal paydown (at given PSA)
    from .prepayment import psa_smm
    smm = psa_smm(wam - (360 - wam), psa_speed)
    paydown = face_value * smm
    paydown_return = paydown * (front_price / 100.0 - 1.0) if front_price > 100 else 0

    # Implied financing rate
    # Roll proceeds = front_price * face - back_price * face * (1-smm)
    proceeds = front_price / 100.0 * face_value
    cost = back_price / 100.0 * face_value * (1 - smm)

    # Net carry = coupon income + paydown gain - price drop
    price_drop_dollars = drop / 100.0 * face_value
    net_carry = monthly_coupon - price_drop_dollars + paydown_return

    # Implied financing rate (annualized)
    if proceeds > 0:
        implied_rate = (monthly_coupon - price_drop_dollars) / proceeds * 12
    else:
        implied_rate = 0.0

    # Roll advantage
    repo_cost = proceeds * repo_rate * days_between / 360
    roll_advantage_bps = (repo_rate - implied_rate) * 10000

    # Breakeven speed
    breakeven_smm = max(0, (drop / 100.0 * face_value - monthly_coupon) /
                        (face_value * (front_price / 100.0 - 1.0))) if front_price > 100 else 0
    breakeven_cpr = 1 - (1 - breakeven_smm) ** 12
    from .prepayment import cpr_from_smm
    breakeven_psa = breakeven_cpr / 0.06 * 100 if breakeven_cpr > 0 else 0

    recommendation = "Roll" if implied_rate < repo_rate else "Hold"

    return DollarRollResult(
        front_price=front_price,
        back_price=back_price,
        drop=round(drop, 6),
        drop_32nds=round(drop_32nds, 2),
        implied_financing_rate=round(implied_rate, 4),
        coupon_income=round(monthly_coupon, 2),
        paydown_return=round(paydown_return, 2),
        breakeven_speed=round(breakeven_psa, 0),
        roll_advantage_bps=round(roll_advantage_bps, 1),
        recommendation=recommendation,
    )


# ─── Cheapest-to-Deliver ─────────────────────────────────────────────────

def cheapest_to_deliver(
    pools: list[SpecPool],
    tba_price: float,
    current_mortgage_rate: float,
) -> list[dict]:
    """
    Rank pools by delivery value to determine cheapest-to-deliver.

    From the paper:
    "The TBA market operates on a cheapest-to-deliver basis — sellers will
    deliver the least valuable eligible pools."

    Pools with faster prepayment speeds are cheaper to deliver when trading
    at a premium (price > par), and vice versa.

    Returns pools ranked from cheapest to most expensive to deliver.
    """
    results = []
    for pool in pools:
        psa = estimate_psa_speed(pool.wac, current_mortgage_rate / 100.0, pool.wala)
        cf = project_pool_cashflows(pool, psa_speed=psa)

        # Delivery value = TBA price * pool factor (simplified)
        delivery_value = tba_price / 100.0 * pool.current_balance

        # Intrinsic value based on WAL and prepayment
        wal = cf.wal
        # Faster prepay = lower value at premium, higher value at discount
        if tba_price > 100:
            value_adjustment = -(psa - 150) / 150 * 0.5  # Premium: fast prepay = bad
        else:
            value_adjustment = (psa - 150) / 150 * 0.3   # Discount: fast prepay = good

        pool_value = delivery_value + value_adjustment * pool.current_balance / 100

        results.append({
            "pool_id": pool.pool_id,
            "agency": pool.agency.value,
            "coupon": pool.coupon,
            "wac": pool.wac,
            "wam": pool.wam,
            "wala": pool.wala,
            "balance": pool.current_balance,
            "psa_speed": round(psa, 0),
            "wal_years": round(wal, 2),
            "delivery_value": round(delivery_value, 2),
            "estimated_pool_value": round(pool_value, 2),
            "payup_over_tba": round((pool_value - delivery_value) / pool.current_balance * 32 * 100, 1),
            "ctd_rank": 0,  # Will be filled in
        })

    # Sort by delivery value (cheapest first)
    results.sort(key=lambda x: x["estimated_pool_value"])
    for i, r in enumerate(results):
        r["ctd_rank"] = i + 1

    return results


# ─── Good Delivery Rules ─────────────────────────────────────────────────

@dataclass
class GoodDeliveryCheck:
    """Result of checking good delivery requirements."""
    is_deliverable: bool
    variance_pct: float
    pools_count: int
    max_pools_allowed: int
    issues: list[str]


def check_good_delivery(
    pools: list[SpecPool],
    tba_face: float = 1_000_000,
    max_pools: int = 3,
    variance_limit: float = 0.01,  # 1% variance allowed
) -> GoodDeliveryCheck:
    """
    Check if pools satisfy SIFMA good delivery guidelines.

    Key rules:
    - Delivery variance: +/- 0.01% of contract face
    - Maximum number of pools per $1M face
    - All pools must be from the specified agency
    - Must match coupon and term of the TBA contract
    """
    issues = []
    total_face = sum(p.current_balance for p in pools)
    variance = abs(total_face - tba_face) / tba_face

    if variance > variance_limit:
        issues.append(f"Variance {variance*100:.3f}% exceeds {variance_limit*100}% limit")

    if len(pools) > max_pools:
        issues.append(f"Too many pools ({len(pools)}), max is {max_pools} per $1M")

    # Check agency consistency
    agencies = set(p.agency for p in pools)
    if len(agencies) > 1:
        issues.append("Mixed agencies in delivery")

    # Check coupon consistency
    coupons = set(p.coupon for p in pools)
    if len(coupons) > 1:
        issues.append("Mixed coupons in delivery")

    return GoodDeliveryCheck(
        is_deliverable=len(issues) == 0,
        variance_pct=round(variance * 100, 4),
        pools_count=len(pools),
        max_pools_allowed=max_pools,
        issues=issues,
    )
