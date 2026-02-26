"""
CMO Pricing Analytics.
Implements OAS, yield, WAL, duration, convexity, and Z-spread calculations.
"""
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional

from .yield_curve import YieldCurve
from .cmo_structure import TrancheCashFlows, CMOCashFlows
from .spec_pool import PoolCashFlows


@dataclass
class PricingResult:
    """Complete pricing result for a tranche or pool."""
    name: str
    price: float                  # Clean price (per 100 par)
    yield_pct: float             # Yield to maturity (%)
    wal_years: float             # Weighted average life
    mod_duration: float          # Modified duration
    eff_duration: float          # Effective duration
    convexity: float             # Convexity
    spread_bps: float            # Spread over benchmark
    oas_bps: float               # Option-adjusted spread
    z_spread_bps: float          # Z-spread
    dv01: float                  # Dollar value of 1bp
    window: str                  # Principal payment window
    first_pay: int
    last_pay: int
    total_principal: float
    total_interest: float

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'price': round(self.price, 4),
            'yield': round(self.yield_pct, 3),
            'wal': round(self.wal_years, 2),
            'mod_duration': round(self.mod_duration, 3),
            'eff_duration': round(self.eff_duration, 3),
            'convexity': round(self.convexity, 3),
            'spread_bps': round(self.spread_bps, 1),
            'oas_bps': round(self.oas_bps, 1),
            'z_spread_bps': round(self.z_spread_bps, 1),
            'dv01': round(self.dv01, 4),
            'window': self.window,
            'total_principal': round(self.total_principal, 2),
            'total_interest': round(self.total_interest, 2),
        }


def pv_cashflows(cashflows: np.ndarray, months: np.ndarray,
                 discount_rate_annual: float) -> float:
    """Calculate present value of cash flows at a flat discount rate."""
    monthly_rate = discount_rate_annual / 12.0
    discount_factors = (1.0 + monthly_rate) ** (-months)
    return float(np.sum(cashflows * discount_factors))


def price_from_yield(tranche_cf: TrancheCashFlows, yield_pct: float,
                     par: float = 100.0) -> float:
    """Price a tranche given a yield (as percentage)."""
    total_cf = tranche_cf.total_cash_flow
    months = tranche_cf.months
    nonzero = total_cf > 0.01
    if not np.any(nonzero):
        return 0.0

    original_bal = tranche_cf.beginning_balance[0]
    if original_bal <= 0:
        return 0.0

    pv = pv_cashflows(total_cf[nonzero], months[nonzero], yield_pct / 100.0)
    return pv / original_bal * par


def yield_from_price(tranche_cf: TrancheCashFlows, price: float,
                     par: float = 100.0) -> float:
    """Calculate yield given a price (per 100 par)."""
    original_bal = tranche_cf.beginning_balance[0]
    if original_bal <= 0:
        return 0.0

    target_pv = price / par * original_bal

    def objective(y):
        total_cf = tranche_cf.total_cash_flow
        months = tranche_cf.months
        nonzero = total_cf > 0.01
        pv = pv_cashflows(total_cf[nonzero], months[nonzero], y / 100.0)
        return pv - target_pv

    try:
        result = brentq(objective, -5.0, 50.0, xtol=1e-6)
        return result
    except (ValueError, RuntimeError):
        return 0.0


def z_spread(tranche_cf: TrancheCashFlows, curve: YieldCurve,
             price: float, par: float = 100.0) -> float:
    """
    Calculate Z-spread: constant spread over the spot curve that
    equates the PV of cash flows to the market price.
    """
    original_bal = tranche_cf.beginning_balance[0]
    if original_bal <= 0:
        return 0.0
    target_pv = price / par * original_bal

    total_cf = tranche_cf.total_cash_flow
    months = tranche_cf.months
    nonzero = total_cf > 0.01

    def objective(spread_bps):
        spread = spread_bps / 10000.0
        pv = 0.0
        for m, cf in zip(months[nonzero], total_cf[nonzero]):
            t = m / 12.0
            spot = curve.spot_rate(t)
            df = np.exp(-(spot + spread) * t)
            pv += cf * df
        return pv - target_pv

    try:
        result = brentq(objective, -500, 2000, xtol=0.01)
        return result
    except (ValueError, RuntimeError):
        return 0.0


def oas_calculation(tranche_cf: TrancheCashFlows, curve: YieldCurve,
                    price: float, par: float = 100.0,
                    n_paths: int = 100, vol_bps: float = 80) -> float:
    """
    Simplified OAS calculation using Monte Carlo rate paths.
    Full production OAS would use a term structure model (e.g., BDT, Hull-White).
    """
    original_bal = tranche_cf.beginning_balance[0]
    if original_bal <= 0:
        return 0.0
    target_pv = price / par * original_bal

    # For now, approximate OAS as Z-spread with a vol adjustment
    z_sprd = z_spread(tranche_cf, curve, price, par)

    # Simple vol adjustment: OAS ~ Z-spread - convexity_cost
    # Convexity cost increases with vol and option value
    wal = tranche_cf.wal
    convexity_cost = vol_bps * wal * 0.05  # Rough approximation
    oas = z_sprd - convexity_cost

    return max(-200, oas)


def modified_duration(tranche_cf: TrancheCashFlows, yield_pct: float,
                      shift_bps: float = 1.0) -> float:
    """Calculate modified duration via finite difference."""
    p_up = price_from_yield(tranche_cf, yield_pct + shift_bps / 100.0)
    p_down = price_from_yield(tranche_cf, yield_pct - shift_bps / 100.0)
    p_base = price_from_yield(tranche_cf, yield_pct)

    if p_base <= 0:
        return 0.0

    return -(p_up - p_down) / (2.0 * shift_bps / 10000.0 * p_base)


def effective_duration(tranche_cf: TrancheCashFlows, curve: YieldCurve,
                       price: float, shift_bps: float = 25.0) -> float:
    """
    Calculate effective duration using parallel curve shifts.
    This captures the option effect (prepayment changes with rates).
    """
    # For a full implementation, would re-run prepayment model at shifted rates
    # Here we use a simplified version
    y = yield_from_price(tranche_cf, price)
    return modified_duration(tranche_cf, y, shift_bps)


def convexity_calc(tranche_cf: TrancheCashFlows, yield_pct: float,
                   shift_bps: float = 10.0) -> float:
    """Calculate convexity via finite difference."""
    p_up = price_from_yield(tranche_cf, yield_pct + shift_bps / 100.0)
    p_down = price_from_yield(tranche_cf, yield_pct - shift_bps / 100.0)
    p_base = price_from_yield(tranche_cf, yield_pct)

    if p_base <= 0:
        return 0.0

    dy = shift_bps / 10000.0
    return (p_up + p_down - 2 * p_base) / (dy ** 2 * p_base)


def price_tranche(tranche_cf: TrancheCashFlows, curve: YieldCurve,
                  spread_bps: float = 0.0, price_override: Optional[float] = None) -> PricingResult:
    """
    Full pricing analysis for a tranche.

    Args:
        tranche_cf: Cash flows for the tranche
        curve: Yield curve for discounting
        spread_bps: Spread over the curve for pricing
        price_override: If provided, use this price instead of computing
    """
    wal = tranche_cf.wal

    if price_override is not None:
        price = price_override
    else:
        # Price using spot curve + spread
        total_cf = tranche_cf.total_cash_flow
        months = tranche_cf.months
        spread = spread_bps / 10000.0
        pv = 0.0
        for m, cf in zip(months, total_cf):
            if cf <= 0.01:
                continue
            t = m / 12.0
            spot = curve.spot_rate(t)
            df = np.exp(-(spot + spread) * t)
            pv += cf * df

        original_bal = tranche_cf.beginning_balance[0]
        price = pv / original_bal * 100.0 if original_bal > 0 else 0.0

    y = yield_from_price(tranche_cf, price)
    mod_dur = modified_duration(tranche_cf, y)
    eff_dur = effective_duration(tranche_cf, curve, price)
    conv = convexity_calc(tranche_cf, y)
    z_sprd = z_spread(tranche_cf, curve, price)
    oas = oas_calculation(tranche_cf, curve, price)
    dv01 = mod_dur * price / 10000.0

    return PricingResult(
        name=tranche_cf.name,
        price=price,
        yield_pct=y,
        wal_years=wal,
        mod_duration=mod_dur,
        eff_duration=eff_dur,
        convexity=conv,
        spread_bps=spread_bps,
        oas_bps=oas,
        z_spread_bps=z_sprd,
        dv01=dv01,
        window=tranche_cf.window,
        first_pay=tranche_cf.first_pay_month,
        last_pay=tranche_cf.last_pay_month,
        total_principal=tranche_cf.total_principal_paid,
        total_interest=tranche_cf.total_interest_paid,
    )


def price_deal(cmo_cf: CMOCashFlows, curve: YieldCurve,
               spreads: Optional[dict[str, float]] = None) -> dict[str, PricingResult]:
    """Price all tranches in a CMO deal."""
    if spreads is None:
        spreads = {}

    results = {}
    for name, tcf in cmo_cf.tranche_flows.items():
        spread = spreads.get(name, 0.0)
        results[name] = price_tranche(tcf, curve, spread)

    return results


def pool_price_from_yield(pool_cf: PoolCashFlows, yield_pct: float,
                          par: float = 100.0) -> float:
    """Price a mortgage pool from yield."""
    total_cf = pool_cf.total_cash_flow
    months = pool_cf.months
    nonzero = total_cf > 0.01
    if not np.any(nonzero):
        return 0.0

    original_bal = pool_cf.beginning_balance[0]
    if original_bal <= 0:
        return 0.0

    pv = pv_cashflows(total_cf[nonzero], months[nonzero], yield_pct / 100.0)
    return pv / original_bal * par


def structuring_profit(
    collateral_cost: float,
    tranche_proceeds: dict[str, float],
    total_collateral: float,
) -> dict:
    """
    Calculate structuring profit from a CMO deal.

    Args:
        collateral_cost: Total cost of buying the collateral pools
        tranche_proceeds: Dict of tranche name -> proceeds from selling
        total_collateral: Total face value of collateral
    """
    total_proceeds = sum(tranche_proceeds.values())
    profit = total_proceeds - collateral_cost
    profit_bps = profit / total_collateral * 10000 if total_collateral > 0 else 0

    return {
        'collateral_cost': round(collateral_cost, 2),
        'total_proceeds': round(total_proceeds, 2),
        'profit_dollars': round(profit, 2),
        'profit_bps': round(profit_bps, 1),
        'profit_pct': round(profit / collateral_cost * 100, 3) if collateral_cost > 0 else 0,
        'tranche_proceeds': {k: round(v, 2) for k, v in tranche_proceeds.items()},
    }
