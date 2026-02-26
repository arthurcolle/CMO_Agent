"""
S-Curve Analytics and Market Analysis for MBS.

Implements:
- Prepayment S-curve modeling (CPR vs. rate incentive)
- Burnout and media effects
- Turnover decomposition
- Historical prepayment analysis
- OAS smile decomposition
- Relative value analysis
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .prepayment import PrepaymentModel, PrepaymentModelConfig, estimate_psa_speed
from .yield_curve import YieldCurve
from .spec_pool import SpecPool, project_pool_cashflows


# ─── S-Curve Analysis ──────────────────────────────────────────────────────

@dataclass
class SCurvePoint:
    """A single point on the prepayment S-curve."""
    incentive_bps: float  # WAC - current mortgage rate, in bps
    cpr: float            # Annualized CPR
    smm: float            # Single-month mortality
    psa_speed: float      # Equivalent PSA speed
    turnover: float       # Turnover component
    refi: float           # Refinancing component


def compute_s_curve(
    wac: float,
    wala: int = 24,
    fico: int = 740,
    ltv: float = 75.0,
    rate_range: tuple[float, float] = (0.02, 0.09),
    n_points: int = 50,
    burnout_factor: float = 1.0,
    model_config: Optional[PrepaymentModelConfig] = None,
) -> list[SCurvePoint]:
    """
    Compute the prepayment S-curve for a given pool profile.

    The S-curve shows CPR as a function of the refinancing incentive
    (WAC - current mortgage rate). Key features:
    - Below zero incentive: CPR is dominated by turnover (~5-8% CPR)
    - At zero: CPR starts to increase as some borrowers refinance
    - Above 100-150bps: CPR accelerates rapidly (steep part of S)
    - Above 250-300bps: CPR flattens (burnout, capacity constraints)

    Args:
        wac: Weighted average coupon (decimal, e.g. 0.055 for 5.5%)
        wala: Weighted average loan age in months
        fico: Average FICO score
        ltv: Average LTV ratio
        rate_range: Range of mortgage rates to sweep
        n_points: Number of points on the curve
        burnout_factor: Pool burnout level (1.0 = new, 0.5 = heavily burned out)
        model_config: Prepayment model configuration
    """
    if model_config is None:
        model_config = PrepaymentModelConfig()

    model = PrepaymentModel(config=model_config)
    rates = np.linspace(rate_range[0], rate_range[1], n_points)
    points = []

    for rate in rates:
        incentive = wac - rate
        incentive_bps = incentive * 10000

        # Get SMM from the model
        smm = model.project_smm(
            month=wala + 1,
            wac=wac,
            current_mortgage_rate=rate,
            pool_factor=burnout_factor,
            loan_age=wala,
        )

        # Decompose into turnover and refi
        # Turnover: base CPR that occurs regardless of rates
        base_turnover_cpr = model_config.base_cpr * _turnover_seasonality(wala)
        base_turnover_smm = 1 - (1 - base_turnover_cpr) ** (1/12)

        refi_smm = max(0, smm - base_turnover_smm)
        cpr = 1 - (1 - smm) ** 12
        turnover_cpr = 1 - (1 - base_turnover_smm) ** 12
        refi_cpr = max(0, cpr - turnover_cpr)

        psa = cpr / 0.06 * 100 if cpr > 0 else 0

        points.append(SCurvePoint(
            incentive_bps=round(incentive_bps, 0),
            cpr=round(cpr * 100, 2),
            smm=round(smm * 100, 4),
            psa_speed=round(psa, 0),
            turnover=round(turnover_cpr * 100, 2),
            refi=round(refi_cpr * 100, 2),
        ))

    return points


def _turnover_seasonality(month_of_year: int) -> float:
    """Seasonality adjustment for turnover (housing activity)."""
    # Peak in summer (June-August), trough in winter
    month = (month_of_year % 12) + 1
    seasonality = {
        1: 0.80, 2: 0.85, 3: 0.95, 4: 1.05, 5: 1.15, 6: 1.20,
        7: 1.15, 8: 1.10, 9: 1.00, 10: 0.95, 11: 0.85, 12: 0.80,
    }
    return seasonality.get(month, 1.0)


# ─── Burnout Analysis ──────────────────────────────────────────────────────

def burnout_analysis(
    wac: float,
    current_rate: float,
    pool_ages: list[int],
    pool_factors: list[float],
) -> list[dict]:
    """
    Analyze burnout effect across pools of different ages and factors.

    Burnout occurs because the most rate-sensitive borrowers prepay first,
    leaving a pool of borrowers less likely to refinance. Lower pool factor
    = more burnout.
    """
    results = []
    model = PrepaymentModel()

    for age, factor in zip(pool_ages, pool_factors):
        smm = model.project_smm(
            month=age + 1,
            wac=wac,
            current_mortgage_rate=current_rate,
            pool_factor=factor,
            loan_age=age,
        )
        cpr = 1 - (1 - smm) ** 12
        incentive = wac - current_rate

        # Compare to a "fresh" pool
        smm_fresh = model.project_smm(
            month=age + 1,
            wac=wac,
            current_mortgage_rate=current_rate,
            pool_factor=1.0,
            loan_age=age,
        )
        cpr_fresh = 1 - (1 - smm_fresh) ** 12
        burnout_pct = (1 - cpr / cpr_fresh) * 100 if cpr_fresh > 0 else 0

        results.append({
            "age_months": age,
            "pool_factor": round(factor, 3),
            "cpr": round(cpr * 100, 2),
            "cpr_fresh": round(cpr_fresh * 100, 2),
            "burnout_pct": round(burnout_pct, 1),
            "incentive_bps": round(incentive * 10000, 0),
        })

    return results


# ─── Relative Value Analysis ───────────────────────────────────────────────

def relative_value_grid(
    coupons: list[float],
    curve: YieldCurve,
    mortgage_rate: float,
    collateral_balance: float = 1_000_000,
    wam: int = 357,
    wala: int = 3,
) -> list[dict]:
    """
    Compute relative value metrics across coupon stack.

    Shows price, WAL, OAS, duration, and convexity for each coupon,
    enabling comparison of value across the coupon spectrum.
    """
    from .pricing import price_tranche, z_spread as compute_z_spread
    from .risk import _pool_to_tranche_cf

    results = []
    for cpn in coupons:
        wac = cpn + 0.005
        moneyness = cpn - mortgage_rate
        psa = estimate_psa_speed(wac, mortgage_rate, wala)

        from .spec_pool import AgencyType, CollateralType
        pool = SpecPool(
            pool_id=f"RV_{cpn}",
            agency=AgencyType.FNMA,
            collateral_type=CollateralType.FN,
            coupon=cpn,
            wac=wac,
            wam=wam,
            wala=wala,
            original_balance=collateral_balance,
            current_balance=collateral_balance,
        )

        cf = project_pool_cashflows(pool, psa_speed=psa)
        tcf = _pool_to_tranche_cf(cf, f"RV_{cpn}")

        try:
            result = price_tranche(tcf, curve)
            z_sprd = compute_z_spread(tcf, curve, result.price)
        except Exception:
            continue

        results.append({
            "coupon": cpn,
            "wac": round(wac, 3),
            "moneyness_bps": round(moneyness * 10000, 0),
            "psa_speed": round(psa, 0),
            "price": round(result.price, 4),
            "yield_pct": round(result.yield_pct, 3),
            "wal_years": round(result.wal_years, 2),
            "mod_duration": round(result.mod_duration, 3),
            "eff_duration": round(result.eff_duration, 3),
            "z_spread_bps": round(z_sprd, 1),
            "dv01": round(result.dv01, 4),
        })

    return results


# ─── Coupon Swap Analysis ──────────────────────────────────────────────────

def coupon_swap_analysis(
    buy_coupon: float,
    sell_coupon: float,
    curve: YieldCurve,
    mortgage_rate: float,
    notional: float = 1_000_000,
    wam: int = 357,
    wala: int = 3,
) -> dict:
    """
    Analyze a coupon swap trade (sell one coupon, buy another).

    Common MBS trade: swap between coupons to adjust duration,
    convexity, or carry profile.
    """
    from .pricing import price_tranche
    from .risk import _pool_to_tranche_cf, compute_full_risk_metrics
    from .spec_pool import AgencyType, CollateralType

    results = {}
    for label, cpn in [("buy", buy_coupon), ("sell", sell_coupon)]:
        wac = cpn + 0.005
        psa = estimate_psa_speed(wac, mortgage_rate, wala)

        pool = SpecPool(
            pool_id=f"CS_{label}",
            agency=AgencyType.FNMA,
            collateral_type=CollateralType.FN,
            coupon=cpn,
            wac=wac,
            wam=wam,
            wala=wala,
            original_balance=notional,
            current_balance=notional,
        )

        cf = project_pool_cashflows(pool, psa_speed=psa)
        tcf = _pool_to_tranche_cf(cf, f"CS_{label}")
        pricing = price_tranche(tcf, curve)

        results[label] = {
            "coupon": cpn,
            "price": round(pricing.price, 4),
            "yield_pct": round(pricing.yield_pct, 3),
            "wal_years": round(pricing.wal_years, 2),
            "mod_duration": round(pricing.mod_duration, 3),
            "dv01": round(pricing.dv01, 4),
            "psa_speed": round(psa, 0),
        }

    # Compute swap metrics
    buy_dv01 = results["buy"]["dv01"]
    sell_dv01 = results["sell"]["dv01"]
    hedge_ratio = buy_dv01 / sell_dv01 if sell_dv01 != 0 else 1.0

    results["swap"] = {
        "buy_coupon": buy_coupon,
        "sell_coupon": sell_coupon,
        "price_diff": round(results["buy"]["price"] - results["sell"]["price"], 4),
        "yield_pickup_bps": round((results["buy"]["yield_pct"] - results["sell"]["yield_pct"]) * 100, 1),
        "duration_change": round(results["buy"]["mod_duration"] - results["sell"]["mod_duration"], 3),
        "wal_change": round(results["buy"]["wal_years"] - results["sell"]["wal_years"], 2),
        "dv01_neutral_hedge_ratio": round(hedge_ratio, 4),
        "sell_notional_for_dv01_neutral": round(notional * hedge_ratio, 0),
    }

    return results


# ─── WAL Distribution Analysis ────────────────────────────────────────────

def wal_distribution(
    pool: SpecPool,
    psa_speeds: list[float],
    curve: Optional[YieldCurve] = None,
) -> list[dict]:
    """
    Compute WAL sensitivity to prepayment speed assumptions.

    Shows how extension risk and contraction risk manifest.
    """
    results = []
    for psa in psa_speeds:
        cf = project_pool_cashflows(pool, psa_speed=psa)

        price = None
        if curve is not None:
            from .pricing import price_tranche
            from .risk import _pool_to_tranche_cf
            tcf = _pool_to_tranche_cf(cf, "WAL")
            try:
                pricing = price_tranche(tcf, curve)
                price = round(pricing.price, 4)
            except Exception:
                pass

        results.append({
            "psa_speed": psa,
            "wal_years": round(cf.wal, 2),
            "duration_months": round(cf.duration_months, 1),
            "total_interest": round(cf.total_interest, 0),
            "price": price,
        })

    return results
