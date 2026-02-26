"""
Advanced Risk Analytics for MBS and CMO.

Implements:
- Key Rate Durations (KRD) / Partial DV01s
- Effective duration and convexity with full prepayment repricing
- Value-at-Risk (VaR) and Expected Shortfall
- Scenario/stress test matrices
- Convexity hedging analytics
- Spread duration and spread DV01

From the paper: "MBS are callable securities, and price appreciation from
lower interest rates is therefore capped — MBS exhibit negative convexity."
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .yield_curve import YieldCurve, YieldCurvePoint
from .spec_pool import SpecPool, project_pool_cashflows, PoolCashFlows
from .prepayment import PrepaymentModel, PrepaymentModelConfig
from .cmo_structure import structure_cmo, TrancheSpec, CMOCashFlows
from .pricing import price_tranche, z_spread, PricingResult


# ─── Key Rate Duration ────────────────────────────────────────────────────

KEY_RATE_TENORS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]


def key_rate_durations(
    pool: SpecPool,
    curve: YieldCurve,
    psa_speed: float = 150,
    shift_bps: float = 25.0,
    tenors: Optional[list[float]] = None,
    prepay_model: Optional[PrepaymentModel] = None,
    recompute_prepay: bool = True,
) -> dict:
    """
    Compute key rate durations (partial DV01s) for an MBS pool.

    KRD measures sensitivity to changes at specific points on the curve.
    Sum of KRDs ≈ effective duration.

    Args:
        pool: The MBS pool
        curve: Base yield curve
        psa_speed: PSA speed assumption
        shift_bps: Size of rate shift at each tenor
        tenors: Key rate tenors (default: standard set)
        prepay_model: If provided + recompute_prepay, recomputes prepayments at shifted rates
        recompute_prepay: Whether to re-estimate prepayment at shifted rates
    """
    if tenors is None:
        tenors = KEY_RATE_TENORS

    shift = shift_bps / 100.0  # Convert to percentage points

    # Base price
    cf_base = project_pool_cashflows(pool, psa_speed=psa_speed)
    base_result = price_tranche(
        _pool_to_tranche_cf(cf_base, "BASE"), curve, spread_bps=0
    )
    base_price = base_result.price

    krds = {}
    for tenor in tenors:
        # Create curve shifted up at this tenor
        curve_up = _shift_curve_at_tenor(curve, tenor, shift)
        curve_down = _shift_curve_at_tenor(curve, tenor, -shift)

        if recompute_prepay and prepay_model:
            # Re-estimate prepayment speed at shifted rates
            shifted_10y_up = curve_up.get_yield(10.0)
            shifted_rate_up = shifted_10y_up / 100 + 0.017
            psa_up = max(25, psa_speed * (1 - (shift_bps / 100) * 2))
            psa_down = min(600, psa_speed * (1 + (shift_bps / 100) * 2))

            cf_up = project_pool_cashflows(pool, psa_speed=psa_up)
            cf_down = project_pool_cashflows(pool, psa_speed=psa_down)
        else:
            cf_up = cf_base
            cf_down = cf_base

        price_up = price_tranche(
            _pool_to_tranche_cf(cf_up, "UP"), curve_up, spread_bps=0
        ).price
        price_down = price_tranche(
            _pool_to_tranche_cf(cf_down, "DOWN"), curve_down, spread_bps=0
        ).price

        krd = -(price_up - price_down) / (2 * shift_bps / 10000 * base_price)
        krds[f"KRD_{tenor}Y"] = round(krd, 4)

    # Sum check
    krds["total_krd"] = round(sum(v for k, v in krds.items() if k != "total_krd"), 4)
    krds["base_price"] = round(base_price, 4)

    return krds


def _shift_curve_at_tenor(curve: YieldCurve, tenor: float, shift_pct: float) -> YieldCurve:
    """Create a curve with a localized shift at a specific tenor."""
    new_points = []
    for p in curve.points:
        # Triangular weight: full shift at tenor, zero at adjacent tenors
        dist = abs(p.maturity_years - tenor)
        weight = max(0, 1.0 - dist / max(tenor * 0.5, 1.0))
        new_yield = p.yield_pct + shift_pct * weight
        new_points.append(YieldCurvePoint(p.maturity_years, new_yield))
    return YieldCurve(as_of_date=curve.as_of_date, points=new_points)


def _pool_to_tranche_cf(cf: PoolCashFlows, name: str = "POOL"):
    """Convert PoolCashFlows to a tranche-like object for pricing."""
    from .cmo_structure import TrancheCashFlows
    return TrancheCashFlows(
        name=name,
        months=cf.months,
        beginning_balance=cf.beginning_balance,
        principal=cf.total_principal,
        interest=cf.interest,
        total_cash_flow=cf.total_cash_flow,
        ending_balance=cf.ending_balance,
        accrued_interest=np.zeros(len(cf.months)),
    )


# ─── Effective Duration and Convexity with Prepayment Repricing ──────────

@dataclass
class FullRiskMetrics:
    """Complete risk metrics with prepayment repricing."""
    price: float
    eff_duration: float
    eff_convexity: float
    mod_duration: float
    spread_duration: float
    dv01: float
    spread_dv01: float
    wal_years: float
    wal_up: float        # WAL if rates +100
    wal_down: float      # WAL if rates -100
    price_up_100: float  # Price if rates +100
    price_down_100: float  # Price if rates -100
    price_up_50: float
    price_down_50: float
    negative_convexity: bool


def compute_full_risk_metrics(
    pool: SpecPool,
    curve: YieldCurve,
    psa_speed: float = 150,
    market_price: Optional[float] = None,
) -> FullRiskMetrics:
    """
    Compute full risk metrics with prepayment repricing at shifted rates.

    This captures the negative convexity that is the hallmark of MBS:
    - When rates fall, prepayments increase, shortening duration
    - When rates rise, prepayments decrease, extending duration
    """
    # Base case
    cf_base = project_pool_cashflows(pool, psa_speed=psa_speed)
    base_result = price_tranche(_pool_to_tranche_cf(cf_base), curve)
    if market_price is None:
        market_price = base_result.price

    # Rate shifts: recompute PSA at each level
    shifts = [-100, -50, 0, 50, 100]
    prices = {}
    wals = {}

    for shift in shifts:
        shifted_curve = curve.shift(shift)
        # Re-estimate PSA at shifted rates
        shifted_10y = curve.get_yield(10.0) + shift / 100.0
        shifted_mortgage_rate = shifted_10y / 100 + 0.017
        new_psa = max(25, min(800, _estimate_shifted_psa(
            pool.wac, shifted_mortgage_rate, pool.wala, psa_speed)))

        cf = project_pool_cashflows(pool, psa_speed=new_psa)
        result = price_tranche(_pool_to_tranche_cf(cf), shifted_curve)
        prices[shift] = result.price
        wals[shift] = cf.wal

    # Effective duration: (P_down - P_up) / (2 * delta_y * P_base)
    eff_dur = -(prices[100] - prices[-100]) / (2 * 0.01 * market_price)

    # Effective convexity: (P_up + P_down - 2*P_base) / (delta_y^2 * P_base)
    eff_conv = (prices[-100] + prices[100] - 2 * prices[0]) / (0.01 ** 2 * market_price)

    # Modified duration (no prepayment repricing)
    cf_base_result = price_tranche(_pool_to_tranche_cf(cf_base), curve)
    from .pricing import modified_duration
    mod_dur = modified_duration(_pool_to_tranche_cf(cf_base), cf_base_result.yield_pct)

    # Spread duration (sensitivity to OAS changes)
    spread_dur = eff_dur * 0.85  # Approximate

    dv01 = eff_dur * market_price / 10000
    spread_dv01 = spread_dur * market_price / 10000

    return FullRiskMetrics(
        price=round(market_price, 4),
        eff_duration=round(eff_dur, 4),
        eff_convexity=round(eff_conv, 4),
        mod_duration=round(mod_dur, 4),
        spread_duration=round(spread_dur, 4),
        dv01=round(dv01, 6),
        spread_dv01=round(spread_dv01, 6),
        wal_years=round(wals[0], 2),
        wal_up=round(wals[100], 2),
        wal_down=round(wals[-100], 2),
        price_up_100=round(prices[100], 4),
        price_down_100=round(prices[-100], 4),
        price_up_50=round(prices[50], 4),
        price_down_50=round(prices[-50], 4),
        negative_convexity=eff_conv < 0,
    )


def _estimate_shifted_psa(wac: float, mortgage_rate: float, wala: int, base_psa: float) -> float:
    """Estimate PSA speed at a shifted mortgage rate."""
    from .prepayment import estimate_psa_speed
    new_psa = estimate_psa_speed(wac, mortgage_rate, wala)
    # Blend with base to avoid extreme jumps
    return 0.7 * new_psa + 0.3 * base_psa


# ─── Scenario Matrix ─────────────────────────────────────────────────────

def scenario_matrix(
    pool: SpecPool,
    curve: YieldCurve,
    rate_shifts: list[int] = None,
    psa_speeds: list[float] = None,
) -> dict:
    """
    Generate a full price/WAL/duration scenario matrix.

    This is the core of Yield Book's scenario analysis:
    rows = rate shifts, columns = PSA speeds.
    """
    if rate_shifts is None:
        rate_shifts = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
    if psa_speeds is None:
        psa_speeds = [50, 75, 100, 150, 200, 250, 300, 400, 500]

    matrix = {
        "rate_shifts": rate_shifts,
        "psa_speeds": psa_speeds,
        "price": [],
        "wal": [],
        "yield": [],
        "duration": [],
    }

    for shift in rate_shifts:
        price_row = []
        wal_row = []
        yield_row = []
        dur_row = []

        shifted_curve = curve.shift(shift)

        for psa in psa_speeds:
            cf = project_pool_cashflows(pool, psa_speed=psa)
            tcf = _pool_to_tranche_cf(cf)
            result = price_tranche(tcf, shifted_curve)

            price_row.append(round(result.price, 3))
            wal_row.append(round(cf.wal, 2))
            yield_row.append(round(result.yield_pct, 3))
            dur_row.append(round(result.mod_duration, 3))

        matrix["price"].append(price_row)
        matrix["wal"].append(wal_row)
        matrix["yield"].append(yield_row)
        matrix["duration"].append(dur_row)

    return matrix


# ─── Value-at-Risk ────────────────────────────────────────────────────────

def historical_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    holding_period_days: int = 1,
) -> dict:
    """Compute historical VaR and Expected Shortfall."""
    if len(returns) < 10:
        return {"var": 0, "es": 0, "n_obs": len(returns)}

    sorted_returns = np.sort(returns)
    var_idx = int((1 - confidence) * len(sorted_returns))
    var = -sorted_returns[var_idx]
    es = -np.mean(sorted_returns[:var_idx]) if var_idx > 0 else var

    # Scale for holding period
    scale = np.sqrt(holding_period_days)

    return {
        "var_1d": round(float(var), 4),
        "var_scaled": round(float(var * scale), 4),
        "es_1d": round(float(es), 4),
        "es_scaled": round(float(es * scale), 4),
        "confidence": confidence,
        "holding_period": holding_period_days,
        "n_observations": len(returns),
        "annualized_vol": round(float(np.std(returns) * np.sqrt(252)), 4),
    }


def parametric_var(
    position_value: float,
    eff_duration: float,
    yield_vol_bps: float = 5.0,
    confidence: float = 0.95,
    holding_period_days: int = 1,
) -> dict:
    """
    Parametric VaR for an MBS position.

    VaR = Position * Duration * Yield_Change * z_score
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)
    daily_yield_change = yield_vol_bps / 10000
    var_1d = position_value * eff_duration * daily_yield_change * z
    var_scaled = var_1d * np.sqrt(holding_period_days)

    return {
        "var_1d_dollars": round(var_1d, 2),
        "var_scaled_dollars": round(var_scaled, 2),
        "var_1d_pct": round(var_1d / position_value * 100, 4),
        "position_value": position_value,
        "eff_duration": eff_duration,
        "yield_vol_bps": yield_vol_bps,
        "confidence": confidence,
        "z_score": round(z, 4),
    }


# ─── Convexity Hedging Analytics ─────────────────────────────────────────

@dataclass
class ConvexityHedge:
    """Convexity hedging recommendation."""
    hedge_instrument: str
    notional: float
    delta_hedge_ratio: float
    gamma_hedge_notional: float
    vega_exposure: float
    net_convexity: float
    hedge_cost_bps: float


def convexity_hedge_analysis(
    pool_balance: float,
    pool_duration: float,
    pool_convexity: float,
    treasury_duration: float = 6.5,
    swaption_gamma: float = 0.5,
    swaption_cost_bps: float = 10,
) -> dict:
    """
    Analyze convexity hedging for an MBS portfolio.

    From the paper (Section 6):
    "Convexity hedging flows lead to important interactions between the
    MBS market and the Treasury yield curve."

    MBS servicers and investors must dynamically hedge the negative convexity
    of MBS portfolios, typically using Treasuries (for duration) and
    swaptions (for convexity).
    """
    # Delta hedge with Treasuries
    hedge_ratio = pool_duration / treasury_duration
    treasury_notional = pool_balance * hedge_ratio

    # Gamma/convexity hedge with swaptions
    convexity_gap = pool_convexity  # Negative for MBS
    swaption_notional = abs(convexity_gap) * pool_balance / swaption_gamma if swaption_gamma > 0 else 0
    hedge_cost = swaption_notional * swaption_cost_bps / 10000

    # Net position
    net_convexity = pool_convexity + swaption_notional * swaption_gamma / pool_balance

    return {
        "pool_balance": pool_balance,
        "pool_duration": pool_duration,
        "pool_convexity": pool_convexity,
        "treasury_hedge_notional": round(treasury_notional, 2),
        "treasury_hedge_ratio": round(hedge_ratio, 4),
        "swaption_notional": round(swaption_notional, 2),
        "swaption_hedge_cost_annual": round(hedge_cost, 2),
        "net_convexity_after_hedge": round(net_convexity, 4),
        "total_hedge_cost_bps": round(hedge_cost / pool_balance * 10000, 1),
    }
