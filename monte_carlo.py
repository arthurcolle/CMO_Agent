"""
Monte Carlo OAS Engine with Hull-White Short Rate Model.

Implements the OAS calculation from Fuster, Lucca & Vickery (2022), equation (1):

    P_M = E[ sum_k  X_k(r_k) / prod_j(1 + OAS + r_j) ]

where X_k are rate-dependent cash flows (through prepayment) and r_j are simulated
short rates. The OAS is found by solving for the spread that equates the model price
to the market price.

Features:
- Hull-White one-factor short rate model calibrated to the term structure
- Rate-dependent prepayment projection along each MC path
- True OAS solving via Brent's method across simulated paths
- Zero-volatility spread (ZVS) for comparison
- Option cost = ZVS - OAS
"""
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional

from .yield_curve import YieldCurve
from .prepayment import PrepaymentModel, PrepaymentModelConfig


@dataclass
class HullWhiteParams:
    """
    Hull-White model parameters: dr = [theta(t) - a*r] dt + sigma dW.

    Defaults calibrated from FRED historical data (2015-2026):
    - Mean reversion estimated via AR(1) on daily 10Y Treasury yields
    - Volatility from annualized daily rate changes
    """
    mean_reversion: float = 0.2032  # a: calibrated from 10Y Treasury AR(1)
    volatility: float = 0.00847     # sigma: calibrated from daily rate vol
    n_steps_per_year: int = 12      # Monthly steps for mortgage cash flows


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_paths: int = 500
    n_months: int = 360
    hw_params: Optional[HullWhiteParams] = None
    seed: Optional[int] = 42
    antithetic: bool = True          # Use antithetic variates for variance reduction

    def __post_init__(self):
        if self.hw_params is None:
            self.hw_params = HullWhiteParams()


def _calibrate_theta(curve: YieldCurve, hw: HullWhiteParams, n_months: int) -> np.ndarray:
    """
    Calibrate the time-dependent drift theta(t) so the model matches
    the initial term structure exactly.

    Under Hull-White: theta(t) = dF/dt + a*F(0,t) + sigma^2/(2a) * (1 - e^{-2at})
    where F(0,t) is the instantaneous forward rate at time t.
    """
    dt = 1.0 / hw.n_steps_per_year
    a = hw.mean_reversion
    sigma = hw.volatility
    theta = np.zeros(n_months)

    for i in range(n_months):
        t = (i + 0.5) * dt
        # Numerical forward rate from the curve
        eps = 0.001
        f_t = curve.forward_rate(max(0, t - eps), t + eps)
        # Forward rate derivative (numerical)
        f_t_plus = curve.forward_rate(max(0, t + eps - eps), t + 2 * eps)
        df_dt = (f_t_plus - f_t) / (eps)

        theta[i] = df_dt + a * f_t + (sigma ** 2) / (2 * a) * (1 - np.exp(-2 * a * t))

    return theta


def simulate_rate_paths(
    curve: YieldCurve,
    config: MonteCarloConfig,
) -> np.ndarray:
    """
    Simulate short rate paths using the Hull-White model.

    Returns:
        rates: array of shape (n_paths, n_months) with monthly short rates
    """
    hw = config.hw_params
    dt = 1.0 / hw.n_steps_per_year
    n_months = config.n_months
    n_paths = config.n_paths

    if config.seed is not None:
        rng = np.random.RandomState(config.seed)
    else:
        rng = np.random.RandomState()

    theta = _calibrate_theta(curve, hw, n_months)

    # Initial short rate from the curve
    r0 = curve.spot_rate(dt)

    if config.antithetic:
        half_paths = n_paths // 2
        z = rng.standard_normal((half_paths, n_months))
        z = np.vstack([z, -z])  # Antithetic pairs
        if n_paths % 2 == 1:
            z = np.vstack([z, rng.standard_normal((1, n_months))])
    else:
        z = rng.standard_normal((n_paths, n_months))

    rates = np.zeros((n_paths, n_months))
    rates[:, 0] = r0

    for i in range(1, n_months):
        dr = (theta[i] - hw.mean_reversion * rates[:, i - 1]) * dt + \
             hw.volatility * np.sqrt(dt) * z[:, i]
        rates[:, i] = rates[:, i - 1] + dr

    return rates


def _mortgage_rate_from_short_rate(short_rate: float, spread: float = 0.0226) -> float:
    """
    Estimate mortgage rate from the short rate path.
    Mortgage rate ~ 10Y equivalent + spread.
    Spread calibrated from FRED PMMS vs DGS10 (2015-2026 avg: 226bps).
    Term premium calibrated near zero in current regime (Fed at 3.50-3.75%, 10Y at 4.22%).
    """
    term_premium = 0.001  # ~1bp — calibrated from FRED (FEDFUNDS vs DGS10)
    return short_rate + term_premium + spread


def _project_cashflows_on_path(
    balance: float,
    wac: float,
    coupon: float,
    wam: int,
    wala: int,
    rate_path: np.ndarray,
    prepay_model: PrepaymentModel,
) -> np.ndarray:
    """
    Project cash flows along a single interest rate path.

    Returns:
        total_cashflows array
    """
    n_months = len(rate_path)
    monthly_coupon = coupon / 12.0
    monthly_wac = wac / 12.0
    cashflows = np.zeros(n_months)
    remaining = min(wam, n_months)
    current_balance = balance
    original_balance = balance

    for m in range(remaining):
        if current_balance <= 0.01:
            break

        # Interest to investor
        interest = current_balance * monthly_coupon

        # Scheduled principal (level payment amortization)
        rem = remaining - m
        if rem > 0 and monthly_wac > 0:
            payment = current_balance * monthly_wac / (1.0 - (1.0 + monthly_wac) ** (-rem))
            sched_principal = payment - current_balance * monthly_wac
            sched_principal = max(0, min(sched_principal, current_balance))
        else:
            sched_principal = current_balance

        after_sched = current_balance - sched_principal

        # Rate-dependent prepayment
        mortgage_rate = _mortgage_rate_from_short_rate(rate_path[m])
        smm = prepay_model.project_smm(
            month=m + 1,
            wac=wac,
            current_mortgage_rate=mortgage_rate,
            pool_factor=current_balance / original_balance if original_balance > 0 else 1.0,
            loan_age=wala,
        )

        prepay = after_sched * smm
        total_principal = sched_principal + prepay
        cashflows[m] = interest + total_principal
        current_balance = after_sched - prepay

    return cashflows


def compute_oas(
    market_price: float,
    par_balance: float,
    wac: float,
    coupon: float,
    wam: int,
    wala: int,
    curve: YieldCurve,
    config: Optional[MonteCarloConfig] = None,
    prepay_config: Optional[PrepaymentModelConfig] = None,
) -> dict:
    """
    Compute the Option-Adjusted Spread (OAS) using Monte Carlo simulation.

    Implements equation (1) from Fuster, Lucca & Vickery (2022):
        P_M = E[ sum_k X_k(r_k) / prod_j(1 + OAS + r_j) ]

    Args:
        market_price: Market price per 100 par
        par_balance: Par/face value of the MBS
        wac: Weighted average coupon (decimal, e.g., 0.055)
        coupon: Pass-through coupon (decimal, e.g., 0.05)
        wam: Weighted average maturity in months
        wala: Weighted average loan age in months
        curve: Yield curve for calibration
        config: Monte Carlo configuration
        prepay_config: Prepayment model configuration

    Returns:
        Dictionary with OAS, ZVS, option cost, and path statistics
    """
    if config is None:
        config = MonteCarloConfig(n_months=wam)
    if prepay_config is None:
        prepay_config = PrepaymentModelConfig()

    config.n_months = min(config.n_months, wam)
    prepay_model = PrepaymentModel(config=prepay_config)

    # Target present value
    target_pv = market_price / 100.0 * par_balance

    # Simulate rate paths
    rate_paths = simulate_rate_paths(curve, config)
    dt = 1.0 / config.hw_params.n_steps_per_year
    n_paths = rate_paths.shape[0]
    n_months = rate_paths.shape[1]

    # Project cash flows on each path
    all_cashflows = np.zeros((n_paths, n_months))
    path_wals = np.zeros(n_paths)

    for p in range(n_paths):
        cfs = _project_cashflows_on_path(
            balance=par_balance,
            wac=wac,
            coupon=coupon,
            wam=wam,
            wala=wala,
            rate_path=rate_paths[p],
            prepay_model=prepay_model,
        )
        all_cashflows[p, :len(cfs)] = cfs[:n_months]

        # WAL for this path
        total_prin = np.sum(cfs[:n_months])
        if total_prin > 0:
            months_arr = np.arange(1, n_months + 1)
            path_wals[p] = np.sum(months_arr * cfs[:n_months]) / (12.0 * total_prin)

    def model_price(oas_annual):
        """Compute average model price across paths for a given OAS."""
        oas_monthly = oas_annual / 12.0
        total_pv = 0.0

        for p in range(n_paths):
            pv = 0.0
            cum_discount = 1.0
            for m in range(n_months):
                monthly_rate = rate_paths[p, m] / 12.0
                cum_discount *= 1.0 / (1.0 + oas_monthly + monthly_rate)
                pv += all_cashflows[p, m] * cum_discount
            total_pv += pv

        return total_pv / n_paths

    # Solve for OAS
    def objective(oas):
        return model_price(oas) - target_pv

    try:
        oas = brentq(objective, -0.05, 0.10, xtol=1e-7, maxiter=100)
    except (ValueError, RuntimeError):
        try:
            oas = brentq(objective, -0.10, 0.20, xtol=1e-6, maxiter=200)
        except (ValueError, RuntimeError):
            oas = 0.0

    # Compute Zero-Volatility Spread (ZVS) - single expected path discounting
    mean_cfs = np.mean(all_cashflows, axis=0)

    def zvs_price(zvs):
        pv = 0.0
        for m in range(n_months):
            t = (m + 1) * dt
            spot = curve.spot_rate(t)
            df = np.exp(-(spot + zvs) * t)
            pv += mean_cfs[m] * df
        return pv

    def zvs_objective(zvs):
        return zvs_price(zvs) - target_pv

    try:
        zvs = brentq(zvs_objective, -0.05, 0.10, xtol=1e-7)
    except (ValueError, RuntimeError):
        zvs = 0.0

    option_cost = zvs - oas

    # Path statistics
    path_pvs = np.zeros(n_paths)
    for p in range(n_paths):
        pv = 0.0
        cum_discount = 1.0
        for m in range(n_months):
            monthly_rate = rate_paths[p, m] / 12.0
            cum_discount *= 1.0 / (1.0 + oas / 12.0 + monthly_rate)
            pv += all_cashflows[p, m] * cum_discount
        path_pvs[p] = pv

    return {
        "oas_bps": round(oas * 10000, 1),
        "zvs_bps": round(zvs * 10000, 1),
        "option_cost_bps": round(option_cost * 10000, 1),
        "model_price": round(model_price(oas) / par_balance * 100, 4),
        "market_price": market_price,
        "n_paths": n_paths,
        "n_months": n_months,
        "avg_wal_years": round(float(np.mean(path_wals)), 2),
        "wal_std_years": round(float(np.std(path_wals)), 2),
        "avg_path_pv": round(float(np.mean(path_pvs)), 2),
        "pv_std": round(float(np.std(path_pvs)), 2),
        "pv_5th_pctile": round(float(np.percentile(path_pvs, 5)), 2),
        "pv_95th_pctile": round(float(np.percentile(path_pvs, 95)), 2),
    }


def compute_oas_smile(
    coupons: list[float],
    mortgage_rate: float,
    curve: YieldCurve,
    par_balance: float = 1_000_000,
    wam: int = 360,
    wala: int = 0,
    config: Optional[MonteCarloConfig] = None,
) -> list[dict]:
    """
    Compute the OAS smile across different coupon rates (moneyness levels).

    From Boyarchenko, Fuster & Lucca (2019):
    - OAS is lowest for at-the-money MBS
    - OAS increases for both in-the-money and out-of-the-money MBS
    - This "smile" reflects prepayment risk premia

    Args:
        coupons: List of pass-through coupon rates (e.g., [3.0, 3.5, 4.0, ...])
        mortgage_rate: Current prevailing mortgage rate
        curve: Yield curve
        par_balance: Notional for pricing
        wam: Weighted average maturity
        wala: Weighted average loan age
        config: MC config (uses fewer paths for speed)
    """
    if config is None:
        config = MonteCarloConfig(n_paths=200, n_months=wam)

    results = []
    for cpn in coupons:
        wac = cpn + 0.005  # WAC ~ coupon + 50bps for servicing/gfee
        moneyness = wac - mortgage_rate

        # Estimate market price from moneyness
        duration_approx = 5.0
        price = 100.0 + moneyness * 100 * duration_approx
        price = max(85.0, min(112.0, price))

        oas_result = compute_oas(
            market_price=price,
            par_balance=par_balance,
            wac=wac,
            coupon=cpn,
            wam=wam,
            wala=wala,
            curve=curve,
            config=config,
        )

        results.append({
            "coupon": cpn,
            "wac": round(wac, 3),
            "moneyness_bps": round(moneyness * 10000, 0),
            "estimated_price": round(price, 3),
            "oas_bps": oas_result["oas_bps"],
            "zvs_bps": oas_result["zvs_bps"],
            "option_cost_bps": oas_result["option_cost_bps"],
            "avg_wal_years": oas_result["avg_wal_years"],
        })

    return results


def rate_path_scenario_analysis(
    curve: YieldCurve,
    scenarios: dict[str, float],
    par_balance: float = 1_000_000,
    coupon: float = 0.055,
    wac: float = 0.06,
    wam: int = 360,
    wala: int = 0,
    market_price: float = 100.0,
    config: Optional[MonteCarloConfig] = None,
) -> dict:
    """
    Run OAS analysis under different volatility scenarios.

    Args:
        scenarios: Dict of scenario name -> vol multiplier
                   e.g., {"low_vol": 0.5, "base": 1.0, "high_vol": 2.0}
    """
    if config is None:
        config = MonteCarloConfig(n_paths=200, n_months=wam)

    results = {}
    base_vol = config.hw_params.volatility

    for name, vol_mult in scenarios.items():
        config.hw_params.volatility = base_vol * vol_mult
        oas_result = compute_oas(
            market_price=market_price,
            par_balance=par_balance,
            wac=wac,
            coupon=coupon,
            wam=wam,
            wala=wala,
            curve=curve,
            config=config,
        )
        results[name] = oas_result

    # Restore
    config.hw_params.volatility = base_vol
    return results
