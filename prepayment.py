"""
Prepayment models for mortgage-backed securities.
Implements PSA, CPR/SMM, a rate-incentive-based prepayment model,
and the Stanton (1995) rational prepayment model with heterogeneous
transaction costs and endogenous burnout.

References:
- Stanton, R. (1995). "Rational Prepayment and the Valuation of
  Mortgage-Backed Securities." Review of Financial Studies 8(3), 677-708.
- Hall, A. and Maingi, R.Q. (2021). "The Behavioral Relationship Between
  Mortgage Prepayment and Default." Philly Fed WP 21-12.
"""
import numpy as np
from scipy.stats import beta as beta_dist
from dataclasses import dataclass
from typing import Optional


def smm_from_cpr(cpr: float) -> float:
    """Convert annual CPR to monthly SMM."""
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def cpr_from_smm(smm: float) -> float:
    """Convert monthly SMM to annual CPR."""
    return 1.0 - (1.0 - smm) ** 12.0


def psa_cpr(month: int, psa_speed: float = 100.0) -> float:
    """
    Calculate CPR under PSA (Public Securities Association) model.
    100 PSA: CPR ramps from 0.2% at month 1 to 6% at month 30, flat after.
    """
    base_cpr = min(month * 0.002, 0.06)  # 0.2% per month, max 6%
    return base_cpr * (psa_speed / 100.0)


def psa_smm(month: int, psa_speed: float = 100.0) -> float:
    """Calculate SMM under PSA model."""
    return smm_from_cpr(psa_cpr(month, psa_speed))


@dataclass
class PrepaymentModelConfig:
    """Configuration for the rate-incentive prepayment model."""
    base_cpr: float = 0.06          # Base CPR (6% = 100 PSA plateau)
    burnout_factor: float = 0.02      # Monthly burnout rate
    seasonality: bool = True          # Apply seasonal adjustment
    housing_turnover: float = 0.08    # Annual housing turnover rate (8%)
    curtailment_rate: float = 0.005   # Monthly curtailment rate
    min_cpr: float = 0.02            # Floor CPR
    max_cpr: float = 0.70            # Cap CPR


# Monthly seasonal factors (Jan=1, Dec=12)
# Blended from Fannie Mae CAS (797M rows, weight 0.81) + CIRT (183M rows, weight 0.19)
# Pattern: summer/fall peak (Aug-Nov) = moving season, winter trough (Dec-Feb)
SEASONAL_FACTORS = [
    0.954, 0.961, 0.977, 0.973, 0.981, 0.987,
    1.000, 1.038, 1.044, 1.056, 1.073, 0.957
]

# CAS cumulative CPR by loan age (months 0-180) — 10.77M loans, 797M rows, 2009-2024 vintages
# These are CUMULATIVE percentages (not marginal rates) — use for validation, not as multipliers
CAS_SEASONING_CPR = [
    0.0, 0.0, 0.002, 0.017, 0.077, 0.199, 0.481, 0.894,
    1.259, 1.640, 2.139, 2.776, 3.623, 4.564, 5.558, 6.525,
    7.491, 8.333, 9.114, 9.854, 10.669, 11.515, 12.454, 13.458,
    14.506, 15.525, 16.550, 17.590, 18.618, 19.641, 20.666, 21.696,
    22.723, 23.727, 24.743, 25.794, 26.844, 27.905, 28.945, 29.932,
    30.875, 31.831, 32.800, 33.793, 34.842, 35.872, 36.939, 38.044,
    39.051, 40.118, 41.103, 42.079, 43.174, 44.373, 45.613, 46.997,
    48.504, 49.953, 51.448, 53.003, 54.397, 55.913, 57.042, 57.708,
    58.369, 59.023, 59.659, 60.305, 60.867, 61.342, 61.756, 62.273,
    62.830, 63.366, 63.892, 64.504, 65.056, 65.429, 65.780, 65.997,
    66.153, 66.326, 66.551, 66.440, 66.137, 65.876, 65.704, 65.631,
    65.571, 65.541, 65.515, 65.439, 65.411, 65.355, 65.359, 65.435,
    65.491, 65.624, 65.733, 65.895, 65.919, 66.011, 65.940, 65.775,
    65.717, 65.581, 65.720, 66.022, 66.394, 66.821, 67.232, 67.597,
    67.798, 68.200, 68.525, 68.741, 69.113, 69.081, 69.318, 69.745,
    69.988, 70.195, 70.123, 69.804, 69.424, 69.115, 68.861, 68.676,
    68.556, 68.161, 67.884, 67.650, 67.499, 67.202, 66.069, 62.300,
    56.082, 51.275, 50.274, 50.701, 51.289, 51.869, 52.444, 53.002,
    53.533, 54.050, 54.538, 55.019, 55.488, 55.948, 56.409, 56.867,
    57.316, 57.776, 58.231, 58.676, 59.119, 59.738, 61.130, 62.021,
    62.448, 62.904, 63.240, 63.521, 64.020, 64.394, 64.641, 64.945,
    65.390, 65.712, 65.789, 65.826, 65.933, 66.302, 66.492, 67.028,
    67.732, 68.273, 69.469, 71.263,
]

# CIRT cumulative CPR by loan age (months 0-60) — 3.66M loans, 183M rows, 2014-2026 vintages
# Faster prepayment curve than CAS due to 2020-2021 refi wave bias
CIRT_SEASONING_CPR = [
    0.0, 0.27, 1.19, 2.14, 3.48, 5.68, 9.96,
    15.29, 19.56, 23.24, 27.82, 33.10, 39.44, 45.82,
    51.98, 57.57, 62.46, 66.97, 70.89, 74.24, 77.37,
    80.16, 82.79, 85.09, 87.14, 88.93, 90.47, 91.77,
    92.86, 93.87, 94.72, 95.37, 96.04, 96.68, 97.13,
    97.51, 97.86, 98.18, 98.47, 98.70, 98.89, 99.07,
    99.21, 99.34, 99.46, 99.56, 99.64, 99.72, 99.78,
    99.84, 99.89, 99.93, 99.95, 99.97, 99.98, 99.99,
    99.99, 99.999, 99.999, 100.0,
]

# CAS CDR seasoning curve by age (annualized %) — 559K defaults across 10.77M loans
# Peaks around month 78 at ~0.108%, more reliable than CIRT due to 12x more default events
CAS_SEASONING_CDR = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.000045, 0.000274, 0.000507,
    0.001894, 0.004437, 0.007804, 0.01135, 0.015416, 0.018398,
    0.022088, 0.025385, 0.027999, 0.030082, 0.032756, 0.034043,
    0.035756, 0.036842, 0.037463, 0.038888, 0.041123, 0.042266,
    0.043851, 0.045316, 0.046773, 0.048844, 0.050654, 0.052304,
    0.053925, 0.055246, 0.056902, 0.058378, 0.059652, 0.060868,
    0.062397, 0.063738, 0.064965, 0.066153, 0.067474, 0.06924,
    0.070277, 0.071506, 0.073392, 0.074515, 0.075814, 0.076853,
    0.077422, 0.07834, 0.079896, 0.081178, 0.081901, 0.083344,
    0.084105, 0.085153, 0.08693, 0.088455, 0.089854, 0.09195,
    0.093406, 0.094296, 0.095492, 0.096583, 0.097842, 0.098829,
    0.099865, 0.100467, 0.101286, 0.102832, 0.103567, 0.104736,
    0.106356, 0.107241, 0.107878, 0.107647, 0.108033, 0.108134,
    0.107276, 0.106629, 0.105986, 0.105336, 0.104562, 0.103172,
    0.101928, 0.101163, 0.100384, 0.099416, 0.09847, 0.09683,
    0.095547, 0.094854, 0.095069, 0.093489, 0.092777, 0.09124,
    0.090792, 0.089904, 0.088833, 0.08819, 0.087838, 0.086788,
    0.086613, 0.086174, 0.086275, 0.086986, 0.086821, 0.086762,
    0.086648, 0.085895, 0.085079, 0.084687, 0.08417, 0.084153,
    0.085619, 0.084875, 0.085043,
]

# CIRT CDR seasoning curve by age (annualized %) — 47K defaults, peaks at month 115 (2.19%)
# Higher absolute CDR than CAS because CIRT covers riskier loans selected for credit risk transfer
CIRT_SEASONING_CDR = [
    0.0, 0.0, 0.0, 0.0, 0.0015, 0.0, 0.0, 0.0,
    0.0006, 0.0019, 0.0029, 0.0038, 0.0044, 0.0072,
    0.0089, 0.0119, 0.0164, 0.0223, 0.0275, 0.0358,
    0.0413, 0.0483, 0.0592, 0.0706, 0.0837, 0.0993,
    0.1150, 0.1319, 0.1452, 0.1599, 0.1762, 0.1866,
    0.2071, 0.2221, 0.2432, 0.2611, 0.2709, 0.2946,
    0.3046, 0.3222, 0.3435, 0.3603, 0.3662, 0.3752,
    0.3858, 0.3951, 0.4138, 0.4370, 0.4557, 0.4719,
    0.4845, 0.5143, 0.5255, 0.5483, 0.5578, 0.5859,
    0.6237, 0.6491, 0.6772, 0.7179, 0.7376, 0.7603,
    0.7746, 0.8137, 0.8385, 0.8661, 0.8605, 0.8760,
    0.8942, 0.9233, 0.9536, 0.9879, 1.0659, 1.1563,
    1.2205, 1.2646, 1.3126, 1.3652, 1.4069, 1.3811,
    1.3229, 1.3708, 1.3199, 1.3234, 1.3046, 1.2972,
    1.2424, 1.3069, 1.2964, 1.3380, 1.4293, 1.5211,
    1.6530, 1.6036, 1.6832, 1.7708, 1.8331, 1.8371,
    1.8375, 1.8746, 1.8345, 1.8740, 1.8338, 1.8729,
    1.8336, 1.9122, 1.9908, 2.0689, 2.1082, 2.1094,
    2.1477, 2.1482, 2.1485, 2.1480, 2.1877, 2.1884,
    2.1490, 2.1875, 2.1480, 2.1870, 2.2260, 2.1875,
]


@dataclass
class PrepaymentModel:
    """
    Multi-factor prepayment model combining:
    - PSA seasoning ramp
    - Refinancing incentive (current rate vs coupon)
    - Burnout effect
    - Seasonal adjustment
    - Housing turnover
    """
    config: PrepaymentModelConfig = None

    def __post_init__(self):
        if self.config is None:
            self.config = PrepaymentModelConfig()

    def project_smm(self, month: int, wac: float, current_mortgage_rate: float,
                    pool_factor: float = 1.0, loan_age: int = 0) -> float:
        """
        Project SMM for a given month.

        Args:
            month: Month number in the projection (1-indexed)
            wac: Weighted average coupon of the pool
            current_mortgage_rate: Current prevailing mortgage rate
            pool_factor: Current pool factor (remaining balance / original)
            loan_age: Age of loans in months at projection start
        """
        effective_age = loan_age + month
        cfg = self.config

        # 1. Seasoning ramp — linear from 0 to 1 over 30 months (PSA-style)
        # CAS data (797M rows) confirms marginal monthly prepayment rate peaks around
        # months 24-36, consistent with PSA's 30-month ramp. CIRT cumulative CPR was
        # biased by the 2020-2021 refi wave and shouldn't be used as a multiplier.
        seasoning = min(effective_age / 30.0, 1.0)

        # 2. Refinancing incentive (S-curve / Hill function)
        # Calibrated so: +100bps→~2.4x, +150bps→~3.7x, +200bps→~4.9x, +250bps→~6x
        # Combined with base_cpr=6% + turnover, produces CPRs matching broker S-curves:
        # +100bps→~21%, +150bps→~29%, +200bps→~36%, +250bps→~43%, +300bps→~48%
        refi_incentive = wac - current_mortgage_rate
        if refi_incentive > 0:
            incentive_pct = refi_incentive * 100  # in percentage points
            # Hill function: smooth S-curve that saturates around 10x
            refi_factor = 1.0 + 10.0 * incentive_pct ** 2 / (incentive_pct ** 2 + 2.5 ** 2)
        else:
            # Negative incentive = higher rates = less prepayments (turnover only)
            refi_factor = max(0.3, 1.0 + refi_incentive * 2.0)

        # 3. Burnout - as pool factor drops, remaining borrowers are less likely to refi
        burnout = 1.0 - cfg.burnout_factor * (1.0 - pool_factor) * 10.0
        burnout = max(0.3, min(1.0, burnout))

        # 4. Seasonality
        if cfg.seasonality:
            month_of_year = ((effective_age - 1) % 12)
            seasonal = SEASONAL_FACTORS[month_of_year]
        else:
            seasonal = 1.0

        # 5. Housing turnover component
        monthly_turnover = cfg.housing_turnover / 12.0

        # Combine factors
        cpr = (cfg.base_cpr * seasoning * refi_factor * burnout * seasonal
               + monthly_turnover * seasoning
               + cfg.curtailment_rate * 12.0)

        # Apply floor and cap
        cpr = max(cfg.min_cpr, min(cfg.max_cpr, cpr))

        return smm_from_cpr(cpr)

    def project_schedule(self, n_months: int, wac: float,
                        current_mortgage_rate: float,
                        loan_age: int = 0,
                        rate_path: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project SMM schedule for n months.

        Args:
            n_months: Number of months to project
            wac: Weighted average coupon
            current_mortgage_rate: Starting mortgage rate
            loan_age: Loan age at start
            rate_path: Optional array of mortgage rates for each month
        """
        smms = np.zeros(n_months)
        pool_factor = 1.0

        for m in range(n_months):
            if rate_path is not None and m < len(rate_path):
                rate = rate_path[m]
            else:
                rate = current_mortgage_rate

            smms[m] = self.project_smm(
                month=m + 1,
                wac=wac,
                current_mortgage_rate=rate,
                pool_factor=pool_factor,
                loan_age=loan_age
            )
            pool_factor *= (1.0 - smms[m])

        return smms


def estimate_psa_speed(wac: float, current_rate: float, wala: int = 0) -> float:
    """
    Estimate PSA speed based on rate incentive and loan age.
    Quick heuristic for initial pricing.
    """
    incentive = (wac - current_rate) * 100  # in bps

    if incentive > 150:
        base_psa = 400 + (incentive - 150) * 2
    elif incentive > 50:
        base_psa = 200 + (incentive - 50) * 2
    elif incentive > 0:
        base_psa = 100 + incentive * 2
    elif incentive > -100:
        base_psa = max(50, 100 + incentive * 0.5)
    else:
        base_psa = max(25, 50 + (incentive + 100) * 0.25)

    # Seasoning adjustment
    if wala < 30:
        seasoning_adj = wala / 30.0
    else:
        seasoning_adj = 1.0

    return base_psa * seasoning_adj if wala > 0 else base_psa


def mortgage_rate_from_treasury(treasury_10y: float, spread_bps: float = 226) -> float:
    """Estimate current mortgage rate from 10Y Treasury + spread.
    Default spread calibrated from FRED PMMS vs DGS10 (2015-2026 avg: 226bps)."""
    return treasury_10y + spread_bps / 100.0


# ─── Stanton (1995) Rational Prepayment Model ──────────────────────────────

@dataclass
class StantonParams:
    """
    Parameters for the Stanton (1995) rational prepayment model.

    From Stanton's GMM estimation (Table III, p. 700):
    - rho: 0.6073 (decision frequency; avg time between decisions = 1/rho years)
    - lam: 0.0345 (exogenous prepayment hazard per year: turnover, divorce, etc.)
    - alpha: 2.9618 (beta distribution shape parameter 1 for transaction costs)
    - beta_param: 4.2268 (beta distribution shape parameter 2 for transaction costs)

    CIR interest rate model: dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
    """
    # Behavioral parameters (Stanton Table III estimates)
    rho: float = 0.6073        # Decision frequency (per year)
    lam: float = 0.0345        # Exogenous prepayment hazard (per year)
    alpha: float = 2.9618      # Beta distribution param 1 (transaction costs)
    beta_param: float = 4.2268 # Beta distribution param 2 (transaction costs)
    x_max: float = 0.40        # Maximum transaction cost as fraction of balance

    # CIR short rate model parameters
    kappa: float = 0.2639      # Mean reversion speed
    theta: float = 0.0808      # Long-run mean of short rate
    sigma: float = 0.0854      # Volatility of short rate

    # Mortgage parameters
    coupon_spread: float = 0.017  # Mortgage rate = short rate + term premium + spread
    term_premium: float = 0.015   # Term premium over short rate

    # Discretization
    n_cost_buckets: int = 20   # Number of transaction cost buckets
    n_rate_steps: int = 50     # Number of rate grid points for PDE

    @property
    def beta_mean(self) -> float:
        """Mean of the beta distribution of transaction costs."""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def beta_var(self) -> float:
        """Variance of the beta distribution of transaction costs."""
        ab = self.alpha + self.beta_param
        return self.alpha * self.beta_param / (ab ** 2 * (ab + 1))


class StantonPrepaymentModel:
    """
    Rational prepayment model from Stanton (1995).

    Key features:
    - Heterogeneous transaction costs: borrowers face different costs X ~ Beta(alpha, beta)
      scaled by x_max. This creates a distribution of refinancing thresholds.
    - Discrete decision frequency: borrowers don't continuously monitor rates. They
      reconsider refinancing at Poisson-distributed intervals with rate rho.
    - Exogenous prepayment: housing turnover, relocation, divorce etc. at rate lambda.
    - Endogenous burnout: as rates fall, low-cost borrowers refinance first, leaving
      higher-cost borrowers in the pool. No ad hoc burnout factor needed.

    The model solves for the optimal prepayment boundary for each cost bucket,
    then aggregates across the distribution to get pool-level prepayment rates.
    """

    def __init__(self, params: Optional[StantonParams] = None):
        self.params = params or StantonParams()
        self._cost_grid = None
        self._cost_weights = None
        self._setup_cost_grid()

    def _setup_cost_grid(self):
        """Set up the discretized beta distribution of transaction costs."""
        p = self.params
        n = p.n_cost_buckets

        # Gauss-Legendre quadrature points on [0, 1] mapped to [0, x_max]
        # Use midpoints of equal-probability bins for simplicity
        quantiles = np.linspace(0.5 / n, 1 - 0.5 / n, n)
        self._cost_grid = beta_dist.ppf(quantiles, p.alpha, p.beta_param) * p.x_max

        # Equal weights (each bucket has probability 1/n of the beta distribution)
        self._cost_weights = np.ones(n) / n

    def mortgage_value(self, short_rate: float, wac: float, remaining_months: int) -> float:
        """
        Compute the value of the mortgage to the borrower.

        The mortgage value is the PV of remaining payments discounted at the
        current short rate (risk-neutral). If mortgage_value > (1+X)*balance,
        a borrower with transaction cost X should refinance.
        """
        if remaining_months <= 0:
            return 0.0

        monthly_wac = wac / 12.0
        # Monthly discount rate based on short rate + spread
        mortgage_rate = short_rate + self.params.term_premium
        monthly_discount = mortgage_rate / 12.0

        if monthly_discount <= 0:
            monthly_discount = 1e-6

        # PV of level annuity at market rate
        # Payment per dollar of remaining balance at the WAC
        if monthly_wac > 0:
            payment = monthly_wac / (1.0 - (1.0 + monthly_wac) ** (-remaining_months))
        else:
            payment = 1.0 / remaining_months

        # PV of those payments at market discount rate
        if abs(monthly_discount) < 1e-10:
            pv_factor = remaining_months
        else:
            pv_factor = (1.0 - (1.0 + monthly_discount) ** (-remaining_months)) / monthly_discount

        return payment * pv_factor

    def optimal_prepay_indicator(self, short_rate: float, wac: float,
                                  remaining_months: int, transaction_cost: float) -> bool:
        """
        Determine if a borrower with given transaction cost should prepay.

        Prepay iff: mortgage_value > (1 + X) * par
        where X is the transaction cost as fraction of balance.
        """
        mv = self.mortgage_value(short_rate, wac, remaining_months)
        return mv > (1.0 + transaction_cost)

    def project_smm(self, month: int, wac: float, short_rate: float,
                    cost_distribution: Optional[np.ndarray] = None,
                    remaining_months: int = 360,
                    loan_age: int = 0) -> tuple[float, np.ndarray]:
        """
        Project single-month mortality for the pool.

        Returns:
            (smm, updated_cost_distribution)

        The cost_distribution tracks the fraction of remaining borrowers
        in each cost bucket. As low-cost borrowers prepay, the distribution
        shifts right (endogenous burnout).
        """
        p = self.params
        n = p.n_cost_buckets
        effective_age = loan_age + month

        if cost_distribution is None:
            cost_distribution = self._cost_weights.copy()

        # Normalize distribution
        total_weight = np.sum(cost_distribution)
        if total_weight <= 1e-12:
            return 0.0, cost_distribution

        normalized = cost_distribution / total_weight

        # Seasoning ramp (PSA-style, transactions costs aside)
        seasoning = min(effective_age / 30.0, 1.0)

        # For each cost bucket, determine if rational to prepay
        smm_total = 0.0
        new_distribution = cost_distribution.copy()

        for i in range(n):
            x = self._cost_grid[i]
            w = normalized[i]

            if w < 1e-15:
                continue

            # Probability borrower reconsiders this month
            # Poisson arrival: P(at least one decision) = 1 - exp(-rho * dt)
            dt = 1.0 / 12.0
            decision_prob = 1.0 - np.exp(-p.rho * dt)

            # If borrower reconsiders, check optimal prepay
            should_prepay = self.optimal_prepay_indicator(
                short_rate, wac, remaining_months - month, x
            )

            # Refinancing rate for this bucket
            if should_prepay:
                refi_rate = decision_prob * seasoning
            else:
                refi_rate = 0.0

            # Exogenous prepayment (turnover, relocation, etc.)
            # Monthly hazard from annual rate
            exog_rate = 1.0 - np.exp(-p.lam * dt)

            # Combined monthly prepayment rate for this bucket
            bucket_smm = 1.0 - (1.0 - refi_rate) * (1.0 - exog_rate)
            bucket_smm = max(0.0, min(1.0, bucket_smm))

            smm_total += w * bucket_smm

            # Update distribution: remove prepaid borrowers
            new_distribution[i] *= (1.0 - bucket_smm)

        return smm_total, new_distribution

    def project_schedule(self, n_months: int, wac: float, initial_short_rate: float,
                         rate_path: Optional[np.ndarray] = None,
                         loan_age: int = 0) -> dict:
        """
        Project prepayment schedule over n months with endogenous burnout.

        Args:
            n_months: Projection horizon
            wac: Weighted average coupon (decimal, e.g. 0.055)
            initial_short_rate: Starting short rate
            rate_path: Optional array of monthly short rates
            loan_age: Starting loan age in months

        Returns:
            Dict with SMM schedule, CPR schedule, pool factor, and
            cost distribution evolution.
        """
        smms = np.zeros(n_months)
        cprs = np.zeros(n_months)
        pool_factors = np.zeros(n_months)
        pool_factor = 1.0

        # Initial cost distribution (fresh pool = beta distribution)
        cost_dist = self._cost_weights.copy()

        # Track distribution evolution for diagnostics
        dist_history = [cost_dist.copy()]

        for m in range(n_months):
            if pool_factor < 1e-6:
                break

            if rate_path is not None and m < len(rate_path):
                r = rate_path[m]
            else:
                r = initial_short_rate

            smm, cost_dist = self.project_smm(
                month=m + 1,
                wac=wac,
                short_rate=r,
                cost_distribution=cost_dist,
                remaining_months=int(360 - loan_age),
                loan_age=loan_age,
            )

            smms[m] = smm
            cprs[m] = cpr_from_smm(smm)
            pool_factor *= (1.0 - smm)
            pool_factors[m] = pool_factor

            if (m + 1) % 12 == 0:
                dist_history.append(cost_dist.copy())

        return {
            "smm": smms,
            "cpr": cprs,
            "pool_factor": pool_factors,
            "cost_distribution_annual": dist_history,
            "avg_cpr_year1": float(np.mean(cprs[:12])) if n_months >= 12 else 0,
            "avg_cpr_year3": float(np.mean(cprs[24:36])) if n_months >= 36 else 0,
            "avg_cpr_year5": float(np.mean(cprs[48:60])) if n_months >= 60 else 0,
            "terminal_pool_factor": float(pool_factors[min(n_months - 1, len(pool_factors) - 1)]),
        }

    def simulate_cir_rates(self, n_months: int, r0: float,
                            n_paths: int = 500, seed: int = 42) -> np.ndarray:
        """
        Simulate CIR short rate paths for Monte Carlo analysis.

        dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW

        Returns:
            rates: shape (n_paths, n_months)
        """
        p = self.params
        dt = 1.0 / 12.0
        rng = np.random.RandomState(seed)

        rates = np.zeros((n_paths, n_months))
        rates[:, 0] = r0

        z = rng.standard_normal((n_paths, n_months))

        for t in range(1, n_months):
            r_prev = np.maximum(rates[:, t - 1], 0.0)
            drift = p.kappa * (p.theta - r_prev) * dt
            diffusion = p.sigma * np.sqrt(r_prev * dt) * z[:, t]
            rates[:, t] = np.maximum(r_prev + drift + diffusion, 0.0)

        return rates

    def monte_carlo_prepayment(self, wac: float, r0: float,
                                 n_months: int = 360,
                                 n_paths: int = 200,
                                 loan_age: int = 0,
                                 seed: int = 42) -> dict:
        """
        Run Monte Carlo prepayment analysis using CIR rate paths.

        For each path, projects prepayments with endogenous burnout.
        Returns distribution of prepayment outcomes.
        """
        rate_paths = self.simulate_cir_rates(n_months, r0, n_paths, seed)

        all_cprs = np.zeros((n_paths, n_months))
        all_factors = np.zeros((n_paths, n_months))
        path_wals = np.zeros(n_paths)

        for p in range(n_paths):
            result = self.project_schedule(
                n_months=n_months,
                wac=wac,
                initial_short_rate=r0,
                rate_path=rate_paths[p],
                loan_age=loan_age,
            )
            all_cprs[p] = result["cpr"]
            all_factors[p] = result["pool_factor"]

            # WAL calculation
            principal = np.diff(np.concatenate([[1.0], result["pool_factor"]]))
            principal = -principal  # make positive
            total_prin = np.sum(principal)
            if total_prin > 0:
                months_arr = np.arange(1, n_months + 1)
                path_wals[p] = np.sum(months_arr * principal) / (12.0 * total_prin)

        return {
            "avg_cpr_by_month": np.mean(all_cprs, axis=0),
            "std_cpr_by_month": np.std(all_cprs, axis=0),
            "avg_pool_factor": np.mean(all_factors, axis=0),
            "avg_wal_years": float(np.mean(path_wals)),
            "wal_std_years": float(np.std(path_wals)),
            "wal_5th_pctile": float(np.percentile(path_wals, 5)),
            "wal_95th_pctile": float(np.percentile(path_wals, 95)),
            "n_paths": n_paths,
            "params": {
                "rho": self.params.rho,
                "lambda": self.params.lam,
                "alpha": self.params.alpha,
                "beta": self.params.beta_param,
                "kappa": self.params.kappa,
                "theta": self.params.theta,
                "sigma": self.params.sigma,
            },
        }


# ─── Hall & Maingi (2021) Disguised Default / Prepayment Interaction ────────

@dataclass
class DisguisedDefaultParams:
    """
    Parameters for the Hall & Maingi (2021) prepayment-default interaction model.

    Key insight: Under the dual-trigger theory, borrowers experiencing liquidity
    shocks will PREPAY if they have positive equity, or DEFAULT if they have
    negative equity. This creates a "disguised default" prepayment channel.

    During rising house prices, defaults are suppressed as liquidity-shocked
    borrowers prepay instead. When prices reverse, these prepayments transform
    into defaults.
    """
    # Liquidity shock parameters
    base_shock_rate: float = 0.03       # Annual probability of liquidity shock (3%)
    unemployment_sensitivity: float = 2.5  # Multiplier per % unemployment increase
    # Equity threshold
    equity_threshold: float = 0.0       # LTV at which behavior switches (100%)
    # Ruthless default parameters (from Table 10)
    strategic_default_ltv: float = 1.20  # LTV above which strategic default kicks in
    strategic_default_rate: float = 0.02 # Additional annual CDR from strategic behavior


def compute_disguised_default_adjustment(
    current_ltv: float,
    unemployment_change: float = 0.0,
    hpa_annual: float = 0.03,
    loan_age_months: int = 24,
    params: Optional[DisguisedDefaultParams] = None,
) -> dict:
    """
    Compute the adjustment to prepayment and default rates from the
    Hall & Maingi disguised default channel.

    When equity is positive (LTV < 100%), liquidity shocks cause prepayment.
    When equity is negative (LTV > 100%), same shocks cause default.

    Returns adjustment to CPR and CDR.
    """
    if params is None:
        params = DisguisedDefaultParams()

    # Liquidity shock probability (annual)
    unemp_mult = max(0.5, 1.0 + params.unemployment_sensitivity * unemployment_change / 100)
    shock_prob = params.base_shock_rate * unemp_mult

    # Seasoning: shocks more impactful for younger loans
    seasoning = min(loan_age_months / 36.0, 1.0)
    shock_prob *= seasoning

    if current_ltv < 100:
        # Positive equity: liquidity shock -> prepayment (disguised default)
        equity_cushion = (100 - current_ltv) / 100
        # Higher equity = more likely to prepay vs default
        prepay_fraction = min(1.0, equity_cushion * 5)
        cpr_adjustment = shock_prob * prepay_fraction
        cdr_adjustment = 0.0
    else:
        # Negative equity: liquidity shock -> default
        underwater_depth = (current_ltv - 100) / 100
        default_fraction = min(1.0, 0.5 + underwater_depth * 2)
        cdr_adjustment = shock_prob * default_fraction
        cpr_adjustment = shock_prob * (1 - default_fraction)

        # Add strategic default component (Table 10 evidence)
        if current_ltv > params.strategic_default_ltv * 100:
            cdr_adjustment += params.strategic_default_rate

    return {
        "cpr_adjustment": cpr_adjustment,
        "cdr_adjustment": cdr_adjustment,
        "liquidity_shock_prob": shock_prob,
        "current_ltv": current_ltv,
        "channel": "disguised_default_prepay" if current_ltv < 100 else "default",
    }
