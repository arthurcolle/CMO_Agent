"""
Specified Pool (Spec Pool) definitions and mortgage pool cash flow projection.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .prepayment import PrepaymentModel, psa_smm, estimate_psa_speed


class AgencyType(Enum):
    GNMA = "GNMA"      # Ginnie Mae (Government)
    FNMA = "FNMA"      # Fannie Mae (GSE)
    FHLMC = "FHLMC"    # Freddie Mac (GSE)


class CollateralType(Enum):
    G2 = "G2"          # Ginnie Mae II
    G1 = "G1"          # Ginnie Mae I
    FN = "FN"          # Fannie Mae
    FH = "FH"          # Freddie Mac
    GN = "GN"          # Ginnie Mae (generic)


class PoolCharacteristic(Enum):
    """Spec pool story types that command payups over TBA."""
    TBA = "TBA"
    LOW_LOAN_BALANCE = "LLB"
    HIGH_LTV = "HLTV"
    NEW_YORK = "NY"
    INVESTOR = "INV"
    LOW_FICO = "LFICO"
    GEO_CONCENTRATED = "GEO"
    HIGH_WAC = "HWAC"
    CUSTOM = "CUSTOM"


@dataclass
class SpecPool:
    """Represents a specified mortgage pool."""
    pool_id: str
    agency: AgencyType
    collateral_type: CollateralType
    coupon: float              # Pass-through coupon rate (e.g., 5.5)
    wac: float                 # Weighted average coupon
    wam: int                   # Weighted average maturity (months)
    wala: int = 0              # Weighted average loan age (months)
    original_balance: float = 0.0
    current_balance: float = 0.0
    pool_factor: float = 1.0
    original_term: int = 360   # 30-year = 360, 15-year = 180
    characteristic: PoolCharacteristic = PoolCharacteristic.TBA
    avg_loan_size: float = 300000.0
    avg_fico: float = 740.0
    avg_ltv: float = 80.0
    num_loans: int = 0
    geography: str = ""
    servicer: str = ""
    payup_32nds: float = 0.0   # Payup over TBA in 32nds

    def __post_init__(self):
        if self.current_balance == 0:
            self.current_balance = self.original_balance
        if self.num_loans == 0 and self.avg_loan_size > 0:
            self.num_loans = max(1, int(self.original_balance / self.avg_loan_size))

    @property
    def remaining_term(self) -> int:
        return self.wam

    @property
    def factor(self) -> float:
        if self.original_balance > 0:
            return self.current_balance / self.original_balance
        return self.pool_factor


@dataclass
class PoolCashFlows:
    """Projected cash flows from a mortgage pool."""
    months: np.ndarray
    beginning_balance: np.ndarray
    scheduled_principal: np.ndarray
    prepaid_principal: np.ndarray
    total_principal: np.ndarray
    interest: np.ndarray
    total_cash_flow: np.ndarray
    smm: np.ndarray
    pool_factor: np.ndarray
    ending_balance: np.ndarray

    @property
    def wal(self) -> float:
        """Weighted average life in years."""
        total_prin = np.sum(self.total_principal)
        if total_prin == 0:
            return 0.0
        return float(np.sum(self.months * self.total_principal) / (12.0 * total_prin))

    @property
    def total_interest(self) -> float:
        return float(np.sum(self.interest))

    @property
    def total_principal_paid(self) -> float:
        return float(np.sum(self.total_principal))

    @property
    def duration_months(self) -> int:
        """Number of months until pool is paid down."""
        nonzero = np.where(self.ending_balance > 0.01)[0]
        return int(nonzero[-1] + 1) if len(nonzero) > 0 else 0


def project_pool_cashflows(
    pool: SpecPool,
    n_months: Optional[int] = None,
    psa_speed: Optional[float] = None,
    prepay_model: Optional[PrepaymentModel] = None,
    current_mortgage_rate: Optional[float] = None,
    rate_path: Optional[np.ndarray] = None,
) -> PoolCashFlows:
    """
    Project monthly cash flows for a mortgage pool.

    Args:
        pool: The specified pool
        n_months: Number of months to project (default: remaining term)
        psa_speed: PSA speed assumption (if not using full model)
        prepay_model: Full prepayment model (overrides psa_speed)
        current_mortgage_rate: Current prevailing mortgage rate
        rate_path: Path of future mortgage rates
    """
    if n_months is None:
        n_months = pool.wam

    monthly_coupon = pool.coupon / 100.0 / 12.0
    monthly_wac = pool.wac / 100.0 / 12.0

    # Servicing spread (WAC - coupon)
    # servicing = (pool.wac - pool.coupon) / 100.0 / 12.0

    months = np.arange(1, n_months + 1, dtype=float)
    beginning_balance = np.zeros(n_months)
    scheduled_principal = np.zeros(n_months)
    prepaid_principal = np.zeros(n_months)
    total_principal = np.zeros(n_months)
    interest = np.zeros(n_months)
    total_cf = np.zeros(n_months)
    smm_arr = np.zeros(n_months)
    pool_factor = np.zeros(n_months)
    ending_balance = np.zeros(n_months)

    balance = pool.current_balance
    remaining = pool.wam

    for m in range(n_months):
        if balance <= 0.01 or remaining <= 0:
            break

        beginning_balance[m] = balance

        # Scheduled payment (level payment amortization on WAC)
        if remaining > 0:
            payment = balance * monthly_wac / (1.0 - (1.0 + monthly_wac) ** (-remaining))
        else:
            payment = balance

        # Interest (at pass-through coupon rate)
        int_payment = balance * monthly_coupon
        interest[m] = int_payment

        # Scheduled principal
        sched_prin = payment - balance * monthly_wac
        sched_prin = max(0, min(sched_prin, balance))
        scheduled_principal[m] = sched_prin

        after_sched = balance - sched_prin

        # Prepayment
        if prepay_model is not None and current_mortgage_rate is not None:
            smm = prepay_model.project_smm(
                month=m + 1,
                wac=pool.wac / 100.0,
                current_mortgage_rate=current_mortgage_rate / 100.0,
                pool_factor=balance / pool.original_balance if pool.original_balance > 0 else 1.0,
                loan_age=pool.wala
            )
        elif psa_speed is not None:
            smm = psa_smm(pool.wala + m + 1, psa_speed)
        else:
            smm = psa_smm(pool.wala + m + 1, 100.0)

        smm_arr[m] = smm
        prepay = after_sched * smm
        prepaid_principal[m] = prepay

        total_prin = sched_prin + prepay
        total_principal[m] = total_prin
        total_cf[m] = int_payment + total_prin

        balance = after_sched - prepay
        ending_balance[m] = max(0, balance)
        pool_factor[m] = balance / pool.original_balance if pool.original_balance > 0 else 0
        remaining -= 1

    return PoolCashFlows(
        months=months,
        beginning_balance=beginning_balance,
        scheduled_principal=scheduled_principal,
        prepaid_principal=prepaid_principal,
        total_principal=total_principal,
        interest=interest,
        total_cash_flow=total_cf,
        smm=smm_arr,
        pool_factor=pool_factor,
        ending_balance=ending_balance,
    )


def price_tba(coupon: float, treasury_10y: float, agency: AgencyType = AgencyType.GNMA) -> float:
    """
    Estimate TBA price based on coupon and current rates.
    Simple model: price approaches par when coupon ~ current rate.
    """
    # Estimate current coupon rate
    spread = 1.70 if agency == AgencyType.GNMA else 1.50
    current_coupon = treasury_10y + spread

    # Price premium/discount based on coupon vs current coupon
    coupon_diff = coupon - current_coupon
    # Rough duration of ~5 years for 30yr MBS
    duration = 5.0
    price_adjustment = coupon_diff * duration

    base_price = 100.0 + price_adjustment
    return max(85.0, min(115.0, base_price))


def spec_pool_payup(pool: SpecPool, base_psa: float = 100.0) -> float:
    """
    Estimate spec pool payup in 32nds over TBA.

    Payups reflect prepayment protection value:
    - Low loan balance: slower prepays
    - High LTV: less refi ability
    - NY/geography: slower prepays
    - Low FICO: less refi access
    """
    payup = 0.0

    if pool.characteristic == PoolCharacteristic.LOW_LOAN_BALANCE:
        if pool.avg_loan_size < 85000:
            payup += 96  # 3 points in 32nds
        elif pool.avg_loan_size < 150000:
            payup += 64  # 2 points
        elif pool.avg_loan_size < 200000:
            payup += 32  # 1 point

    elif pool.characteristic == PoolCharacteristic.HIGH_LTV:
        if pool.avg_ltv > 95:
            payup += 48
        elif pool.avg_ltv > 90:
            payup += 32
        elif pool.avg_ltv > 80:
            payup += 16

    elif pool.characteristic == PoolCharacteristic.NEW_YORK:
        payup += 24  # NY foreclosure laws slow prepays

    elif pool.characteristic == PoolCharacteristic.LOW_FICO:
        if pool.avg_fico < 680:
            payup += 40
        elif pool.avg_fico < 720:
            payup += 20

    elif pool.characteristic == PoolCharacteristic.INVESTOR:
        payup += 16

    # Coupon-dependent: higher coupon pools have larger payups (more prepay protection value)
    if pool.coupon >= 6.0:
        payup *= 1.5
    elif pool.coupon >= 5.5:
        payup *= 1.2
    elif pool.coupon <= 3.0:
        payup *= 0.5

    pool.payup_32nds = payup
    return payup
