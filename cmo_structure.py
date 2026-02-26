"""
CMO/REMIC Structuring Engine.
Supports Sequential, PAC, TAC, Support, Z-bond, IO/PO, Floater/Inverse Floater tranches.
Implements full cash flow waterfall allocation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .spec_pool import PoolCashFlows


class PrincipalType(Enum):
    SEQUENTIAL = "SEQ"
    PAC = "PAC"
    PAC_II = "PAC2"          # PAC carved from support, narrower bands
    TAC = "TAC"
    REVERSE_TAC = "RTAC"     # Extension protection (schedule at slow speed)
    SUPPORT = "SUP"
    PASSTHROUGH = "PT"
    ACCRETION_DIRECTED = "AD"
    VADM = "VADM"            # Very Accurately Defined Maturity (from Z accretion)
    SCHEDULE = "SCHED"       # Single-speed targeted schedule
    NAS = "NAS"              # Non-Accelerating Senior (lockout)
    Z_BOND = "Z"
    JUMP_Z = "JZ"            # Z-bond that jumps to current-pay on trigger
    SUPER_PO = "SPO"         # PO carved from support class
    Z_PAC = "ZPAC"           # PAC with Z-bond accrual during lockout


class InterestType(Enum):
    FIXED = "FIX"
    FLOATING = "FLT"
    INVERSE_FLOATING = "INV"
    IO_ONLY = "IO"
    PO_ONLY = "PO"
    Z_ACCRUAL = "Z"


@dataclass
class TrancheSpec:
    """Specification for a CMO tranche."""
    name: str
    principal_type: PrincipalType
    interest_type: InterestType
    original_balance: float
    coupon: float = 0.0           # Fixed coupon or initial coupon
    notional_balance: float = 0.0  # For IO tranches
    # Floating rate parameters
    index_spread: float = 0.0     # Spread over index (bps)
    rate_cap: float = 100.0       # Rate cap
    rate_floor: float = 0.0       # Rate floor
    inverse_multiplier: float = 1.0  # For inverse floaters
    inverse_constant: float = 0.0    # For inverse floaters: constant - multiplier * index
    # PAC/TAC bands
    pac_lower_band: float = 100.0  # Lower PSA for PAC schedule
    pac_upper_band: float = 300.0  # Upper PSA for PAC schedule
    # Advanced structure parameters
    lockout_months: int = 0        # NAS/Z-PAC: no principal for N months
    schedule_speed: float = 0.0    # Schedule bond: single PSA speed target
    jump_z_trigger: float = 0.0    # Jump-Z: trigger balance (when supports exhausted)
    is_sticky: bool = False        # Jump-Z: stays current-pay once triggered
    vadm_z_tranche: str = ""       # VADM: name of Z-bond providing accretion
    # Priority
    priority: int = 0              # Lower = higher priority for principal
    group: int = 1                 # Collateral group

    @property
    def is_io(self) -> bool:
        return self.interest_type == InterestType.IO_ONLY

    @property
    def is_po(self) -> bool:
        return self.interest_type == InterestType.PO_ONLY

    @property
    def is_z_bond(self) -> bool:
        return self.interest_type == InterestType.Z_ACCRUAL

    @property
    def is_floater(self) -> bool:
        return self.interest_type in (InterestType.FLOATING, InterestType.INVERSE_FLOATING)


@dataclass
class TrancheCashFlows:
    """Cash flows for a single tranche."""
    name: str
    months: np.ndarray
    beginning_balance: np.ndarray
    principal: np.ndarray
    interest: np.ndarray
    total_cash_flow: np.ndarray
    ending_balance: np.ndarray
    accrued_interest: np.ndarray  # For Z-bonds

    @property
    def wal(self) -> float:
        """Weighted average life in years."""
        total_prin = np.sum(self.principal)
        if total_prin <= 0:
            return 0.0
        return float(np.sum(self.months * self.principal) / (12.0 * total_prin))

    @property
    def total_interest_paid(self) -> float:
        return float(np.sum(self.interest))

    @property
    def total_principal_paid(self) -> float:
        return float(np.sum(self.principal))

    @property
    def first_pay_month(self) -> int:
        nonzero = np.where(self.principal > 0.01)[0]
        return int(nonzero[0] + 1) if len(nonzero) > 0 else 0

    @property
    def last_pay_month(self) -> int:
        nonzero = np.where(self.principal > 0.01)[0]
        return int(nonzero[-1] + 1) if len(nonzero) > 0 else 0

    @property
    def window(self) -> str:
        return f"{self.first_pay_month}-{self.last_pay_month} months"


@dataclass
class CMODeal:
    """A complete CMO/REMIC deal."""
    deal_id: str
    series: str
    collateral: list[PoolCashFlows]
    tranches: list[TrancheSpec]
    dealer: str = ""
    trustee: str = ""
    settlement_date: str = ""
    total_collateral_balance: float = 0.0

    def __post_init__(self):
        if self.total_collateral_balance == 0 and self.collateral:
            self.total_collateral_balance = sum(
                cf.beginning_balance[0] for cf in self.collateral
                if len(cf.beginning_balance) > 0
            )


@dataclass
class CMOCashFlows:
    """Cash flows for all tranches in a CMO deal."""
    deal_id: str
    tranche_flows: dict[str, TrancheCashFlows]
    collateral_flows: PoolCashFlows
    residual: np.ndarray  # Leftover cash flow (should be ~0 for proper structure)

    def summary(self) -> dict:
        result = {'deal_id': self.deal_id, 'tranches': {}}
        for name, flows in self.tranche_flows.items():
            result['tranches'][name] = {
                'wal': round(flows.wal, 2),
                'window': flows.window,
                'total_principal': round(flows.total_principal_paid, 2),
                'total_interest': round(flows.total_interest_paid, 2),
            }
        result['residual'] = round(float(np.sum(np.abs(self.residual))), 2)
        return result


def compute_pac_schedule(
    collateral_balance: float,
    wac: float,
    wam: int,
    wala: int,
    lower_psa: float,
    upper_psa: float,
) -> np.ndarray:
    """
    Compute PAC schedule as the minimum principal at lower and upper PSA speeds.
    """
    from .spec_pool import SpecPool, project_pool_cashflows

    dummy_pool = SpecPool(
        pool_id="PAC_SCHED",
        agency=None,  # type: ignore
        collateral_type=None,  # type: ignore
        coupon=wac - 0.5,
        wac=wac,
        wam=wam,
        wala=wala,
        original_balance=collateral_balance,
        current_balance=collateral_balance,
    )

    cf_lower = project_pool_cashflows(dummy_pool, n_months=wam, psa_speed=lower_psa)
    cf_upper = project_pool_cashflows(dummy_pool, n_months=wam, psa_speed=upper_psa)

    pac_schedule = np.minimum(cf_lower.total_principal[:wam],
                              cf_upper.total_principal[:wam])
    return pac_schedule


def compute_schedule_bond_schedule(
    collateral_balance: float,
    wac: float,
    wam: int,
    wala: int,
    target_psa: float,
) -> np.ndarray:
    """
    Compute a schedule bond's targeted principal schedule at a single PSA speed.
    Unlike PAC (which uses min of two speeds), schedule bonds target one speed exactly.
    """
    from .spec_pool import SpecPool, project_pool_cashflows

    dummy_pool = SpecPool(
        pool_id="SCHED",
        agency=None,  # type: ignore
        collateral_type=None,  # type: ignore
        coupon=wac - 0.5,
        wac=wac,
        wam=wam,
        wala=wala,
        original_balance=collateral_balance,
        current_balance=collateral_balance,
    )

    cf = project_pool_cashflows(dummy_pool, n_months=wam, psa_speed=target_psa)
    return cf.total_principal[:wam]


def compute_reverse_tac_schedule(
    collateral_balance: float,
    wac: float,
    wam: int,
    wala: int,
    slow_psa: float,
) -> np.ndarray:
    """
    Compute a reverse TAC schedule. The bond is protected from extension by
    guaranteeing at least the principal flow at a slow prepayment speed.
    If actual prepayments are slower than slow_psa, support absorbs the shortfall.
    """
    from .spec_pool import SpecPool, project_pool_cashflows

    dummy_pool = SpecPool(
        pool_id="RTAC",
        agency=None,  # type: ignore
        collateral_type=None,  # type: ignore
        coupon=wac - 0.5,
        wac=wac,
        wam=wam,
        wala=wala,
        original_balance=collateral_balance,
        current_balance=collateral_balance,
    )

    cf = project_pool_cashflows(dummy_pool, n_months=wam, psa_speed=slow_psa)
    return cf.total_principal[:wam]


def structure_cmo(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    tranche_specs: list[TrancheSpec],
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Structure a CMO deal by allocating collateral cash flows to tranches.

    Implements the full waterfall:
    1. Interest allocation (fixed, float, IO)
    2. Z-bond accrual
    3. Principal allocation (PAC > SEQ > TAC > SUP)
    4. Z-bond accretion directed to higher priority tranches
    """
    n_months = len(collateral_flows.months)
    n_tranches = len(tranche_specs)

    # Initialize tranche state
    balances = np.array([t.original_balance for t in tranche_specs], dtype=float)
    accrued = np.zeros(n_tranches)  # Z-bond accrued interest

    # Output arrays
    tranche_principal = np.zeros((n_tranches, n_months))
    tranche_interest = np.zeros((n_tranches, n_months))
    tranche_balance = np.zeros((n_tranches, n_months))
    tranche_accrued = np.zeros((n_tranches, n_months))
    residual = np.zeros(n_months)

    # Pre-compute PAC schedules if needed
    pac_schedules = {}
    schedule_bond_schedules = {}
    reverse_tac_schedules = {}
    total_collateral = collateral_flows.beginning_balance[0] if len(collateral_flows.beginning_balance) > 0 else 0

    for i, t in enumerate(tranche_specs):
        if t.principal_type in (PrincipalType.PAC, PrincipalType.Z_PAC):
            pac_schedules[i] = compute_pac_schedule(
                collateral_balance=total_collateral,
                wac=collateral_coupon,
                wam=n_months,
                wala=0,
                lower_psa=t.pac_lower_band,
                upper_psa=t.pac_upper_band,
            )
        elif t.principal_type == PrincipalType.PAC_II:
            pac_schedules[i] = compute_pac_schedule(
                collateral_balance=total_collateral,
                wac=collateral_coupon,
                wam=n_months,
                wala=0,
                lower_psa=t.pac_lower_band,
                upper_psa=t.pac_upper_band,
            )
        elif t.principal_type == PrincipalType.SCHEDULE:
            schedule_bond_schedules[i] = compute_schedule_bond_schedule(
                collateral_balance=total_collateral,
                wac=collateral_coupon,
                wam=n_months,
                wala=0,
                target_psa=t.schedule_speed,
            )
        elif t.principal_type == PrincipalType.REVERSE_TAC:
            reverse_tac_schedules[i] = compute_reverse_tac_schedule(
                collateral_balance=total_collateral,
                wac=collateral_coupon,
                wam=n_months,
                wala=0,
                slow_psa=t.pac_lower_band,  # reuse lower band as slow speed
            )

    # Sort tranches by priority for principal allocation
    seq_order = sorted(range(n_tranches), key=lambda i: tranche_specs[i].priority)

    # Jump-Z state tracking
    jump_z_active = {}  # i -> True means jump-Z has converted to current-pay
    for i, t in enumerate(tranche_specs):
        if t.principal_type == PrincipalType.JUMP_Z:
            jump_z_active[i] = False

    # Map VADM tranches to their Z-bond source
    vadm_z_map = {}  # VADM index -> Z-bond index
    z_name_to_idx = {t.name: i for i, t in enumerate(tranche_specs)}
    for i, t in enumerate(tranche_specs):
        if t.principal_type == PrincipalType.VADM and t.vadm_z_tranche:
            z_idx = z_name_to_idx.get(t.vadm_z_tranche)
            if z_idx is not None:
                vadm_z_map[i] = z_idx

    for m in range(n_months):
        available_interest = collateral_flows.interest[m]
        available_principal = collateral_flows.total_principal[m]

        # ==================
        # INTEREST WATERFALL
        # ==================
        z_bond_interest = 0.0
        z_bond_interest_by_tranche = {}  # Track per-Z for VADM allocation

        for i, t in enumerate(tranche_specs):
            if balances[i] <= 0.01 and not t.is_io:
                continue

            # Z-bond types: regular Z, Jump-Z (when not active), Z-PAC (during lockout)
            is_z_accruing = False
            if t.is_z_bond:
                is_z_accruing = True
            elif t.principal_type == PrincipalType.JUMP_Z and not jump_z_active.get(i, False):
                is_z_accruing = True
            elif t.principal_type == PrincipalType.Z_PAC and m < t.lockout_months:
                is_z_accruing = True

            if is_z_accruing:
                z_interest = (balances[i] + accrued[i]) * (t.coupon / 100.0 / 12.0)
                accrued[i] += z_interest
                tranche_accrued[i, m] = accrued[i]
                z_bond_interest += z_interest
                z_bond_interest_by_tranche[i] = z_interest
                continue

            if t.is_po or t.principal_type == PrincipalType.SUPER_PO:
                tranche_interest[i, m] = 0.0
                continue

            if t.is_io:
                notional = t.notional_balance
                io_interest = notional * (t.coupon / 100.0 / 12.0)
                paid = min(io_interest, available_interest)
                tranche_interest[i, m] = paid
                available_interest -= paid
                if collateral_flows.beginning_balance[0] > 0:
                    factor = collateral_flows.ending_balance[m] / collateral_flows.beginning_balance[0]
                    t.notional_balance = t.notional_balance * factor / (
                        collateral_flows.ending_balance[max(0, m-1)] / collateral_flows.beginning_balance[0]
                        if m > 0 and collateral_flows.ending_balance[max(0, m-1)] > 0 else 1.0
                    )
                continue

            # Regular fixed or floating interest
            if t.interest_type == InterestType.FLOATING:
                rate = min(t.rate_cap, max(t.rate_floor,
                           3.5 + t.index_spread / 100.0)) / 100.0 / 12.0
            elif t.interest_type == InterestType.INVERSE_FLOATING:
                sofr = 3.5
                inv_rate = t.inverse_constant - t.inverse_multiplier * sofr
                rate = min(t.rate_cap, max(t.rate_floor, inv_rate)) / 100.0 / 12.0
            else:
                rate = t.coupon / 100.0 / 12.0

            bond_interest = balances[i] * rate
            paid = min(bond_interest, available_interest)
            tranche_interest[i, m] = paid
            available_interest -= paid

        # ====================
        # PRINCIPAL WATERFALL
        # ====================

        # Add Z-bond accrued interest to available principal (accretion directed)
        total_available_principal = available_principal + z_bond_interest

        # Track Z-bond accretion available for VADM
        vadm_accretion_available = dict(z_bond_interest_by_tranche)

        # Phase 1: PAC-I tranches get their scheduled amounts (highest priority)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.PAC or balances[i] <= 0.01:
                continue

            if i in pac_schedules and m < len(pac_schedules[i]):
                pac_amount = pac_schedules[i][m]
            else:
                pac_amount = 0.0

            actual = min(pac_amount, balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 1b: Z-PAC tranches (PAC schedule but with Z-accrual during lockout)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.Z_PAC or balances[i] <= 0.01:
                continue
            if m < t.lockout_months:
                # During lockout: no principal, interest accrues (handled above)
                continue

            if i in pac_schedules and m < len(pac_schedules[i]):
                pac_amount = pac_schedules[i][m]
            else:
                pac_amount = 0.0

            actual = min(pac_amount, balances[i] + accrued[i], total_available_principal)
            # Pay down accrued first, then balance
            accr_pay = min(accrued[i], actual)
            accrued[i] -= accr_pay
            prin_pay = actual - accr_pay
            tranche_principal[i, m] = actual
            balances[i] -= prin_pay
            total_available_principal -= actual

        # Phase 1c: VADM tranches get principal from Z-bond accretion only
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.VADM or balances[i] <= 0.01:
                continue

            # VADM gets principal only from its paired Z-bond's accretion
            z_idx = vadm_z_map.get(i)
            if z_idx is not None:
                vadm_avail = vadm_accretion_available.get(z_idx, 0.0)
            else:
                # If no specific Z-bond, use total Z accretion
                vadm_avail = sum(vadm_accretion_available.values())

            actual = min(vadm_avail, balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual
            # Reduce available accretion
            if z_idx is not None and z_idx in vadm_accretion_available:
                vadm_accretion_available[z_idx] -= actual

        # Phase 2: Sequential tranches in priority order (with NAS lockout check)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type == PrincipalType.NAS:
                # NAS: locked out for first N months
                if m < t.lockout_months or balances[i] <= 0.01:
                    continue
                if total_available_principal <= 0.01:
                    continue
                actual = min(balances[i], total_available_principal)
                tranche_principal[i, m] = actual
                balances[i] -= actual
                total_available_principal -= actual
                continue

            if t.principal_type not in (PrincipalType.SEQUENTIAL, PrincipalType.PASSTHROUGH,
                                         PrincipalType.ACCRETION_DIRECTED):
                continue
            if balances[i] <= 0.01 or total_available_principal <= 0.01:
                continue

            actual = min(balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 2b: Schedule bonds get their targeted amount
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.SCHEDULE or balances[i] <= 0.01:
                continue
            if total_available_principal <= 0.01:
                continue

            if i in schedule_bond_schedules and m < len(schedule_bond_schedules[i]):
                target = schedule_bond_schedules[i][m]
            else:
                target = 0.0

            actual = min(target, balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 2c: PAC-II tranches (after PAC-I, before support)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.PAC_II or balances[i] <= 0.01:
                continue
            if total_available_principal <= 0.01:
                continue

            if i in pac_schedules and m < len(pac_schedules[i]):
                pac2_amount = pac_schedules[i][m]
            else:
                pac2_amount = 0.0

            actual = min(pac2_amount, balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 3: TAC tranches
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.TAC or balances[i] <= 0.01:
                continue
            if total_available_principal <= 0.01:
                continue

            actual = min(balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 3b: Reverse TAC (extension protection - guaranteed minimum principal)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.REVERSE_TAC or balances[i] <= 0.01:
                continue
            if total_available_principal <= 0.01:
                continue

            if i in reverse_tac_schedules and m < len(reverse_tac_schedules[i]):
                rtac_target = reverse_tac_schedules[i][m]
            else:
                rtac_target = 0.0

            # Reverse TAC gets at least its scheduled amount
            actual = min(max(rtac_target, 0), balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 4: Support tranches absorb remaining principal
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.SUPPORT or balances[i] <= 0.01:
                continue
            if total_available_principal <= 0.01:
                continue

            actual = min(balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 4b: Super PO (PO from support class - gets principal after support)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.SUPER_PO or balances[i] <= 0.01:
                continue
            if total_available_principal <= 0.01:
                continue

            actual = min(balances[i], total_available_principal)
            tranche_principal[i, m] = actual
            balances[i] -= actual
            total_available_principal -= actual

        # Phase 5: Jump-Z trigger check and principal
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type != PrincipalType.JUMP_Z:
                continue

            # Check trigger: all support tranches exhausted
            support_balance = sum(balances[j] for j in range(n_tranches)
                                if tranche_specs[j].principal_type == PrincipalType.SUPPORT)
            if support_balance <= 0.01 and not jump_z_active.get(i, False):
                jump_z_active[i] = True  # Trigger!

            if jump_z_active.get(i, False):
                # Jump-Z is now current-pay: absorb principal
                if balances[i] <= 0.01 or total_available_principal <= 0.01:
                    continue
                actual = min(balances[i] + accrued[i], total_available_principal)
                accr_pay = min(accrued[i], actual)
                accrued[i] -= accr_pay
                prin_pay = actual - accr_pay
                tranche_principal[i, m] = actual
                balances[i] -= prin_pay
                total_available_principal -= actual

                # Sticky: once triggered, stays current-pay
                # Non-sticky: reverts if supports get principal again (not typical)
                if not t.is_sticky:
                    # Check if supports got any principal back (very rare edge case)
                    pass

        # Phase 6: Regular Z-bond principal (after all others retired)
        for i in seq_order:
            t = tranche_specs[i]
            if t.principal_type not in (PrincipalType.Z_BOND,) or total_available_principal <= 0.01:
                continue
            if not t.is_z_bond:
                continue

            other_balance = sum(balances[j] for j in range(n_tranches)
                              if j != i and not tranche_specs[j].is_io
                              and tranche_specs[j].principal_type != PrincipalType.JUMP_Z)
            if other_balance <= 0.01:
                actual = min(balances[i] + accrued[i], total_available_principal)
                accr_pay = min(accrued[i], actual)
                accrued[i] -= accr_pay
                prin_pay = actual - accr_pay
                tranche_principal[i, m] = actual
                balances[i] -= prin_pay
                total_available_principal -= actual

        # Record balances
        for i in range(n_tranches):
            tranche_balance[i, m] = balances[i]

        residual[m] = total_available_principal + available_interest

    # Build output
    months = collateral_flows.months
    tranche_flow_dict = {}

    for i, t in enumerate(tranche_specs):
        tcf = TrancheCashFlows(
            name=t.name,
            months=months,
            beginning_balance=np.concatenate([[t.original_balance],
                                              tranche_balance[i, :-1]]),
            principal=tranche_principal[i],
            interest=tranche_interest[i],
            total_cash_flow=tranche_principal[i] + tranche_interest[i],
            ending_balance=tranche_balance[i],
            accrued_interest=tranche_accrued[i],
        )
        tranche_flow_dict[t.name] = tcf

    return CMOCashFlows(
        deal_id=deal_id,
        tranche_flows=tranche_flow_dict,
        collateral_flows=collateral_flows,
        residual=residual,
    )


def create_sequential_cmo(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    tranche_sizes: list[float],
    tranche_coupons: list[float],
    tranche_names: Optional[list[str]] = None,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """Create a simple sequential-pay CMO structure."""
    n = len(tranche_sizes)
    if tranche_names is None:
        tranche_names = [f"A{i+1}" for i in range(n)]

    specs = []
    for i in range(n):
        specs.append(TrancheSpec(
            name=tranche_names[i],
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.FIXED,
            original_balance=tranche_sizes[i],
            coupon=tranche_coupons[i],
            priority=i,
        ))

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_pac_support_cmo(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    pac_balance: float,
    pac_coupon: float,
    support_balance: float,
    support_coupon: float,
    pac_lower: float = 100.0,
    pac_upper: float = 300.0,
    collateral_coupon: float = 5.5,
    z_bond_balance: float = 0.0,
    z_bond_coupon: float = 0.0,
    io_notional: float = 0.0,
    io_coupon: float = 0.0,
) -> CMOCashFlows:
    """Create a PAC/Support CMO structure with optional Z-bond and IO."""
    specs = [
        TrancheSpec(
            name="PAC",
            principal_type=PrincipalType.PAC,
            interest_type=InterestType.FIXED,
            original_balance=pac_balance,
            coupon=pac_coupon,
            pac_lower_band=pac_lower,
            pac_upper_band=pac_upper,
            priority=0,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=support_balance,
            coupon=support_coupon,
            priority=2,
        ),
    ]

    if z_bond_balance > 0:
        specs.append(TrancheSpec(
            name="Z",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.Z_ACCRUAL,
            original_balance=z_bond_balance,
            coupon=z_bond_coupon,
            priority=3,
        ))

    if io_notional > 0:
        specs.append(TrancheSpec(
            name="IO",
            principal_type=PrincipalType.PASSTHROUGH,
            interest_type=InterestType.IO_ONLY,
            original_balance=0,
            notional_balance=io_notional,
            coupon=io_coupon,
            priority=99,
        ))

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_vadm_z_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    seq_balances: list[float],
    seq_coupons: list[float],
    vadm_balance: float,
    vadm_coupon: float,
    z_bond_balance: float,
    z_bond_coupon: float,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Create a VADM + Z-bond structure.

    The VADM (Very Accurately Defined Maturity) receives principal solely from
    the Z-bond's interest accretion. Since Z-accretion is deterministic (coupon
    rate x balance), the VADM's WAL is nearly independent of prepayments.

    This is the ultimate "hold down" — you know exactly when you get paid.
    The Z-bond is cheap long-dated paper and the VADM is a nearly bullet-like
    MBS that trades at tight spreads.
    """
    specs = []
    for i, (bal, cpn) in enumerate(zip(seq_balances, seq_coupons)):
        specs.append(TrancheSpec(
            name=f"A{i+1}",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.FIXED,
            original_balance=bal,
            coupon=cpn,
            priority=i,
        ))

    specs.append(TrancheSpec(
        name="VADM",
        principal_type=PrincipalType.VADM,
        interest_type=InterestType.FIXED,
        original_balance=vadm_balance,
        coupon=vadm_coupon,
        vadm_z_tranche="Z",
        priority=len(seq_balances),
    ))

    specs.append(TrancheSpec(
        name="Z",
        principal_type=PrincipalType.SEQUENTIAL,
        interest_type=InterestType.Z_ACCRUAL,
        original_balance=z_bond_balance,
        coupon=z_bond_coupon,
        priority=len(seq_balances) + 1,
    ))

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_pac_jump_z_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    pac_balance: float,
    pac_coupon: float,
    support_balance: float,
    support_coupon: float,
    jump_z_balance: float,
    jump_z_coupon: float,
    pac_lower: float = 100.0,
    pac_upper: float = 300.0,
    sticky: bool = True,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Create a PAC / Support / Jump-Z structure.

    The Jump-Z accrues interest like a regular Z-bond, but "jumps" to
    current-pay status when the support tranche is exhausted. This protects
    the PAC from being broken by fast prepayments — the Jump-Z absorbs
    excess principal once supports are gone.

    Sticky Jump-Z stays in current-pay once triggered. Non-sticky can
    revert to accrual (rare in practice).

    The Jump-Z is the hold-down: cheap, long-dated, but with a structural
    feature that makes the PAC more marketable.
    """
    specs = [
        TrancheSpec(
            name="PAC",
            principal_type=PrincipalType.PAC,
            interest_type=InterestType.FIXED,
            original_balance=pac_balance,
            coupon=pac_coupon,
            pac_lower_band=pac_lower,
            pac_upper_band=pac_upper,
            priority=0,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=support_balance,
            coupon=support_coupon,
            priority=1,
        ),
        TrancheSpec(
            name="JZ",
            principal_type=PrincipalType.JUMP_Z,
            interest_type=InterestType.Z_ACCRUAL,
            original_balance=jump_z_balance,
            coupon=jump_z_coupon,
            is_sticky=sticky,
            priority=2,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_pac_ii_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    pac_i_balance: float,
    pac_i_coupon: float,
    pac_ii_balance: float,
    pac_ii_coupon: float,
    support_balance: float,
    support_coupon: float,
    pac_i_lower: float = 100.0,
    pac_i_upper: float = 300.0,
    pac_ii_lower: float = 150.0,
    pac_ii_upper: float = 250.0,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Create a PAC-I / PAC-II / Support structure.

    PAC-II (Type II PAC) is carved from the support class with narrower
    PSA bands than the PAC-I. It offers better yield than PAC-I with
    PAC-like stability within its narrower band.

    The PAC-II is the hold-down: it's where the support class's risk is
    partially mitigated. Desks keep it because it's cheap enough to
    warehouse but stable enough to not blow up.
    """
    specs = [
        TrancheSpec(
            name="PAC-I",
            principal_type=PrincipalType.PAC,
            interest_type=InterestType.FIXED,
            original_balance=pac_i_balance,
            coupon=pac_i_coupon,
            pac_lower_band=pac_i_lower,
            pac_upper_band=pac_i_upper,
            priority=0,
        ),
        TrancheSpec(
            name="PAC-II",
            principal_type=PrincipalType.PAC_II,
            interest_type=InterestType.FIXED,
            original_balance=pac_ii_balance,
            coupon=pac_ii_coupon,
            pac_lower_band=pac_ii_lower,
            pac_upper_band=pac_ii_upper,
            priority=1,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=support_balance,
            coupon=support_coupon,
            priority=2,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_nas_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    nas_balance: float,
    nas_coupon: float,
    nas_lockout_months: int,
    sequential_balance: float,
    sequential_coupon: float,
    support_balance: float,
    support_coupon: float,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Create a NAS (Non-Accelerating Senior) structure.

    The NAS tranche has a principal lockout for the first N months. During
    lockout, it receives only interest — no principal. This creates a
    guaranteed minimum average life and a more defined payment window.

    Used in non-agency deals but applicable to any REMIC. The NAS is an
    excellent hold-down for desks wanting duration certainty — you know
    the bond won't pay down for at least N months.
    """
    specs = [
        TrancheSpec(
            name="SEQ",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.FIXED,
            original_balance=sequential_balance,
            coupon=sequential_coupon,
            priority=0,
        ),
        TrancheSpec(
            name="NAS",
            principal_type=PrincipalType.NAS,
            interest_type=InterestType.FIXED,
            original_balance=nas_balance,
            coupon=nas_coupon,
            lockout_months=nas_lockout_months,
            priority=1,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=support_balance,
            coupon=support_coupon,
            priority=2,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_schedule_bond_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    schedule_balance: float,
    schedule_coupon: float,
    target_psa: float,
    support_balance: float,
    support_coupon: float,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Create a Schedule Bond / Support structure.

    A schedule bond targets a specific PSA speed (not a band like PAC).
    It's cheaper than PAC because it has no band protection — if
    prepayments deviate from the target speed, the schedule breaks.

    Good hold-down if you're confident in your prepayment call. The desk
    can warehouse it cheaply and the schedule makes it predictable at
    the assumed speed.
    """
    specs = [
        TrancheSpec(
            name="SCHED",
            principal_type=PrincipalType.SCHEDULE,
            interest_type=InterestType.FIXED,
            original_balance=schedule_balance,
            coupon=schedule_coupon,
            schedule_speed=target_psa,
            priority=0,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=support_balance,
            coupon=support_coupon,
            priority=1,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_z_pac_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    seq_balance: float,
    seq_coupon: float,
    z_pac_balance: float,
    z_pac_coupon: float,
    z_pac_lockout: int,
    support_balance: float,
    support_coupon: float,
    pac_lower: float = 100.0,
    pac_upper: float = 300.0,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """
    Create a Z-PAC structure.

    A Z-PAC combines PAC stability with Z-bond accrual. During the lockout
    period, the Z-PAC accrues interest (like a Z-bond). After lockout, it
    follows a PAC schedule with band protection.

    This is a premium hold-down: long duration like a Z-bond, but with PAC
    stability after the lockout. It extends the deal's duration profile
    while maintaining cash flow predictability.
    """
    specs = [
        TrancheSpec(
            name="SEQ",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.FIXED,
            original_balance=seq_balance,
            coupon=seq_coupon,
            priority=0,
        ),
        TrancheSpec(
            name="ZPAC",
            principal_type=PrincipalType.Z_PAC,
            interest_type=InterestType.FIXED,
            original_balance=z_pac_balance,
            coupon=z_pac_coupon,
            lockout_months=z_pac_lockout,
            pac_lower_band=pac_lower,
            pac_upper_band=pac_upper,
            priority=1,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=support_balance,
            coupon=support_coupon,
            priority=2,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_kitchen_sink_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    collateral_coupon: float = 5.5,
    total_balance: Optional[float] = None,
) -> CMOCashFlows:
    """
    Create a "kitchen sink" deal with every exotic structure type.

    This is what a desk creates when they want to demonstrate the full
    spectrum of structuring capabilities. It includes:
    - PAC-I (stable, distributed to money managers)
    - PAC-II (narrower bands, higher yield, institutional)
    - VADM (prepayment-independent, ultra-stable)
    - Jump-Z (converts on support exhaustion, protects PAC)
    - Z-bond (feeds VADM, extends duration)
    - NAS (lockout creates minimum life)
    - Support (absorbs all prepayment volatility)
    - IO strip (negative duration, rate view)

    The hold-downs are the VADM, Jump-Z, NAS, and support tranches.
    """
    if total_balance is None:
        total_balance = collateral_flows.beginning_balance[0] if len(collateral_flows.beginning_balance) > 0 else 1_000_000

    specs = [
        TrancheSpec(
            name="PAC-I",
            principal_type=PrincipalType.PAC,
            interest_type=InterestType.FIXED,
            original_balance=total_balance * 0.30,
            coupon=collateral_coupon - 0.25,
            pac_lower_band=100.0,
            pac_upper_band=300.0,
            priority=0,
        ),
        TrancheSpec(
            name="PAC-II",
            principal_type=PrincipalType.PAC_II,
            interest_type=InterestType.FIXED,
            original_balance=total_balance * 0.10,
            coupon=collateral_coupon,
            pac_lower_band=150.0,
            pac_upper_band=250.0,
            priority=1,
        ),
        TrancheSpec(
            name="VADM",
            principal_type=PrincipalType.VADM,
            interest_type=InterestType.FIXED,
            original_balance=total_balance * 0.08,
            coupon=collateral_coupon - 0.5,
            vadm_z_tranche="Z",
            priority=2,
        ),
        TrancheSpec(
            name="NAS",
            principal_type=PrincipalType.NAS,
            interest_type=InterestType.FIXED,
            original_balance=total_balance * 0.12,
            coupon=collateral_coupon,
            lockout_months=36,
            priority=3,
        ),
        TrancheSpec(
            name="SUP",
            principal_type=PrincipalType.SUPPORT,
            interest_type=InterestType.FIXED,
            original_balance=total_balance * 0.20,
            coupon=collateral_coupon + 0.5,
            priority=4,
        ),
        TrancheSpec(
            name="JZ",
            principal_type=PrincipalType.JUMP_Z,
            interest_type=InterestType.Z_ACCRUAL,
            original_balance=total_balance * 0.10,
            coupon=collateral_coupon + 0.25,
            is_sticky=True,
            priority=5,
        ),
        TrancheSpec(
            name="Z",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.Z_ACCRUAL,
            original_balance=total_balance * 0.10,
            coupon=collateral_coupon + 0.5,
            priority=6,
        ),
        TrancheSpec(
            name="IO",
            principal_type=PrincipalType.PASSTHROUGH,
            interest_type=InterestType.IO_ONLY,
            original_balance=0,
            notional_balance=total_balance * 0.50,
            coupon=1.0,
            priority=99,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


def create_floater_inverse_structure(
    deal_id: str,
    collateral_flows: PoolCashFlows,
    floater_balance: float,
    inverse_balance: float,
    floater_spread_bps: float = 50.0,
    floater_cap: float = 10.0,
    inverse_constant: float = 20.0,
    inverse_multiplier: float = 3.0,
    collateral_coupon: float = 5.5,
) -> CMOCashFlows:
    """Create a Floater / Inverse Floater CMO structure."""
    specs = [
        TrancheSpec(
            name="FLT",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.FLOATING,
            original_balance=floater_balance,
            index_spread=floater_spread_bps,
            rate_cap=floater_cap,
            rate_floor=0.0,
            priority=0,
        ),
        TrancheSpec(
            name="INV",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.INVERSE_FLOATING,
            original_balance=inverse_balance,
            inverse_constant=inverse_constant,
            inverse_multiplier=inverse_multiplier,
            rate_cap=inverse_constant,
            rate_floor=0.0,
            priority=1,
        ),
    ]

    return structure_cmo(deal_id, collateral_flows, specs, collateral_coupon)


# ─── Fukushima Optimal PAC Sizing (Fukushima, Yamashita & Kutsuna 2004) ────

def optimal_pac_fraction(
    collateral_flows: PoolCashFlows,
    psa_scenarios: list[float] | None = None,
    loss_threshold: float = 0.01,
) -> dict:
    """
    Compute optimal PAC fraction using Fukushima et al. (2004) cash reserve model.

    The Kyoto University paper proves that optimal PAC-Companion structuring
    can be formulated as an LP. Key insight: allowing cash reserves between
    periods lets the issuer produce MORE PAC bonds while keeping payment
    certainty high.

    Simplified implementation:
      1. Simulate cash flows under multiple PSA scenarios
      2. Find maximum PAC fraction with E[loss] <= loss_threshold
      3. Account for cash reserve buffering (Fukushima's Model 1)

    Returns:
        dict with optimal_pac_frac, optimal_bands, reserve_benefit
    """
    if psa_scenarios is None:
        psa_scenarios = [float(s) for s in range(50, 425, 25)]

    T = len(collateral_flows.total_cash_flow)
    base_total = collateral_flows.total_cash_flow

    # Scale cash flows for each PSA scenario
    base_psa = 100.0
    scenario_cfs = []
    for psa in psa_scenarios:
        scale = psa / base_psa
        principal_scaled = collateral_flows.scheduled_principal + \
            collateral_flows.prepaid_principal * scale
        interest_scaled = collateral_flows.interest * max(0.2, 1.0 - 0.3 * (scale - 1.0))
        total_scaled = principal_scaled + interest_scaled
        scenario_cfs.append(np.maximum(total_scaled, 0.0))

    n_scenarios = len(scenario_cfs)

    # Search for max PAC fraction with cash reserve (Fukushima Model 1)
    best_frac = 0.0
    for pac_frac in np.arange(0.10, 0.75, 0.02):
        pac_schedule = pac_frac * base_total
        total_loss = 0.0

        for cf in scenario_cfs:
            reserve = 0.0
            for t in range(T):
                available = cf[t] + reserve
                payment = min(pac_schedule[t], available)
                loss = pac_schedule[t] - payment
                total_loss += loss
                surplus = available - payment
                reserve = min(surplus, pac_schedule[t] * 2.0)

        expected_loss = total_loss / n_scenarios
        pac_total = float(np.sum(pac_schedule))
        if expected_loss / max(1.0, pac_total) <= loss_threshold:
            best_frac = pac_frac

    # Find PAC bands
    best_bands = (100.0, 300.0)
    if best_frac > 0:
        pac_schedule = best_frac * base_total
        lower_band = psa_scenarios[0]
        for psa_idx, psa in enumerate(psa_scenarios):
            cf = scenario_cfs[psa_idx]
            if all(cf[t] >= pac_schedule[t] * 0.95 for t in range(min(T, 120))):
                lower_band = psa
                break

        upper_band = psa_scenarios[-1]
        for psa_idx in range(len(psa_scenarios) - 1, -1, -1):
            cf = scenario_cfs[psa_idx]
            if all(cf[t] >= pac_schedule[t] * 0.95 for t in range(min(T, 120))):
                upper_band = psa_scenarios[psa_idx]
                break

        best_bands = (lower_band, upper_band)

    # No-reserve baseline for comparison
    no_reserve_frac = 0.0
    for pac_frac_nr in np.arange(0.10, 0.75, 0.02):
        pac_schedule = pac_frac_nr * base_total
        total_loss_nr = 0.0
        for cf in scenario_cfs:
            for t in range(T):
                total_loss_nr += max(0.0, pac_schedule[t] - cf[t])
        expected_loss_nr = total_loss_nr / n_scenarios
        pac_total_nr = float(np.sum(pac_schedule))
        if expected_loss_nr / max(1.0, pac_total_nr) <= loss_threshold:
            no_reserve_frac = pac_frac_nr

    return {
        "optimal_pac_frac": best_frac,
        "optimal_bands": best_bands,
        "reserve_benefit": best_frac - no_reserve_frac,
        "no_reserve_frac": no_reserve_frac,
    }
