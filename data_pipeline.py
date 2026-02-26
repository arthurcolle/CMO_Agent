"""
Data Processing Pipeline for CMO Agent Training.

Processes raw data into training-ready formats:
1. CAS loan-level data → prepayment/credit curves by coupon/FICO/LTV/vintage
2. REMIC issuance data → expert action sequences for imitation learning
3. Combined calibration → updated model parameters
4. Expert demonstrations → action sequences for RL environment

Data sources:
- CAS (Credit Access Summary): 113-field pipe-delimited loan data
- REMIC issuance: Monthly Ginnie Mae deal structure summaries
- CIRT: Credit Insurance Risk Transfer loan performance
"""
import csv
import json
import os
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .remic_loader import load_all_remic_data, REMICDeal
from .cmo_structure import PrincipalType, InterestType
from .yield_book_env import ActionType


# ═══════════════════════════════════════════════════════════════════════════
# CAS Loan-Level Data Processing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CASLoan:
    """Parsed CAS loan record (key fields only)."""
    loan_id: str
    orig_rate: float
    current_rate: float
    orig_upb: float
    current_upb: float
    orig_term: int
    loan_age: int
    remaining_months: int
    orig_ltv: float
    orig_cltv: float
    dti: float
    fico: int
    co_fico: int
    first_time_buyer: str
    purpose: str        # P=Purchase, C=Cash-out, N=No cash-out refi
    property_type: str  # SF, CO, CP, MH, PU
    occupancy: str      # P=Primary, S=Second, I=Investment
    state: str
    channel: str        # R=Retail, C=Correspondent, B=Broker
    delinquency: str
    zero_bal_code: str
    mi_pct: float
    deal_name: str


@dataclass
class PrepaymentCurve:
    """Calibrated prepayment curve from loan data."""
    coupon_bucket: str
    fico_bucket: str
    ltv_bucket: str
    loan_count: int
    monthly_smm: np.ndarray       # 360-month SMM vector
    monthly_cpr: np.ndarray       # annualized CPR
    monthly_cdr: np.ndarray       # conditional default rate
    active_counts: np.ndarray     # loans active at each month
    prepaid_counts: np.ndarray
    defaulted_counts: np.ndarray


@dataclass
class CASProcessingResult:
    """Result of processing CAS loan data."""
    total_loans: int = 0
    by_coupon: dict = field(default_factory=dict)
    by_fico: dict = field(default_factory=dict)
    by_ltv: dict = field(default_factory=dict)
    by_purpose: dict = field(default_factory=dict)
    by_state: dict = field(default_factory=dict)
    by_vintage: dict = field(default_factory=dict)
    prepayment_curves: list = field(default_factory=list)
    avg_fico: float = 0.0
    avg_ltv: float = 0.0
    avg_dti: float = 0.0
    avg_rate: float = 0.0
    prepaid_pct: float = 0.0
    defaulted_pct: float = 0.0
    current_pct: float = 0.0


def parse_cas_header(header_path: str) -> list[str]:
    """Parse CAS header file to get column names."""
    with open(header_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        return [h.strip() for h in header]


def _compute_loan_age(orig_date_str: str, report_period_str: str) -> int:
    """Compute loan age in months from origination date and report period.
    Dates in format MMYYYY."""
    try:
        if not orig_date_str or not report_period_str:
            return 0
        orig_m = int(orig_date_str[:2])
        orig_y = int(orig_date_str[2:])
        rpt_m = int(report_period_str[:2])
        rpt_y = int(report_period_str[2:])
        return (rpt_y - orig_y) * 12 + (rpt_m - orig_m)
    except (ValueError, IndexError):
        return 0


def parse_cas_loans(filepath: str, max_rows: int = 0) -> list[CASLoan]:
    """Parse CAS investor file (pipe-delimited).

    CAS column layout (113 fields):
    [1] Loan ID, [2] Report Period, [3] Channel, [7] Orig Rate, [8] Current Rate,
    [9] Orig UPB, [11] Current UPB, [12] Orig Term, [13] Orig Date,
    [15] Loan Age, [19] Orig LTV, [20] Orig CLTV, [22] DTI,
    [23] FICO, [24] Co-FICO, [25] First-time buyer, [26] Purpose,
    [27] Property Type, [29] Occupancy, [30] State,
    [39] Delinquency, [43] Zero Balance Code (NOT 42!),
    [33] MI Pct, [97] Deal Name

    Args:
        filepath: Path to CAS investor CSV (pipe-delimited)
        max_rows: Max rows to read (0 = all)
    """
    loans = []
    count = 0

    # Handle both raw files and zip archives
    import zipfile
    if filepath.endswith('.zip'):
        z = zipfile.ZipFile(filepath)
        names = z.namelist()
        csv_name = [n for n in names if n.endswith('.csv')][0]
        f = z.open(csv_name)
        lines = (line.decode('utf-8') for line in f)
    else:
        f = open(filepath, 'r')
        lines = f

    for line in lines:
        fields = line.strip().split('|')
        if len(fields) < 44:
            continue
        try:
            # Compute age from origination date if loan_age field is empty
            age = _safe_int(fields[15])
            if age == 0 and len(fields) > 13:
                age = _compute_loan_age(fields[13].strip(), fields[2].strip())

            loan = CASLoan(
                loan_id=fields[1],
                orig_rate=_safe_float(fields[7]),
                current_rate=_safe_float(fields[8]),
                orig_upb=_safe_float(fields[9]),
                current_upb=_safe_float(fields[11]),
                orig_term=_safe_int(fields[12]),
                loan_age=age,
                remaining_months=_safe_int(fields[16]),
                orig_ltv=_safe_float(fields[19]),
                orig_cltv=_safe_float(fields[20]),
                dti=_safe_float(fields[22]),
                fico=_safe_int(fields[23]),
                co_fico=_safe_int(fields[24]),
                first_time_buyer=fields[25].strip(),
                purpose=fields[26].strip(),
                property_type=fields[27].strip(),
                occupancy=fields[29].strip(),
                state=fields[30].strip(),
                channel=fields[3].strip(),
                delinquency=fields[39].strip(),
                zero_bal_code=fields[43].strip(),  # Column 43, NOT 42
                mi_pct=_safe_float(fields[33]) if len(fields) > 33 else 0.0,
                deal_name=fields[97].strip() if len(fields) > 97 else "",
            )
            loans.append(loan)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break
        except (IndexError, ValueError):
            continue

    f.close()
    return loans


def process_cas_data(filepath: str, max_rows: int = 0) -> CASProcessingResult:
    """Process CAS loan file into aggregated statistics.

    Builds prepayment/default curves bucketed by:
    - Coupon: <4%, 4-5%, 5-6%, 6-7%, >7%
    - FICO: <660, 660-700, 700-740, 740-780, >780
    - LTV: <70, 70-80, 80-90, 90-95, 95-100, >100
    - Purpose: Purchase, Refi, Cash-out
    """
    print(f"Loading CAS data from {filepath}...")
    t0 = time.time()
    loans = parse_cas_loans(filepath, max_rows)
    print(f"  Loaded {len(loans):,} loans in {time.time()-t0:.1f}s")

    result = CASProcessingResult(total_loans=len(loans))
    if not loans:
        return result

    # Aggregate stats
    ficos = [l.fico for l in loans if l.fico > 0]
    ltvs = [l.orig_ltv for l in loans if l.orig_ltv > 0]
    dtis = [l.dti for l in loans if l.dti > 0]
    rates = [l.orig_rate for l in loans if l.orig_rate > 0]

    result.avg_fico = float(np.mean(ficos)) if ficos else 0
    result.avg_ltv = float(np.mean(ltvs)) if ltvs else 0
    result.avg_dti = float(np.mean(dtis)) if dtis else 0
    result.avg_rate = float(np.mean(rates)) if rates else 0

    prepaid = sum(1 for l in loans if l.zero_bal_code in ("01", "1"))
    defaulted = sum(1 for l in loans if l.zero_bal_code in ("03", "09", "15", "3", "9"))
    result.prepaid_pct = prepaid / len(loans) * 100
    result.defaulted_pct = defaulted / len(loans) * 100
    result.current_pct = 100 - result.prepaid_pct - result.defaulted_pct

    # Bucket by coupon
    coupon_buckets = {"<4%": (0, 4), "4-5%": (4, 5), "5-6%": (5, 6), "6-7%": (6, 7), ">7%": (7, 20)}
    for name, (lo, hi) in coupon_buckets.items():
        bucket = [l for l in loans if lo <= l.orig_rate < hi]
        if bucket:
            result.by_coupon[name] = _bucket_stats(bucket)

    # Bucket by FICO
    fico_buckets = {"<660": (0, 660), "660-700": (660, 700), "700-740": (700, 740),
                    "740-780": (740, 780), ">780": (780, 900)}
    for name, (lo, hi) in fico_buckets.items():
        bucket = [l for l in loans if lo <= l.fico < hi]
        if bucket:
            result.by_fico[name] = _bucket_stats(bucket)

    # Bucket by LTV
    ltv_buckets = {"<70": (0, 70), "70-80": (70, 80), "80-90": (80, 90),
                   "90-95": (90, 95), "95-100": (95, 101)}
    for name, (lo, hi) in ltv_buckets.items():
        bucket = [l for l in loans if lo <= l.orig_ltv < hi]
        if bucket:
            result.by_ltv[name] = _bucket_stats(bucket)

    # Bucket by purpose
    purpose_map = {"P": "Purchase", "C": "Cash-out", "N": "No-cash-refi", "R": "Refi"}
    for code, name in purpose_map.items():
        bucket = [l for l in loans if l.purpose == code]
        if bucket:
            result.by_purpose[name] = _bucket_stats(bucket)

    # Bucket by state (top 10)
    state_counts = defaultdict(list)
    for l in loans:
        if l.state:
            state_counts[l.state].append(l)
    top_states = sorted(state_counts.items(), key=lambda x: -len(x[1]))[:10]
    for state, bucket in top_states:
        result.by_state[state] = _bucket_stats(bucket)

    # Build prepayment curves by coupon × FICO
    print("  Building prepayment curves...")
    for coupon_name, (c_lo, c_hi) in coupon_buckets.items():
        for fico_name, (f_lo, f_hi) in fico_buckets.items():
            bucket = [l for l in loans if c_lo <= l.orig_rate < c_hi and f_lo <= l.fico < f_hi]
            if len(bucket) >= 50:  # need statistical significance
                curve = _build_prepayment_curve(bucket, coupon_name, fico_name, "all")
                result.prepayment_curves.append(curve)

    print(f"  Built {len(result.prepayment_curves)} prepayment curves")
    return result


def _bucket_stats(loans: list[CASLoan]) -> dict:
    """Compute summary stats for a bucket of loans."""
    n = len(loans)
    prepaid = sum(1 for l in loans if l.zero_bal_code in ("01", "1"))
    defaulted = sum(1 for l in loans if l.zero_bal_code in ("03", "09", "15", "3", "9"))
    ficos = [l.fico for l in loans if l.fico > 0]
    ltvs = [l.orig_ltv for l in loans if l.orig_ltv > 0]
    rates = [l.orig_rate for l in loans if l.orig_rate > 0]

    return {
        "count": n,
        "prepaid": prepaid,
        "defaulted": defaulted,
        "prepaid_pct": round(prepaid / n * 100, 2) if n > 0 else 0,
        "default_pct": round(defaulted / n * 100, 3) if n > 0 else 0,
        "avg_fico": round(float(np.mean(ficos)), 0) if ficos else 0,
        "avg_ltv": round(float(np.mean(ltvs)), 1) if ltvs else 0,
        "avg_rate": round(float(np.mean(rates)), 3) if rates else 0,
        "avg_upb": round(float(np.mean([l.orig_upb for l in loans if l.orig_upb > 0])), 0),
    }


def _build_prepayment_curve(
    loans: list[CASLoan],
    coupon_bucket: str,
    fico_bucket: str,
    ltv_bucket: str,
    max_age: int = 120,
) -> PrepaymentCurve:
    """Build monthly prepayment/default curve from loan data."""
    active = np.zeros(max_age)
    prepaid = np.zeros(max_age)
    defaulted = np.zeros(max_age)

    for loan in loans:
        age = loan.loan_age
        if 0 <= age < max_age:
            active[age] += 1
            if loan.zero_bal_code in ("01", "1"):
                prepaid[age] += 1
            elif loan.zero_bal_code in ("03", "09", "15", "3", "9"):
                defaulted[age] += 1

    # SMM = prepaid / active (where statistically significant)
    smm = np.where(active > 10, prepaid / active, 0)
    cpr = 1 - (1 - smm) ** 12
    cdr = np.where(active > 10, defaulted / active * 12, 0)  # annualized

    return PrepaymentCurve(
        coupon_bucket=coupon_bucket,
        fico_bucket=fico_bucket,
        ltv_bucket=ltv_bucket,
        loan_count=len(loans),
        monthly_smm=smm,
        monthly_cpr=cpr,
        monthly_cdr=cdr,
        active_counts=active,
        prepaid_counts=prepaid,
        defaulted_counts=defaulted,
    )


def _safe_float(s: str) -> float:
    try:
        return float(s.strip()) if s.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _safe_int(s: str) -> int:
    try:
        return int(float(s.strip())) if s.strip() else 0
    except (ValueError, AttributeError):
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# REMIC Deal → Expert Demonstration Conversion
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from REMIC structure codes to ActionType
_REMIC_PRINCIPAL_TO_ACTION = {
    "SEQ": ActionType.ADD_SEQ,
    "Sequential": ActionType.ADD_SEQ,
    "PAC": ActionType.ADD_PAC,
    "TAC": ActionType.ADD_TAC,
    "SUP": ActionType.ADD_SUPPORT,
    "Support": ActionType.ADD_SUPPORT,
    "PT": ActionType.ADD_SEQ,       # passthrough → sequential
    "AD": ActionType.ADD_Z_BOND,     # accretion directed → z-bond
    "VADM": ActionType.ADD_Z_BOND,
    "NAS": ActionType.ADD_SEQ,       # NAS → sequential with lockout
    "SCHED": ActionType.ADD_SEQ,     # schedule → sequential
    "Z": ActionType.ADD_Z_BOND,
}

_REMIC_INTEREST_TO_ACTION = {
    "FIX": None,  # default, handled by principal type
    "Fixed": None,
    "FLT": ActionType.ADD_FLOATER,
    "Floating": ActionType.ADD_FLOATER,
    "INV": ActionType.ADD_INV_FLOAT,
    "Inverse": ActionType.ADD_INV_FLOAT,
    "IO": ActionType.ADD_IO,
    "PO": ActionType.ADD_PO,
    "Z": ActionType.ADD_Z_BOND,
    "Z-Accrual": ActionType.ADD_Z_BOND,
}


@dataclass
class ExpertDemonstration:
    """An expert action sequence from a real REMIC deal."""
    deal_id: str
    dealer: str
    deal_type: str
    period: str
    total_issuance: float
    actions: list[list[int]]  # list of [action_type, tranche_idx, size_bucket, coupon_bucket]
    structure_types: list[str]
    n_groups: int


def remic_deal_to_actions(deal: REMICDeal, collateral_balance: float = 100_000_000) -> ExpertDemonstration:
    """Convert a real REMIC deal into expert action sequence.

    Maps each group's principal/interest types to RL actions.
    Sizes are mapped to the nearest size bucket.
    Coupons are mapped to the nearest coupon offset bucket.
    """
    actions = []
    structure_types = []
    size_buckets = np.linspace(0.02, 0.50, 20)
    coupon_offsets = np.linspace(-2.0, 1.0, 20)

    # Estimate deal collateral coupon from groups
    coupons = [g.get('coupon', 0) for g in deal.groups if g.get('coupon', 0) > 0]
    deal_coupon = np.mean(coupons) if coupons else 5.5

    for i, group in enumerate(deal.groups):
        principal_types = [t.strip() for t in group.get('principal_type', '').split('/') if t.strip()]
        interest_types = [t.strip() for t in group.get('interest_type', '').split('/') if t.strip()]
        issuance = group.get('bond_issuance', 0) or 0
        notional = group.get('notional', 0) or 0
        coupon = group.get('coupon', deal_coupon) or deal_coupon

        if not principal_types and not interest_types:
            continue

        # Determine action type from interest type first (more specific)
        action_type = None
        for it in interest_types:
            if it in _REMIC_INTEREST_TO_ACTION and _REMIC_INTEREST_TO_ACTION[it] is not None:
                action_type = _REMIC_INTEREST_TO_ACTION[it]
                break

        # Fall back to principal type
        if action_type is None:
            for pt in principal_types:
                if pt in _REMIC_PRINCIPAL_TO_ACTION:
                    action_type = _REMIC_PRINCIPAL_TO_ACTION[pt]
                    break

        if action_type is None:
            action_type = ActionType.ADD_SEQ  # default fallback

        # Map size to bucket
        effective_balance = deal.total_bond_issuance if deal.total_bond_issuance > 0 else collateral_balance
        if action_type == ActionType.ADD_IO:
            size_pct = (notional / effective_balance) if notional > 0 and effective_balance > 0 else 0.10
        else:
            size_pct = (issuance / effective_balance) if issuance > 0 and effective_balance > 0 else 0.10
        size_pct = np.clip(size_pct, 0.02, 0.50)
        size_idx = int(np.argmin(np.abs(size_buckets - size_pct)))

        # Map coupon to offset bucket
        coupon_offset = coupon - deal_coupon
        coupon_offset = np.clip(coupon_offset, -2.0, 1.0)
        coupon_idx = int(np.argmin(np.abs(coupon_offsets - coupon_offset)))

        tranche_idx = min(i, 9)  # cap at 10 tranches

        actions.append([int(action_type), tranche_idx, size_idx, coupon_idx])
        structure_types.extend(principal_types + interest_types)

    # Add execute action
    actions.append([int(ActionType.EXECUTE_DEAL), 0, 0, 0])

    return ExpertDemonstration(
        deal_id=deal.series,
        dealer=deal.dealer,
        deal_type=deal.deal_type,
        period=f"{deal.month} {deal.year}",
        total_issuance=deal.total_bond_issuance,
        actions=actions,
        structure_types=list(set(structure_types)),
        n_groups=len(deal.groups),
    )


def process_remic_deals(remic_dir: str) -> list[ExpertDemonstration]:
    """Load all REMIC data and convert to expert demonstrations."""
    print(f"Loading REMIC data from {remic_dir}...")
    t0 = time.time()
    deals = load_all_remic_data(remic_dir)
    print(f"  Loaded {len(deals)} deals in {time.time()-t0:.1f}s")

    # Filter to single-family deals (most relevant for training)
    sf_deals = [d for d in deals if 'Single' in d.deal_type or 'SF' in d.deal_type or not d.deal_type]
    print(f"  {len(sf_deals)} single-family deals")

    demos = []
    for deal in sf_deals:
        if deal.groups:
            demo = remic_deal_to_actions(deal)
            if len(demo.actions) >= 3:  # at least 2 tranches + execute
                demos.append(demo)

    print(f"  Generated {len(demos)} expert demonstrations")

    # Summary stats
    if demos:
        avg_actions = np.mean([len(d.actions) for d in demos])
        all_types = defaultdict(int)
        dealer_counts = defaultdict(int)
        for d in demos:
            for t in d.structure_types:
                all_types[t] += 1
            if d.dealer:
                dealer_counts[d.dealer] += 1

        print(f"  Avg actions/deal: {avg_actions:.1f}")
        print(f"  Structure types: {dict(sorted(all_types.items(), key=lambda x: -x[1])[:10])}")
        top_dealers = sorted(dealer_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  Top dealers: {dict(top_dealers)}")

    return demos


# ═══════════════════════════════════════════════════════════════════════════
# Model Recalibration from Processed Data
# ═══════════════════════════════════════════════════════════════════════════

def recalibrate_from_cas(cas_result: CASProcessingResult, existing_cal: dict) -> dict:
    """Update calibration parameters using CAS loan-level data."""
    cal = existing_cal.copy()

    if cas_result.total_loans == 0:
        return cal

    # Update prepayment model
    if "prepayment" in cal:
        prep = cal["prepayment"]
        # Compute empirical CPR from the loan data
        if cas_result.prepaid_pct > 0:
            # Base CPR: annualized from the fraction that prepaid
            # Since these are recent originations (10-month seasoning), adjust
            prep["base_cpr"] = round(cas_result.prepaid_pct / 100 * 12 / max(1, 10), 4)

        # FICO adjustments from bucketed data
        fico_adj = {}
        base_rate = cas_result.by_fico.get("740-780", {}).get("prepaid_pct", 1)
        if base_rate > 0:
            for bucket, stats in cas_result.by_fico.items():
                if stats.get("prepaid_pct", 0) > 0:
                    fico_adj[bucket] = round(stats["prepaid_pct"] / base_rate, 3)
        prep["fico_adjustments"] = fico_adj

        # Coupon adjustments
        coupon_adj = {}
        base_rate = cas_result.by_coupon.get("6-7%", {}).get("prepaid_pct", 1)
        if base_rate > 0:
            for bucket, stats in cas_result.by_coupon.items():
                if stats.get("prepaid_pct", 0) > 0:
                    coupon_adj[bucket] = round(stats["prepaid_pct"] / base_rate, 3)
        prep["coupon_adjustments"] = coupon_adj

        prep["calibration_date"] = str(np.datetime64("today"))
        prep["loan_count"] = cas_result.total_loans
        prep["observation_count"] = cas_result.total_loans

    # Update credit model
    if "credit" in cal:
        cred = cal["credit"]
        if cas_result.defaulted_pct > 0:
            # Base annual default rate
            cred["base_annual_default_rate"] = round(cas_result.defaulted_pct / 100 * 12 / max(1, 10), 5)

        # LTV multipliers from data
        base_default = cas_result.by_ltv.get("70-80", {}).get("default_pct", 0.001)
        if base_default > 0:
            ltv_mults = []
            for bucket in ["<70", "70-80", "80-90", "90-95", "95-100"]:
                stats = cas_result.by_ltv.get(bucket, {})
                dpct = stats.get("default_pct", base_default)
                ltv_mults.append(round(dpct / base_default, 2))
            if len(ltv_mults) >= 5:
                cred["ltv_multipliers_cas"] = ltv_mults

        cred["calibration_date"] = str(np.datetime64("today"))
        cred["loan_count"] = cas_result.total_loans

    return cal


# ═══════════════════════════════════════════════════════════════════════════
# Calibrated Market Simulator
# ═══════════════════════════════════════════════════════════════════════════

def build_calibrated_scenarios_from_remic(
    demos: list[ExpertDemonstration],
    cas_result: Optional[CASProcessingResult] = None,
) -> list[dict]:
    """Build realistic market scenarios from REMIC data + CAS calibration.

    Each scenario has:
    - Market conditions (rate regime implied by coupon levels)
    - Collateral characteristics (from CAS data)
    - Expert structure (from REMIC deal)
    """
    scenarios = []
    for demo in demos:
        if demo.total_issuance <= 0:
            continue

        # Infer rate regime from deal period and structure
        has_io = "IO" in demo.structure_types
        has_inv = "INV" in demo.structure_types or "Inverse" in demo.structure_types
        has_z = "Z" in demo.structure_types or "Z-Accrual" in demo.structure_types

        # More complex structures suggest steeper curves / more opportunities
        if has_inv and has_io:
            regime = "steep"
        elif has_z and has_io:
            regime = "normal"
        elif has_io:
            regime = "normal"
        else:
            regime = "flat"

        scenario = {
            "deal_id": demo.deal_id,
            "dealer": demo.dealer,
            "period": demo.period,
            "total_issuance": demo.total_issuance,
            "regime": regime,
            "expert_actions": demo.actions,
            "structure_types": demo.structure_types,
            "n_tranches": demo.n_groups,
        }

        # Add CAS-derived collateral stats if available
        if cas_result and cas_result.total_loans > 0:
            scenario["avg_fico"] = cas_result.avg_fico
            scenario["avg_ltv"] = cas_result.avg_ltv
            scenario["avg_rate"] = cas_result.avg_rate

        scenarios.append(scenario)

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    cas_file: Optional[str] = None,
    remic_dir: Optional[str] = None,
    calibration_file: Optional[str] = None,
    output_dir: str = ".",
    max_cas_rows: int = 0,
) -> dict:
    """Run the complete data processing pipeline.

    1. Process CAS loan data
    2. Process REMIC deals
    3. Recalibrate models
    4. Generate training scenarios
    5. Save everything
    """
    print("=" * 60)
    print("CMO Agent Data Processing Pipeline")
    print("=" * 60)
    t0 = time.time()

    results = {
        "cas": None,
        "remic_demos": [],
        "calibration": None,
        "scenarios": [],
    }

    # Step 1: CAS loan data
    cas_result = None
    if cas_file and os.path.exists(cas_file):
        print(f"\n[1/4] Processing CAS data: {cas_file}")
        cas_result = process_cas_data(cas_file, max_rows=max_cas_rows)
        results["cas"] = {
            "total_loans": cas_result.total_loans,
            "avg_fico": round(cas_result.avg_fico, 0),
            "avg_ltv": round(cas_result.avg_ltv, 1),
            "avg_rate": round(cas_result.avg_rate, 3),
            "prepaid_pct": round(cas_result.prepaid_pct, 2),
            "defaulted_pct": round(cas_result.defaulted_pct, 3),
            "by_coupon": cas_result.by_coupon,
            "by_fico": cas_result.by_fico,
            "by_ltv": cas_result.by_ltv,
            "by_purpose": cas_result.by_purpose,
            "by_state": cas_result.by_state,
            "n_prepayment_curves": len(cas_result.prepayment_curves),
        }

        # Save CAS stats
        cas_output = os.path.join(output_dir, "cas_calibration.json")
        with open(cas_output, 'w') as f:
            json.dump(results["cas"], f, indent=2)
        print(f"  Saved CAS stats to {cas_output}")
    else:
        print("\n[1/4] Skipping CAS data (file not provided)")

    # Step 2: REMIC deals
    demos = []
    if remic_dir and os.path.exists(remic_dir):
        print(f"\n[2/4] Processing REMIC deals: {remic_dir}")
        demos = process_remic_deals(remic_dir)
        results["remic_demos"] = [
            {
                "deal_id": d.deal_id,
                "dealer": d.dealer,
                "period": d.period,
                "n_actions": len(d.actions),
                "structure_types": d.structure_types,
                "total_issuance": d.total_issuance,
            }
            for d in demos
        ]

        # Save expert demos
        demo_output = os.path.join(output_dir, "expert_demonstrations.json")
        with open(demo_output, 'w') as f:
            json.dump([
                {
                    "deal_id": d.deal_id,
                    "dealer": d.dealer,
                    "period": d.period,
                    "actions": d.actions,
                    "structure_types": d.structure_types,
                    "n_groups": d.n_groups,
                    "total_issuance": d.total_issuance,
                }
                for d in demos
            ], f, indent=2)
        print(f"  Saved {len(demos)} demonstrations to {demo_output}")
    else:
        print("\n[2/4] Skipping REMIC data (dir not provided)")

    # Step 3: Recalibrate
    print(f"\n[3/4] Recalibrating models...")
    existing_cal = {}
    if calibration_file and os.path.exists(calibration_file):
        with open(calibration_file) as f:
            existing_cal = json.load(f)

    if cas_result:
        updated_cal = recalibrate_from_cas(cas_result, existing_cal)
        results["calibration"] = updated_cal

        cal_output = os.path.join(output_dir, "calibration_results.json")
        with open(cal_output, 'w') as f:
            json.dump(updated_cal, f, indent=2)
        print(f"  Updated calibration saved to {cal_output}")
    else:
        results["calibration"] = existing_cal
        print("  No CAS data available for recalibration, keeping existing")

    # Step 4: Build training scenarios
    print(f"\n[4/4] Building training scenarios...")
    if demos:
        scenarios = build_calibrated_scenarios_from_remic(demos, cas_result)
        results["scenarios"] = scenarios

        scenario_output = os.path.join(output_dir, "training_scenarios.json")
        with open(scenario_output, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"  Saved {len(scenarios)} scenarios to {scenario_output}")
    else:
        print("  No REMIC demos available for scenario generation")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  CAS loans processed: {cas_result.total_loans if cas_result else 0:,}")
    print(f"  REMIC demos generated: {len(demos)}")
    print(f"  Training scenarios: {len(results['scenarios'])}")
    print(f"{'='*60}")

    return results
