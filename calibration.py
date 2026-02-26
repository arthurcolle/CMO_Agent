"""
Model Calibration Pipeline.

Ingests Fannie Mae loan-level data (acquisition + performance) to calibrate:
1. Prepayment model: base CPR, refi sensitivity, burnout, seasonal factors
2. Credit model: base CDR, LTV/FICO/unemployment elasticities, LGD
3. Hull-White rate model: mean reversion, volatility (from FRED historical rates)
4. MBS-Treasury spread model

Designed for streaming processing — handles 50GB+ performance files
without loading everything into memory.
"""
import os
import csv
import io
import json
import zipfile
import gzip
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional, Generator
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Result Container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrepaymentCalibration:
    """Calibrated prepayment model parameters."""
    base_cpr: float = 0.06           # Annualized base CPR (turnover)
    refi_multiplier: float = 0.3     # Refi incentive sensitivity
    refi_threshold_bps: float = 50   # Min incentive for refi to kick in
    burnout_rate: float = 0.02       # Monthly burnout decay
    max_cpr: float = 0.70            # Cap
    seasonal_factors: list = field(default_factory=lambda: [
        0.88, 0.88, 0.92, 1.02, 1.10, 1.18,
        1.14, 1.08, 1.00, 0.96, 0.90, 0.88,
    ])
    # By-coupon adjustments
    coupon_adjustments: dict = field(default_factory=dict)
    # By-vintage adjustments
    vintage_adjustments: dict = field(default_factory=dict)
    # By-FICO band adjustments
    fico_adjustments: dict = field(default_factory=dict)
    # Calibration metadata
    calibration_date: str = ""
    loan_count: int = 0
    observation_count: int = 0
    r_squared: float = 0.0


@dataclass
class CreditCalibration:
    """Calibrated credit model parameters."""
    base_annual_default_rate: float = 0.003  # 0.3% for current GSE cohorts
    ltv_breakpoints: list = field(default_factory=lambda: [60, 70, 80, 90, 95, 100, 110, 120])
    ltv_multipliers: list = field(default_factory=lambda: [0.3, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0, 10.0])
    fico_base: int = 740
    fico_sensitivity: float = 0.02
    peak_default_month: int = 36
    base_lgd: float = 0.35
    foreclosure_cost: float = 0.10
    unemployment_elasticity: float = 2.5
    hpa_elasticity: float = -3.0
    calibration_date: str = ""
    loan_count: int = 0


@dataclass
class RateModelCalibration:
    """Calibrated Hull-White / rate model parameters."""
    hw_mean_reversion: float = 0.03
    hw_volatility: float = 0.008       # Annualized short rate vol
    mortgage_treasury_spread: float = 1.70  # Percentage points
    term_premium: float = 1.50
    calibration_date: str = ""
    data_start: str = ""
    data_end: str = ""


@dataclass
class FullCalibration:
    """Complete calibration result for all models."""
    prepayment: PrepaymentCalibration = field(default_factory=PrepaymentCalibration)
    credit: CreditCalibration = field(default_factory=CreditCalibration)
    rate_model: RateModelCalibration = field(default_factory=RateModelCalibration)
    calibration_timestamp: str = ""

    def save(self, filepath: str):
        """Save calibration to JSON."""
        self.calibration_timestamp = datetime.now().isoformat()
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'FullCalibration':
        """Load calibration from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        cal = cls()
        cal.prepayment = PrepaymentCalibration(**data.get("prepayment", {}))
        cal.credit = CreditCalibration(**data.get("credit", {}))
        cal.rate_model = RateModelCalibration(**data.get("rate_model", {}))
        cal.calibration_timestamp = data.get("calibration_timestamp", "")
        return cal


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Fannie Mae Data Parser
# ═══════════════════════════════════════════════════════════════════════════

def _stream_pipe_delimited(filepath: str, max_rows: int = 0) -> Generator[list[str], None, None]:
    """
    Stream rows from a pipe-delimited file (plain text, .gz, or inside .zip).
    Yields one row at a time — never loads full file into memory.
    """
    count = 0

    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.csv') or name.endswith('.txt') or '|' in name or not name.endswith('/'):
                    with zf.open(name) as member:
                        for line in io.TextIOWrapper(member, encoding='utf-8', errors='replace'):
                            fields = line.strip().split('|')
                            if len(fields) > 5:
                                yield fields
                                count += 1
                                if max_rows and count >= max_rows:
                                    return
    elif filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) > 5:
                    yield fields
                    count += 1
                    if max_rows and count >= max_rows:
                        return
    else:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) > 5:
                    yield fields
                    count += 1
                    if max_rows and count >= max_rows:
                        return


def _find_data_files(data_dir: str, pattern: str = "") -> list[str]:
    """Find all data files in directory matching pattern."""
    files = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return files
    for ext in ('*.txt', '*.csv', '*.zip', '*.gz'):
        for f in data_path.glob(ext):
            if not pattern or pattern.lower() in f.name.lower():
                files.append(str(f))
    # Also check subdirectories one level deep
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            for ext in ('*.txt', '*.csv', '*.zip', '*.gz'):
                for f in subdir.glob(ext):
                    if not pattern or pattern.lower() in f.name.lower():
                        files.append(str(f))
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════════
# Prepayment Calibration from Loan-Level Data
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _PrepayAccumulator:
    """Accumulates prepayment statistics during streaming pass."""
    # Monthly: active loans and prepayments by age bucket
    active_by_age: np.ndarray = field(default_factory=lambda: np.zeros(361))
    prepaid_by_age: np.ndarray = field(default_factory=lambda: np.zeros(361))
    # By coupon bucket (0.5% increments from 2.0% to 8.0%)
    active_by_coupon_age: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(361)))
    prepaid_by_coupon_age: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(361)))
    # By FICO band
    active_by_fico_age: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(361)))
    prepaid_by_fico_age: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(361)))
    # By calendar month (for seasonality)
    active_by_month: np.ndarray = field(default_factory=lambda: np.zeros(12))
    prepaid_by_month: np.ndarray = field(default_factory=lambda: np.zeros(12))
    # Defaults (for credit calibration too)
    defaulted_by_age: np.ndarray = field(default_factory=lambda: np.zeros(361))
    # Counters
    total_observations: int = 0
    total_loans: int = 0


def _coupon_bucket(rate: float) -> str:
    """Round rate to nearest 0.5% bucket."""
    bucket = round(rate * 2) / 2
    return f"{bucket:.1f}"


def _fico_band(fico: int) -> str:
    """Map FICO to band."""
    if fico < 620:
        return "<620"
    elif fico < 660:
        return "620-659"
    elif fico < 700:
        return "660-699"
    elif fico < 740:
        return "700-739"
    elif fico < 780:
        return "740-779"
    else:
        return "780+"


def calibrate_prepayment_from_files(
    performance_path: str,
    acquisition_path: Optional[str] = None,
    max_rows: int = 0,
    progress_interval: int = 5_000_000,
) -> PrepaymentCalibration:
    """
    Calibrate prepayment model from Fannie Mae loan-level data.

    Streams through the performance file, building:
    - Monthly SMM/CPR by loan age (seasoning curve)
    - CPR by coupon bucket (refi sensitivity)
    - CPR by FICO band
    - Seasonal adjustment factors
    - Burnout estimates (CPR by pool factor proxy)

    Args:
        performance_path: Path to Performance_All.zip or individual performance file
        acquisition_path: Optional path to acquisition data (for FICO/rate lookup)
        max_rows: Limit rows for testing (0 = unlimited)
        progress_interval: Print progress every N rows
    """
    # If we have acquisition data, build lookup tables
    loan_rates = {}
    loan_ficos = {}
    if acquisition_path and os.path.exists(acquisition_path):
        print(f"Loading acquisition data from {acquisition_path}...")
        for fields in _stream_pipe_delimited(acquisition_path, max_rows=max_rows or 50_000_000):
            try:
                loan_id = fields[1] if len(fields) > 1 else ""
                rate = float(fields[5]) if len(fields) > 5 and fields[5] else 0
                fico = int(fields[14]) if len(fields) > 14 and fields[14] else 0
                if loan_id and rate > 0:
                    loan_rates[loan_id] = rate
                if loan_id and fico > 0:
                    loan_ficos[loan_id] = fico
            except (ValueError, IndexError):
                continue
        print(f"  Loaded {len(loan_rates)} loan rates, {len(loan_ficos)} FICOs")

    # Stream through performance data
    acc = _PrepayAccumulator()
    print(f"Streaming performance data from {performance_path}...")

    for fields in _stream_pipe_delimited(performance_path, max_rows=max_rows):
        acc.total_observations += 1

        if acc.total_observations % progress_interval == 0:
            print(f"  Processed {acc.total_observations:,} rows...")

        try:
            # Performance file columns:
            # 0: LOAN_ID, 1: REPORT_PERIOD, 3: CURRENT_RATE, 4: CURRENT_BALANCE,
            # 5: LOAN_AGE, 9: DELINQUENCY_STATUS, 11: ZERO_BALANCE_CODE
            loan_id = fields[0]
            loan_age = int(fields[5]) if fields[5] else -1
            zero_bal_code = fields[11].strip() if len(fields) > 11 else ""
            report_period = fields[1] if len(fields) > 1 else ""

            if loan_age < 0 or loan_age > 360:
                continue

            # Track active loans
            acc.active_by_age[loan_age] += 1

            # Calendar month for seasonality
            if report_period and len(report_period) >= 6:
                try:
                    cal_month = int(report_period[4:6]) - 1  # 0-indexed
                    if 0 <= cal_month < 12:
                        acc.active_by_month[cal_month] += 1
                except ValueError:
                    pass

            # Track by coupon bucket
            rate = loan_rates.get(loan_id, 0)
            if rate > 0:
                bucket = _coupon_bucket(rate)
                acc.active_by_coupon_age[bucket][loan_age] += 1

            # Track by FICO band
            fico = loan_ficos.get(loan_id, 0)
            if fico > 0:
                band = _fico_band(fico)
                acc.active_by_fico_age[band][loan_age] += 1

            # Prepayment event (zero_balance_code = "01")
            if zero_bal_code == "01":
                acc.prepaid_by_age[loan_age] += 1
                if rate > 0:
                    acc.prepaid_by_coupon_age[_coupon_bucket(rate)][loan_age] += 1
                if fico > 0:
                    acc.prepaid_by_fico_age[_fico_band(fico)][loan_age] += 1
                if report_period and len(report_period) >= 6:
                    try:
                        cal_month = int(report_period[4:6]) - 1
                        if 0 <= cal_month < 12:
                            acc.prepaid_by_month[cal_month] += 1
                    except ValueError:
                        pass

            # Default events (03=short sale, 09=REO, 15=note sale)
            if zero_bal_code in ("03", "09", "15"):
                acc.defaulted_by_age[loan_age] += 1

        except (ValueError, IndexError):
            continue

    print(f"  Done: {acc.total_observations:,} total observations")

    # ─── Compute calibrated parameters ────────────────────────────────────

    # 1. Monthly SMM by age → base CPR (turnover component)
    smm_by_age = np.where(
        acc.active_by_age > 100,  # min 100 loans for stat significance
        acc.prepaid_by_age / acc.active_by_age,
        0,
    )
    cpr_by_age = 1 - (1 - smm_by_age) ** 12

    # Base CPR = average CPR for seasoned loans (age 30-120) in low-incentive environment
    # Approximate: use overall average as starting point
    seasoned_mask = (np.arange(361) >= 30) & (np.arange(361) <= 120) & (acc.active_by_age > 1000)
    if seasoned_mask.any():
        base_cpr = float(np.mean(cpr_by_age[seasoned_mask]))
    else:
        base_cpr = 0.06

    # 2. Seasonal factors
    total_smm_by_month = np.where(
        acc.active_by_month > 0,
        acc.prepaid_by_month / acc.active_by_month,
        0,
    )
    if total_smm_by_month.sum() > 0:
        avg_smm = total_smm_by_month.mean()
        seasonal = (total_smm_by_month / avg_smm if avg_smm > 0
                    else np.ones(12))
        seasonal = np.clip(seasonal, 0.5, 1.5)
        seasonal_factors = [round(float(s), 3) for s in seasonal]
    else:
        seasonal_factors = [0.88, 0.88, 0.92, 1.02, 1.10, 1.18,
                           1.14, 1.08, 1.00, 0.96, 0.90, 0.88]

    # 3. Coupon-level CPR adjustments (refi sensitivity proxy)
    coupon_adjustments = {}
    for bucket, active_arr in acc.active_by_coupon_age.items():
        prepaid_arr = acc.prepaid_by_coupon_age[bucket]
        total_active = active_arr.sum()
        total_prepaid = prepaid_arr.sum()
        if total_active > 10000:
            bucket_smm = total_prepaid / total_active
            bucket_cpr = 1 - (1 - bucket_smm) ** 12
            coupon_adjustments[bucket] = round(float(bucket_cpr), 4)

    # 4. FICO band adjustments
    fico_adjustments = {}
    for band, active_arr in acc.active_by_fico_age.items():
        prepaid_arr = acc.prepaid_by_fico_age[band]
        total_active = active_arr.sum()
        total_prepaid = prepaid_arr.sum()
        if total_active > 10000:
            band_smm = total_prepaid / total_active
            band_cpr = 1 - (1 - band_smm) ** 12
            fico_adjustments[band] = round(float(band_cpr), 4)

    # 5. Refi multiplier estimation
    # If we have coupon buckets, estimate sensitivity as slope of CPR vs coupon
    refi_mult = 0.3  # default
    if len(coupon_adjustments) >= 3:
        coupons = sorted(coupon_adjustments.keys(), key=float)
        rates = [float(c) for c in coupons]
        cprs = [coupon_adjustments[c] for c in coupons]
        if len(rates) > 2:
            # Simple linear regression: CPR = a + b * rate
            coeffs = np.polyfit(rates, cprs, 1)
            refi_mult = max(0.05, min(1.0, abs(float(coeffs[0])) / 10))

    # 6. Burnout estimation from age curve
    # Burnout = how much CPR declines after peak seasoning
    peak_age = int(np.argmax(cpr_by_age[:120])) if cpr_by_age[:120].max() > 0 else 36
    if peak_age > 0 and cpr_by_age[peak_age] > 0:
        late_mask = (np.arange(361) >= 120) & (np.arange(361) <= 240) & (acc.active_by_age > 1000)
        if late_mask.any():
            late_cpr = float(np.mean(cpr_by_age[late_mask]))
            burnout_ratio = late_cpr / cpr_by_age[peak_age] if cpr_by_age[peak_age] > 0 else 1.0
            # Convert to monthly decay rate
            months_diff = 180 - peak_age
            if months_diff > 0 and burnout_ratio < 1.0:
                burnout_rate = -np.log(max(0.01, burnout_ratio)) / months_diff
                burnout_rate = float(np.clip(burnout_rate, 0.001, 0.05))
            else:
                burnout_rate = 0.02
        else:
            burnout_rate = 0.02
    else:
        burnout_rate = 0.02

    # 7. Compute R² — validate model predictions against empirical CPR curve
    # Load CAS historical seasoning curve as validation baseline if available
    r_squared = 0.0
    cas_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cas_historical_calibration.json",
    )
    if os.path.exists(cas_path):
        try:
            with open(cas_path, 'r') as f:
                cas_data = json.load(f)
            cas_cpr = cas_data.get("seasoning_cpr", [])
            if cas_cpr and len(cas_cpr) >= 120:
                # CAS seasoning_cpr is cumulative % — convert to marginal monthly CPR
                # Marginal CPR at month m ≈ Δcumulative / (100 - cumulative_prev)
                cas_marginal = np.zeros(min(len(cas_cpr), 361))
                for m in range(1, len(cas_marginal)):
                    prev = cas_cpr[m - 1]
                    curr = cas_cpr[m]
                    remaining = 100.0 - prev
                    if remaining > 0.1:
                        smm = (curr - prev) / remaining
                        cas_marginal[m] = 1 - (1 - max(0, smm)) ** 12
                    else:
                        cas_marginal[m] = cas_marginal[m - 1]

                # Build model predicted CPR for same months
                n = min(len(cas_marginal), 361)
                valid_mask = (np.arange(n) >= 6) & (np.arange(n) <= min(n - 1, 300))
                # Filter to months with sufficient data in both curves
                valid_mask &= (cas_marginal[:n] > 0.001) & (cpr_by_age[:n] > 0)
                if valid_mask.sum() >= 20:
                    actual = cas_marginal[:n][valid_mask]
                    predicted = cpr_by_age[:n][valid_mask]
                    ss_res = float(np.sum((actual - predicted) ** 2))
                    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
                    if ss_tot > 0:
                        r_squared = max(0.0, 1.0 - ss_res / ss_tot)
                        r_squared = round(r_squared, 4)
                print(f"  R² vs CAS historical: {r_squared:.4f} ({valid_mask.sum()} months)")
        except Exception as e:
            print(f"  Warning: Could not load CAS historical for R² validation: {e}")

    # Also compute R² from the model's own internal fit if no CAS data
    if r_squared == 0.0 and seasoned_mask.any():
        actual_cpr = cpr_by_age[seasoned_mask]
        # Model prediction: base_cpr * seasonal * seasoning_ramp
        ages = np.arange(361)[seasoned_mask]
        seasoning_ramp = np.minimum(ages / 30.0, 1.0)
        predicted_cpr = base_cpr * seasoning_ramp
        ss_res = float(np.sum((actual_cpr - predicted_cpr) ** 2))
        ss_tot = float(np.sum((actual_cpr - np.mean(actual_cpr)) ** 2))
        if ss_tot > 0:
            r_squared = max(0.0, 1.0 - ss_res / ss_tot)
            r_squared = round(r_squared, 4)

    result = PrepaymentCalibration(
        base_cpr=round(base_cpr, 4),
        refi_multiplier=round(refi_mult, 4),
        burnout_rate=round(burnout_rate, 4),
        seasonal_factors=seasonal_factors,
        coupon_adjustments=coupon_adjustments,
        fico_adjustments=fico_adjustments,
        calibration_date=str(date.today()),
        loan_count=len(loan_rates) if loan_rates else 0,
        observation_count=acc.total_observations,
        r_squared=r_squared,
    )

    print(f"\nPrepayment calibration complete:")
    print(f"  Base CPR: {result.base_cpr*100:.2f}%")
    print(f"  Refi multiplier: {result.refi_multiplier:.4f}")
    print(f"  Burnout rate: {result.burnout_rate:.4f}/month")
    print(f"  R²: {result.r_squared:.4f}")
    print(f"  Seasonal factors: {result.seasonal_factors}")
    print(f"  Coupon buckets: {len(result.coupon_adjustments)}")
    print(f"  FICO bands: {len(result.fico_adjustments)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Credit Model Calibration from Loan-Level Data
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_credit_from_files(
    performance_path: str,
    acquisition_path: Optional[str] = None,
    max_rows: int = 0,
    progress_interval: int = 5_000_000,
) -> CreditCalibration:
    """
    Calibrate credit model from Fannie Mae loan-level data.

    Computes:
    - Base annual default rate by vintage/cohort
    - Default rate by LTV bucket
    - Default rate by FICO band
    - CDR seasoning curve (peak default month)
    - LGD estimates from disposition data
    """
    # Load acquisition data for LTV/FICO
    loan_ltvs = {}
    loan_ficos = {}
    if acquisition_path and os.path.exists(acquisition_path):
        print(f"Loading acquisition data for credit calibration...")
        for fields in _stream_pipe_delimited(acquisition_path, max_rows=max_rows or 50_000_000):
            try:
                loan_id = fields[1] if len(fields) > 1 else ""
                ltv = float(fields[10]) if len(fields) > 10 and fields[10] else 0
                fico = int(fields[14]) if len(fields) > 14 and fields[14] else 0
                if loan_id:
                    if ltv > 0:
                        loan_ltvs[loan_id] = ltv
                    if fico > 0:
                        loan_ficos[loan_id] = fico
            except (ValueError, IndexError):
                continue
        print(f"  Loaded {len(loan_ltvs)} LTVs, {len(loan_ficos)} FICOs")

    # Stream performance data
    active_by_age = np.zeros(361)
    defaulted_by_age = np.zeros(361)

    # By LTV bucket
    ltv_buckets = [0, 60, 70, 80, 90, 95, 100, 110, 120, 200]
    active_by_ltv = defaultdict(int)
    defaulted_by_ltv = defaultdict(int)

    # By FICO band
    active_by_fico = defaultdict(int)
    defaulted_by_fico = defaultdict(int)

    # LGD tracking
    losses = []
    total_obs = 0

    print(f"Streaming performance data for credit calibration...")

    for fields in _stream_pipe_delimited(performance_path, max_rows=max_rows):
        total_obs += 1
        if total_obs % progress_interval == 0:
            print(f"  Processed {total_obs:,} rows...")

        try:
            loan_id = fields[0]
            loan_age = int(fields[5]) if fields[5] else -1
            zero_bal_code = fields[11].strip() if len(fields) > 11 else ""
            current_balance = float(fields[4]) if len(fields) > 4 and fields[4] else 0

            if loan_age < 0 or loan_age > 360:
                continue

            active_by_age[loan_age] += 1

            # LTV bucket tracking
            ltv = loan_ltvs.get(loan_id, 0)
            if ltv > 0:
                for i in range(len(ltv_buckets) - 1):
                    if ltv_buckets[i] <= ltv < ltv_buckets[i + 1]:
                        bucket_key = f"{ltv_buckets[i]}-{ltv_buckets[i+1]}"
                        active_by_ltv[bucket_key] += 1
                        break

            # FICO band tracking
            fico = loan_ficos.get(loan_id, 0)
            if fico > 0:
                band = _fico_band(fico)
                active_by_fico[band] += 1

            # Default events
            if zero_bal_code in ("03", "09", "15"):
                defaulted_by_age[loan_age] += 1

                if ltv > 0:
                    for i in range(len(ltv_buckets) - 1):
                        if ltv_buckets[i] <= ltv < ltv_buckets[i + 1]:
                            defaulted_by_ltv[f"{ltv_buckets[i]}-{ltv_buckets[i+1]}"] += 1
                            break

                if fico > 0:
                    defaulted_by_fico[_fico_band(fico)] += 1

                # LGD from disposition data
                if len(fields) > 24 and current_balance > 0:
                    try:
                        net_proceeds = float(fields[21]) if fields[21] else 0
                        if net_proceeds > 0:
                            lgd = 1 - (net_proceeds / current_balance)
                            losses.append(max(0, min(1, lgd)))
                    except (ValueError, IndexError):
                        pass

        except (ValueError, IndexError):
            continue

    print(f"  Done: {total_obs:,} observations")

    # ─── Compute calibrated credit parameters ─────────────────────────────

    # 1. Base annual default rate
    smdr_by_age = np.where(active_by_age > 100, defaulted_by_age / active_by_age, 0)
    cdr_by_age = 1 - (1 - smdr_by_age) ** 12
    seasoned_mask = (np.arange(361) >= 12) & (np.arange(361) <= 120) & (active_by_age > 1000)
    if seasoned_mask.any():
        base_cdr = float(np.mean(cdr_by_age[seasoned_mask]))
    else:
        base_cdr = 0.003

    # 2. Peak default month
    if cdr_by_age[:120].max() > 0:
        peak_month = int(np.argmax(cdr_by_age[:120]))
    else:
        peak_month = 36

    # 3. LTV multipliers
    base_ltv_default = defaulted_by_ltv.get("80-90", 1) / max(1, active_by_ltv.get("80-90", 1))
    ltv_multipliers = {}
    for bucket_key in sorted(active_by_ltv.keys()):
        if active_by_ltv[bucket_key] > 10000:
            bucket_rate = defaulted_by_ltv[bucket_key] / active_by_ltv[bucket_key]
            mult = bucket_rate / base_ltv_default if base_ltv_default > 0 else 1.0
            ltv_multipliers[bucket_key] = round(mult, 2)

    # 4. FICO multipliers
    fico_default_rates = {}
    for band in sorted(active_by_fico.keys()):
        if active_by_fico[band] > 10000:
            rate = defaulted_by_fico[band] / active_by_fico[band]
            fico_default_rates[band] = round(rate * 12, 6)  # annualize

    # 5. LGD
    if losses:
        base_lgd = float(np.mean(losses))
    else:
        base_lgd = 0.35

    result = CreditCalibration(
        base_annual_default_rate=round(base_cdr, 6),
        peak_default_month=peak_month,
        base_lgd=round(base_lgd, 4),
        calibration_date=str(date.today()),
        loan_count=len(loan_ltvs) if loan_ltvs else 0,
    )

    print(f"\nCredit calibration complete:")
    print(f"  Base annual CDR: {result.base_annual_default_rate*100:.3f}%")
    print(f"  Peak default month: {result.peak_default_month}")
    print(f"  Base LGD: {result.base_lgd*100:.1f}%")
    print(f"  LTV multipliers: {ltv_multipliers}")
    print(f"  FICO default rates: {fico_default_rates}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Rate Model Calibration from FRED Historical Data
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_rate_model_from_fred(
    start_date: str = "2000-01-01",
) -> RateModelCalibration:
    """
    Calibrate Hull-White parameters and MBS-Treasury spread from FRED data.

    Uses historical Treasury yields and mortgage rates to estimate:
    - Hull-White mean reversion (kappa) from rate autocorrelation
    - Hull-White volatility (sigma) from rate changes
    - Mortgage-Treasury spread from PMMS vs 10Y Treasury
    """
    from .data_sources import fred_fetch_series

    print("Fetching FRED historical data for rate model calibration...")

    # Fetch 10Y Treasury yields (daily)
    treasury_10y = fred_fetch_series("DGS10", start_date=start_date, limit=5000)
    # Fetch 30yr mortgage rates (weekly)
    mortgage_30y = fred_fetch_series("MORTGAGE30US", start_date=start_date, limit=5000)
    # Fetch Fed Funds rate
    fed_funds = fred_fetch_series("FEDFUNDS", start_date=start_date, limit=5000)

    result = RateModelCalibration(calibration_date=str(date.today()))

    if not treasury_10y:
        print("  WARNING: No FRED data available, using defaults")
        return result

    # ─── Hull-White volatility from daily rate changes ─────────────────────
    rates = np.array([d["value"] for d in reversed(treasury_10y)]) / 100.0  # Convert to decimal
    if len(rates) > 20:
        # Daily changes
        daily_changes = np.diff(rates)
        # Annualized volatility (sqrt(252) for trading days)
        daily_vol = np.std(daily_changes)
        annual_vol = daily_vol * np.sqrt(252)
        result.hw_volatility = round(float(annual_vol), 5)

        # Mean reversion from AR(1) regression: dr = kappa*(theta - r)*dt + sigma*dW
        # Discrete: r_{t+1} - r_t = kappa*(theta - r_t)*(1/252) + noise
        # => Regress changes on levels
        r_t = rates[:-1]
        dr = daily_changes
        if len(r_t) > 100:
            # OLS: dr = a + b * r_t
            A = np.column_stack([np.ones_like(r_t), r_t])
            try:
                coeffs = np.linalg.lstsq(A, dr, rcond=None)[0]
                # b = -kappa / 252  => kappa = -b * 252
                kappa = -coeffs[1] * 252
                kappa = float(np.clip(kappa, 0.01, 1.0))
                result.hw_mean_reversion = round(kappa, 4)
                # theta = -a / b (long-run rate)
                if coeffs[1] != 0:
                    theta = -coeffs[0] / coeffs[1]
                    theta = float(np.clip(theta, 0.01, 0.10))
            except np.linalg.LinAlgError:
                pass

    result.data_start = treasury_10y[-1]["date"] if treasury_10y else ""
    result.data_end = treasury_10y[0]["date"] if treasury_10y else ""

    # ─── Mortgage-Treasury spread ──────────────────────────────────────────
    if mortgage_30y and treasury_10y:
        # Build date-indexed lookups
        tsy_by_date = {d["date"]: d["value"] for d in treasury_10y}

        spreads = []
        for m in mortgage_30y:
            tsy = tsy_by_date.get(m["date"])
            if tsy is not None and tsy > 0:
                spread = m["value"] - tsy
                if 0 < spread < 5:  # sanity check
                    spreads.append(spread)

        if spreads:
            # Use recent 52 weeks average
            recent_spreads = spreads[:min(52, len(spreads))]
            result.mortgage_treasury_spread = round(float(np.mean(recent_spreads)), 3)
            # Term premium ≈ 10Y - Fed Funds
            if fed_funds:
                recent_ff = np.mean([d["value"] for d in fed_funds[:12]])
                recent_10y = np.mean([d["value"] for d in treasury_10y[:60]])
                result.term_premium = round(float(recent_10y - recent_ff), 3)

    print(f"\nRate model calibration complete:")
    print(f"  HW mean reversion (kappa): {result.hw_mean_reversion}")
    print(f"  HW volatility (sigma): {result.hw_volatility}")
    print(f"  Mortgage-Treasury spread: {result.mortgage_treasury_spread}%")
    print(f"  Term premium: {result.term_premium}%")
    print(f"  Data range: {result.data_start} to {result.data_end}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Full Calibration Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_full_calibration(
    data_dir: str = "",
    performance_path: str = "",
    acquisition_path: str = "",
    output_path: str = "",
    max_rows: int = 0,
    skip_loan_level: bool = False,
    skip_rate_model: bool = False,
) -> FullCalibration:
    """
    Run complete calibration pipeline.

    Args:
        data_dir: Directory containing Fannie Mae data files
        performance_path: Direct path to performance file
        acquisition_path: Direct path to acquisition file
        output_path: Where to save calibration JSON
        max_rows: Limit rows for testing (0 = unlimited)
        skip_loan_level: Skip loan-level calibration (if no data available)
        skip_rate_model: Skip FRED-based rate model calibration
    """
    cal = FullCalibration()

    # Auto-detect files if data_dir given
    if data_dir and not performance_path:
        files = _find_data_files(data_dir)
        for f in files:
            fname = os.path.basename(f).lower()
            if 'performance' in fname and not performance_path:
                performance_path = f
            elif 'acquisition' in fname or 'harp' in fname:
                if not acquisition_path:
                    acquisition_path = f

    # 1. Rate model calibration (from FRED, no loan data needed)
    if not skip_rate_model:
        print("=" * 60)
        print("PHASE 1: Rate Model Calibration (FRED)")
        print("=" * 60)
        try:
            cal.rate_model = calibrate_rate_model_from_fred()
        except Exception as e:
            print(f"  Rate model calibration failed: {e}")

    # 2. Prepayment calibration (from loan-level data)
    if not skip_loan_level and performance_path:
        print()
        print("=" * 60)
        print("PHASE 2: Prepayment Model Calibration (Loan-Level)")
        print("=" * 60)
        try:
            cal.prepayment = calibrate_prepayment_from_files(
                performance_path, acquisition_path, max_rows=max_rows,
            )
        except Exception as e:
            print(f"  Prepayment calibration failed: {e}")

        # 3. Credit calibration (from same loan-level data)
        print()
        print("=" * 60)
        print("PHASE 3: Credit Model Calibration (Loan-Level)")
        print("=" * 60)
        try:
            cal.credit = calibrate_credit_from_files(
                performance_path, acquisition_path, max_rows=max_rows,
            )
        except Exception as e:
            print(f"  Credit calibration failed: {e}")
    elif not skip_loan_level:
        print("\nNo loan-level data found. Skipping prepayment/credit calibration.")
        print("Download from: https://datadynamics.fanniemae.com")

    # Save results
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "calibration_results.json",
        )
    cal.save(output_path)
    print(f"\nCalibration saved to: {output_path}")

    return cal


# ═══════════════════════════════════════════════════════════════════════════
# Apply Calibration to Models
# ═══════════════════════════════════════════════════════════════════════════

def apply_calibration(cal: FullCalibration):
    """
    Apply calibration results to the live model instances.
    Updates PrepaymentModelConfig, CreditModelConfig, and HullWhiteParams defaults.
    """
    from .prepayment import PrepaymentModelConfig
    from .monte_carlo import HullWhiteParams

    # Update prepayment defaults
    PrepaymentModelConfig.__dataclass_fields__['base_cpr'].default = cal.prepayment.base_cpr
    PrepaymentModelConfig.__dataclass_fields__['burnout_factor'].default = cal.prepayment.burnout_rate

    # Update Hull-White defaults
    HullWhiteParams.__dataclass_fields__['mean_reversion'].default = cal.rate_model.hw_mean_reversion
    HullWhiteParams.__dataclass_fields__['volatility'].default = cal.rate_model.hw_volatility

    print(f"Applied calibration from {cal.calibration_timestamp}")
    print(f"  Prepayment base CPR: {cal.prepayment.base_cpr*100:.2f}%")
    print(f"  HW kappa: {cal.rate_model.hw_mean_reversion}, sigma: {cal.rate_model.hw_volatility}")
    print(f"  MBS-TSY spread: {cal.rate_model.mortgage_treasury_spread}%")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_full_calibration(data_dir=data_dir, max_rows=max_rows)
