"""
Data Sources Module - Live market data connectors for self-enclosed MBS analytics.

Connectors:
- FRED API: Treasury yields, mortgage rates, housing/macro data (requires free API key)
- Treasury.gov XML: Daily Treasury par yield curve (no key needed)
- FHFA HPI: House price indices for credit models
- Fannie Mae / Freddie Mac: Loan-level performance data framework
- FINRA TRACE: Bond trade reporting (MBS/ABS)
"""
import json
import csv
import io
import os
from datetime import date, datetime
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from dataclasses import dataclass, field

import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred"
TREASURY_XML_BASE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
FHFA_HPI_BASE = "https://www.fhfa.gov/sites/default/files"


@dataclass
class DataSourceStatus:
    """Status of a data source connection."""
    name: str
    available: bool
    last_fetch: Optional[str] = None
    record_count: int = 0
    error: Optional[str] = None


def check_all_sources() -> list[DataSourceStatus]:
    """Check connectivity to all data sources."""
    results = []
    # FRED
    if FRED_API_KEY:
        try:
            data = fred_fetch_series("DGS10", limit=1)
            results.append(DataSourceStatus("FRED", True, record_count=len(data)))
        except Exception as e:
            results.append(DataSourceStatus("FRED", False, error=str(e)))
    else:
        results.append(DataSourceStatus("FRED", False, error="FRED_API_KEY not set"))

    # Treasury.gov
    try:
        curve = fetch_treasury_curve_xml()
        results.append(DataSourceStatus(
            "Treasury.gov", True,
            last_fetch=str(curve.get("date", "")),
            record_count=len(curve.get("yields", {})),
        ))
    except Exception as e:
        results.append(DataSourceStatus("Treasury.gov", False, error=str(e)))

    # FHFA HPI
    try:
        hpi = fetch_fhfa_hpi_national()
        results.append(DataSourceStatus("FHFA HPI", True, record_count=len(hpi)))
    except Exception as e:
        results.append(DataSourceStatus("FHFA HPI", False, error=str(e)))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FRED API - Federal Reserve Economic Data
# ═══════════════════════════════════════════════════════════════════════════

def _fred_request(endpoint: str, params: dict) -> dict:
    """Make a FRED API request."""
    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    params["api_key"] = FRED_API_KEY
    params["file_type"] = "json"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FRED_BASE}/{endpoint}?{query}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def fred_fetch_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500,
) -> list[dict]:
    """
    Fetch a FRED time series.

    Key series for MBS analytics:
        Treasury Yields: DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5,
                         DGS7, DGS10, DGS20, DGS30
        Mortgage Rates:  MORTGAGE30US, MORTGAGE15US, MORTGAGE5US
        Housing:         HOUST, PERMIT, HSN1F, CSUSHPINSA, MSPUS
        Macro:           UNRATE, CPIAUCSL, GDP, FEDFUNDS, T10Y2Y, T10Y3M
        MBS Specific:    WSHOMCB (MBS held by commercial banks)
    """
    params = {"series_id": series_id, "sort_order": "desc", "limit": str(limit)}
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    data = _fred_request("series/observations", params)
    observations = data.get("observations", [])
    return [
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in observations
        if obs.get("value") not in (".", "", None)
    ]


def fred_fetch_series_bulk(
    series_ids: list[str],
    start_date: str = "2006-01-01",
    end_date: Optional[str] = None,
    delay: float = 0.5,
) -> dict[str, list[dict]]:
    """Fetch multiple FRED series with rate-limit delay.

    Args:
        series_ids: List of FRED series IDs to fetch.
        start_date: Earliest observation date.
        end_date: Latest observation date (default: today).
        delay: Seconds to wait between API calls (FRED limit: 120/min).

    Returns:
        Dict mapping series_id -> list of {date, value} observations.
    """
    import time as _time

    if end_date is None:
        end_date = str(date.today())

    results: dict[str, list[dict]] = {}
    for i, sid in enumerate(series_ids):
        try:
            obs = fred_fetch_series(
                sid, start_date=start_date, end_date=end_date, limit=100_000
            )
            results[sid] = obs
            if i < len(series_ids) - 1:
                _time.sleep(delay)
        except Exception as e:
            print(f"  FRED fetch failed for {sid}: {e}")
            results[sid] = []
    return results


def fred_search(query: str, limit: int = 20) -> list[dict]:
    """Search FRED for series matching a query."""
    data = _fred_request("series/search", {
        "search_text": query.replace(" ", "+"),
        "limit": str(limit),
    })
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "frequency": s.get("frequency_short", ""),
            "units": s.get("units_short", ""),
            "last_updated": s.get("last_updated", ""),
        }
        for s in data.get("seriess", [])
    ]


# ─── FRED Treasury Curve Builder ──────────────────────────────────────────

FRED_TREASURY_SERIES = {
    1/12:  "DGS1MO",
    3/12:  "DGS3MO",
    6/12:  "DGS6MO",
    1.0:   "DGS1",
    2.0:   "DGS2",
    3.0:   "DGS3",
    5.0:   "DGS5",
    7.0:   "DGS7",
    10.0:  "DGS10",
    20.0:  "DGS20",
    30.0:  "DGS30",
}


def fetch_treasury_curve_fred(as_of: Optional[str] = None) -> dict:
    """
    Build a complete Treasury yield curve from FRED data.

    Returns dict with 'date', 'yields' (maturity_years -> yield_pct),
    and 'source' metadata.
    """
    if as_of is None:
        # Fetch latest
        target_date = None
    else:
        target_date = as_of

    yields = {}
    fetch_date = None

    for maturity, series_id in FRED_TREASURY_SERIES.items():
        try:
            params = {"series_id": series_id, "sort_order": "desc", "limit": "5"}
            if target_date:
                params["observation_end"] = target_date
            data = _fred_request("series/observations", params)
            for obs in data.get("observations", []):
                if obs.get("value") not in (".", "", None):
                    yields[maturity] = float(obs["value"])
                    if fetch_date is None:
                        fetch_date = obs["date"]
                    break
        except Exception:
            continue

    return {
        "date": fetch_date or str(date.today()),
        "yields": yields,
        "source": "FRED",
        "series_count": len(yields),
    }


# ─── FRED Mortgage Rate Fetcher ───────────────────────────────────────────

def fetch_current_mortgage_rates() -> dict:
    """Fetch current primary mortgage market survey rates from FRED."""
    series = {
        "30yr_fixed": "MORTGAGE30US",
        "15yr_fixed": "MORTGAGE15US",
        "5yr_arm": "MORTGAGE5US",
    }
    rates = {}
    for label, sid in series.items():
        try:
            obs = fred_fetch_series(sid, limit=1)
            if obs:
                rates[label] = {
                    "rate": obs[0]["value"],
                    "date": obs[0]["date"],
                }
        except Exception:
            continue
    return rates


def fetch_mortgage_rate_history(
    product: str = "30yr",
    start_date: str = "2000-01-01",
) -> list[dict]:
    """Fetch historical mortgage rates. product: '30yr', '15yr', '5yr_arm'."""
    series_map = {"30yr": "MORTGAGE30US", "15yr": "MORTGAGE15US", "5yr_arm": "MORTGAGE5US"}
    sid = series_map.get(product, "MORTGAGE30US")
    return fred_fetch_series(sid, start_date=start_date, limit=5000)


# ─── FRED Housing / Macro Data ────────────────────────────────────────────

def fetch_housing_starts(start_date: str = "2000-01-01") -> list[dict]:
    """Fetch housing starts (SAAR, thousands)."""
    return fred_fetch_series("HOUST", start_date=start_date, limit=5000)


def fetch_case_shiller_national(start_date: str = "2000-01-01") -> list[dict]:
    """Fetch S&P/Case-Shiller National Home Price Index."""
    return fred_fetch_series("CSUSHPINSA", start_date=start_date, limit=5000)


def fetch_unemployment_rate(start_date: str = "2000-01-01") -> list[dict]:
    """Fetch civilian unemployment rate."""
    return fred_fetch_series("UNRATE", start_date=start_date, limit=5000)


def fetch_mbs_bank_holdings(start_date: str = "2000-01-01") -> list[dict]:
    """Fetch MBS held by commercial banks (billions $)."""
    return fred_fetch_series("WSHOMCB", start_date=start_date, limit=5000)


def fetch_spread_history(start_date: str = "2000-01-01") -> dict:
    """Fetch Treasury spread indicators (10Y-2Y, 10Y-3M)."""
    return {
        "10y_2y": fred_fetch_series("T10Y2Y", start_date=start_date, limit=5000),
        "10y_3m": fred_fetch_series("T10Y3M", start_date=start_date, limit=5000),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Treasury.gov - Daily Treasury Par Yield Curve (No API key needed)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_treasury_curve_xml(year: Optional[int] = None) -> dict:
    """
    Fetch Treasury par yield curve from Treasury.gov CSV feed.
    No API key required. Returns the most recent available curve.
    """
    if year is None:
        year = date.today().year

    url = (
        f"https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/{year}/all"
        f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
    )

    req = Request(url, headers={
        "User-Agent": "CMO-Agent/0.5.0",
        "Accept": "text/csv",
    })
    try:
        with urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8")
    except (URLError, HTTPError):
        # Try previous year if current year not yet available
        if year == date.today().year:
            return fetch_treasury_curve_xml(year - 1)
        raise

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        if year == date.today().year:
            return fetch_treasury_curve_xml(year - 1)
        return {"date": None, "yields": {}, "source": "Treasury.gov"}

    # Most recent row is first
    latest = rows[0]

    # Column mapping: Treasury.gov CSV column names -> maturity in years
    col_map = {
        "1 Mo": 1/12, "2 Mo": 2/12, "3 Mo": 3/12, "4 Mo": 4/12,
        "6 Mo": 6/12, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
        "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0,
        "30 Yr": 30.0,
    }

    yields = {}
    for col_name, maturity in col_map.items():
        val = latest.get(col_name, "")
        if val and val.strip() not in ("", "N/A"):
            try:
                yields[maturity] = float(val)
            except ValueError:
                continue

    curve_date = latest.get("Date", str(date.today()))

    return {
        "date": curve_date,
        "yields": yields,
        "source": "Treasury.gov",
        "series_count": len(yields),
    }


def fetch_treasury_curve_history(
    year: int,
    maturity_years: float = 10.0,
) -> list[dict]:
    """Fetch full year of Treasury yield data for a given maturity."""
    url = (
        f"https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/{year}/all"
        f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
    )
    req = Request(url, headers={"User-Agent": "CMO-Agent/0.5.0", "Accept": "text/csv"})
    with urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    # Find the column name for the requested maturity
    col_map = {
        1/12: "1 Mo", 2/12: "2 Mo", 3/12: "3 Mo", 4/12: "4 Mo",
        6/12: "6 Mo", 1.0: "1 Yr", 2.0: "2 Yr", 3.0: "3 Yr",
        5.0: "5 Yr", 7.0: "7 Yr", 10.0: "10 Yr", 20.0: "20 Yr",
        30.0: "30 Yr",
    }
    col_name = col_map.get(maturity_years, "10 Yr")

    reader = csv.DictReader(io.StringIO(text))
    results = []
    for row in reader:
        val = row.get(col_name, "")
        if val and val.strip() not in ("", "N/A"):
            try:
                results.append({
                    "date": row.get("Date", ""),
                    "yield_pct": float(val),
                    "maturity": maturity_years,
                })
            except ValueError:
                continue

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FHFA - House Price Index
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fhfa_hpi_national() -> list[dict]:
    """
    Fetch FHFA national house price index (quarterly, purchase-only).
    Falls back to FRED USSTHPI if direct FHFA download fails.
    """
    # Try FRED first (more reliable API)
    try:
        data = fred_fetch_series("USSTHPI", limit=200)
        if data:
            return [{"date": d["date"], "hpi": d["value"], "source": "FRED/FHFA"} for d in data]
    except Exception:
        pass

    # Direct FHFA CSV fallback
    try:
        url = f"{FHFA_HPI_BASE}/files/2025-01/2q24hpius_nsa.csv"
        req = Request(url, headers={"User-Agent": "CMO-Agent/0.5.0"})
        with urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        results = []
        for row in reader:
            try:
                results.append({
                    "year": int(row.get("yr", row.get("Year", 0))),
                    "quarter": int(row.get("qtr", row.get("Quarter", 0))),
                    "hpi": float(row.get("index_nsa", row.get("Index", 0))),
                    "source": "FHFA",
                })
            except (ValueError, KeyError):
                continue
        return results
    except Exception:
        return []


def fetch_fhfa_hpi_by_state() -> dict:
    """Fetch FHFA HPI by state (for geographic credit analysis).

    Tries FRED API first, falls back to local fhfa_hpi.json if available.
    """
    result = {}

    # Try FRED API if key is available
    if FRED_API_KEY:
        try:
            data = fred_fetch_series("USSTHPI", limit=1)
            states = {
                "CA": "CASTHPI", "FL": "FLSTHPI", "TX": "TXSTHPI",
                "NY": "NYSTHPI", "AZ": "AZSTHPI", "NV": "NVSTHPI",
                "IL": "ILSTHPI", "PA": "PASTHPI", "OH": "OHSTHPI",
                "GA": "GASTHPI", "NC": "NCSTHPI", "MI": "MISTHPI",
                "NJ": "NJSTHPI", "VA": "VASTHPI", "WA": "WASTHPI",
                "MA": "MASTHPI", "CO": "COSTHPI", "TN": "TNSTHPI",
                "MD": "MDSTHPI", "WI": "WISTHPI",
            }
            if data:
                result["national"] = data[0]
            for state, sid in states.items():
                try:
                    obs = fred_fetch_series(sid, limit=1)
                    if obs:
                        result[state] = obs[0]
                except Exception:
                    continue
            if result:
                return result
        except Exception:
            pass

    # Fallback: load from local fhfa_hpi.json if FRED unavailable
    import os, json
    hpi_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fhfa_hpi.json",
    )
    if os.path.exists(hpi_path):
        try:
            with open(hpi_path, 'r') as f:
                hpi_data = json.load(f)
            states_data = hpi_data.get("states", {})
            for state, info in states_data.items():
                result[state] = {
                    "value": info.get("hpi", 0),
                    "yoy_pct": info.get("yoy_pct", 0),
                    "period": info.get("period", ""),
                }
            return result
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Fannie Mae / Freddie Mac Loan-Level Data Framework
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LoanRecord:
    """Standardized loan record from GSE loan-level data."""
    loan_id: str
    origination_date: str
    original_balance: float
    current_balance: float
    original_rate: float
    current_rate: float
    original_term: int
    remaining_term: int
    ltv: float
    cltv: float
    fico: int
    dti: float
    state: str
    property_type: str
    occupancy: str
    channel: str  # R=Retail, C=Correspondent, B=Broker
    first_payment_date: str
    loan_purpose: str  # P=Purchase, R=Refi, C=Cash-out
    delinquency_status: str
    zero_balance_code: str  # 01=Prepaid, 03=Short Sale, 09=REO, etc.
    zero_balance_date: Optional[str] = None


@dataclass
class LoanLevelDataset:
    """Container for loan-level performance data."""
    source: str  # "FNMA" or "FHLMC"
    vintage: str  # e.g. "2020Q1"
    loans: list[LoanRecord] = field(default_factory=list)
    acquisition_count: int = 0
    performance_count: int = 0

    @property
    def prepayment_rate(self) -> float:
        """Fraction of loans that prepaid."""
        prepaid = sum(1 for l in self.loans if l.zero_balance_code == "01")
        return prepaid / len(self.loans) if self.loans else 0.0

    @property
    def default_rate(self) -> float:
        """Fraction of loans that defaulted (codes 03,09,15)."""
        defaulted = sum(
            1 for l in self.loans
            if l.zero_balance_code in ("03", "09", "15")
        )
        return defaulted / len(self.loans) if self.loans else 0.0

    @property
    def avg_fico(self) -> float:
        ficos = [l.fico for l in self.loans if l.fico > 0]
        return np.mean(ficos) if ficos else 0.0

    @property
    def avg_ltv(self) -> float:
        ltvs = [l.ltv for l in self.loans if l.ltv > 0]
        return np.mean(ltvs) if ltvs else 0.0

    def to_prepayment_vectors(self, max_age: int = 360) -> np.ndarray:
        """Convert loan-level data to monthly CPR vector for model calibration."""
        monthly_prepay = np.zeros(max_age)
        monthly_active = np.zeros(max_age)

        for loan in self.loans:
            age = loan.original_term - loan.remaining_term
            if 0 <= age < max_age:
                monthly_active[age] += 1
                if loan.zero_balance_code == "01":
                    monthly_prepay[age] += 1

        smm = np.where(monthly_active > 0, monthly_prepay / monthly_active, 0)
        cpr = 1 - (1 - smm) ** 12
        return cpr


def parse_fannie_acquisition(filepath: str) -> list[dict]:
    """
    Parse Fannie Mae Single-Family Loan Acquisition file.

    Download from: https://datadynamics.fanniemae.com/data-dynamics/#/reportMenu;702
    File format: pipe-delimited, no header. ~95M loans available free.

    Columns (30): POOL_ID|LOAN_ID|CHANNEL|SELLER|SERVICER|ORIG_RATE|ORIG_BALANCE|
    ORIG_TERM|ORIG_DATE|FIRST_PAY_DATE|ORIG_LTV|ORIG_CLTV|NUM_BORROWERS|DTI|
    FICO|FIRST_TIME_BUYER|PURPOSE|PROPERTY_TYPE|NUM_UNITS|OCCUPANCY|STATE|ZIP3|
    MI_PCT|PRODUCT_TYPE|CO_BORROWER_FICO|MI_TYPE|RELOCATION|...
    """
    records = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) < 25:
                    continue
                try:
                    records.append({
                        "loan_id": fields[1],
                        "channel": fields[2],
                        "orig_rate": float(fields[5]) if fields[5] else 0,
                        "orig_balance": float(fields[6]) if fields[6] else 0,
                        "orig_term": int(fields[7]) if fields[7] else 360,
                        "orig_date": fields[8],
                        "first_pay_date": fields[9],
                        "ltv": float(fields[10]) if fields[10] else 0,
                        "cltv": float(fields[11]) if fields[11] else 0,
                        "dti": float(fields[13]) if fields[13] else 0,
                        "fico": int(fields[14]) if fields[14] else 0,
                        "purpose": fields[16],
                        "property_type": fields[17],
                        "occupancy": fields[19],
                        "state": fields[20],
                    })
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return records


def parse_fannie_performance(filepath: str, max_rows: int = 1_000_000) -> list[dict]:
    """
    Parse Fannie Mae Single-Family Loan Performance file.

    Columns (31): LOAN_ID|REPORT_PERIOD|SERVICER|CURRENT_RATE|CURRENT_BALANCE|
    LOAN_AGE|REMAINING_MONTHS|MATURITY_DATE|MSA|DELINQUENCY_STATUS|MOD_FLAG|
    ZERO_BALANCE_CODE|ZERO_BALANCE_DATE|LAST_PAID_DATE|FORECLOSURE_DATE|
    DISPOSITION_DATE|FORECLOSURE_COSTS|PRESERVATION_COSTS|RECOVERY_COSTS|
    MISC_COSTS|TAX_COSTS|NET_SALE_PROCEEDS|CREDIT_ENHANCEMENT_PROCEEDS|
    REPURCHASE_PROCEEDS|OTHER_FORECLOSURE_PROCEEDS|NON_INT_BEARING_BALANCE|
    PRINCIPAL_FORGIVENESS|REPURCHASE_MAKE_WHOLE|...
    """
    records = []
    count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if count >= max_rows:
                    break
                fields = line.strip().split('|')
                if len(fields) < 12:
                    continue
                try:
                    records.append({
                        "loan_id": fields[0],
                        "report_period": fields[1],
                        "current_rate": float(fields[3]) if fields[3] else 0,
                        "current_balance": float(fields[4]) if fields[4] else 0,
                        "loan_age": int(fields[5]) if fields[5] else 0,
                        "remaining_months": int(fields[6]) if fields[6] else 0,
                        "delinquency_status": fields[9],
                        "zero_balance_code": fields[11],
                        "zero_balance_date": fields[12] if len(fields) > 12 else "",
                    })
                    count += 1
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return records


def build_historical_prepayment_curve(
    acquisition_file: str,
    performance_file: str,
    coupon_range: tuple[float, float] = (3.0, 7.0),
    vintage_year: Optional[int] = None,
) -> dict:
    """
    Build historical prepayment curve from Fannie Mae loan-level data.

    Returns monthly CPR/SMM vectors for model calibration, bucketed
    by WAC range and FICO band.
    """
    # Load acquisition data for filtering
    acq = parse_fannie_acquisition(acquisition_file)
    if vintage_year:
        acq = [r for r in acq if r["orig_date"][:4] == str(vintage_year)]
    acq = [r for r in acq if coupon_range[0] <= r["orig_rate"] <= coupon_range[1]]
    loan_ids = {r["loan_id"] for r in acq}
    loan_rates = {r["loan_id"]: r["orig_rate"] for r in acq}
    loan_ficos = {r["loan_id"]: r["fico"] for r in acq}

    # Load performance data
    perf = parse_fannie_performance(performance_file)
    perf = [r for r in perf if r["loan_id"] in loan_ids]

    # Build monthly prepayment vectors by WAC bucket
    buckets = {
        "low_coupon": (coupon_range[0], 4.0),
        "mid_coupon": (4.0, 5.5),
        "high_coupon": (5.5, coupon_range[1]),
    }

    results = {}
    for bucket_name, (lo, hi) in buckets.items():
        bucket_loans = {lid for lid, rate in loan_rates.items() if lo <= rate < hi}
        bucket_perf = [r for r in perf if r["loan_id"] in bucket_loans]

        # Monthly SMM computation
        max_age = 360
        active = np.zeros(max_age)
        prepaid = np.zeros(max_age)

        for r in bucket_perf:
            age = r["loan_age"]
            if 0 <= age < max_age:
                active[age] += 1
                if r["zero_balance_code"] == "01":
                    prepaid[age] += 1

        smm = np.where(active > 100, prepaid / active, 0)  # min 100 loans for stat significance
        cpr = 1 - (1 - smm) ** 12

        results[bucket_name] = {
            "coupon_range": (lo, hi),
            "loan_count": len(bucket_loans),
            "smm": smm.tolist(),
            "cpr": cpr.tolist(),
            "avg_fico": np.mean([loan_ficos[lid] for lid in bucket_loans if lid in loan_ficos]) if bucket_loans else 0,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Live Yield Curve Builder (integrates FRED + Treasury.gov + fallback)
# ═══════════════════════════════════════════════════════════════════════════

def build_live_treasury_curve(as_of: Optional[str] = None) -> dict:
    """
    Build Treasury yield curve from best available live source.
    Priority: FRED API > Treasury.gov CSV > hardcoded fallback.

    Returns dict compatible with yield_curve.build_curve_from_dict().
    """
    # Try FRED first (most granular, real-time)
    if FRED_API_KEY:
        try:
            result = fetch_treasury_curve_fred(as_of)
            if result["series_count"] >= 8:
                return result
        except Exception:
            pass

    # Try Treasury.gov CSV (no key needed)
    try:
        year = int(as_of[:4]) if as_of else None
        result = fetch_treasury_curve_xml(year)
        if result.get("yields") and len(result["yields"]) >= 8:
            return result
    except Exception:
        pass

    # Hardcoded fallback (Feb 2026)
    return {
        "date": "2026-02-07",
        "yields": {
            1/12: 3.699, 2/12: 3.701, 3/12: 3.690, 4/12: 3.684,
            6/12: 3.640, 1.0: 3.460, 2.0: 3.506, 3.0: 3.579,
            5.0: 3.769, 7.0: 3.988, 10.0: 4.221, 20.0: 4.812,
            30.0: 4.867,
        },
        "source": "hardcoded_fallback",
        "series_count": 13,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Mortgage Rate / Refi Index Integration
# ═══════════════════════════════════════════════════════════════════════════

def compute_refi_incentive(
    pool_wac: float,
    current_30yr_rate: Optional[float] = None,
) -> dict:
    """
    Compute refinancing incentive for a pool, using live rate data.

    Returns incentive in bps, estimated PSA speed, and rate source.
    """
    if current_30yr_rate is None:
        try:
            rates = fetch_current_mortgage_rates()
            current_30yr_rate = rates["30yr_fixed"]["rate"]
        except Exception:
            current_30yr_rate = 6.65  # fallback

    from .prepayment import estimate_psa_speed

    incentive = pool_wac - current_30yr_rate / 100.0
    psa = estimate_psa_speed(pool_wac, current_30yr_rate / 100.0, wala=12)

    return {
        "pool_wac": pool_wac,
        "current_rate": current_30yr_rate,
        "incentive_bps": round(incentive * 10000, 0),
        "estimated_psa": round(psa, 0),
        "in_the_money": incentive > 0.005,
        "deep_in_the_money": incentive > 0.015,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Market Data Aggregator
# ═══════════════════════════════════════════════════════════════════════════

def get_full_market_snapshot() -> dict:
    """
    Get comprehensive market data snapshot combining all sources.
    Used for daily portfolio marking and risk computation.
    """
    from . import fed_api

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "treasury_curve": {},
        "mortgage_rates": {},
        "fed_rates": {},
        "soma_holdings": {},
    }

    # Treasury curve
    try:
        curve_data = build_live_treasury_curve()
        snapshot["treasury_curve"] = curve_data
    except Exception as e:
        snapshot["treasury_curve"] = {"error": str(e)}

    # Mortgage rates
    try:
        snapshot["mortgage_rates"] = fetch_current_mortgage_rates()
    except Exception as e:
        snapshot["mortgage_rates"] = {"error": str(e)}

    # Fed rates (SOFR, EFFR)
    try:
        rates = fed_api.get_latest_rates()
        if "error" not in rates:
            snapshot["fed_rates"] = {
                "sofr": fed_api.extract_sofr_rate(rates),
                "effr": fed_api.extract_effr_rate(rates),
                "raw": rates,
            }
    except Exception as e:
        snapshot["fed_rates"] = {"error": str(e)}

    # SOMA MBS holdings
    try:
        soma = fed_api.get_soma_mbs_holdings()
        if "error" not in soma:
            snapshot["soma_holdings"] = soma
    except Exception as e:
        snapshot["soma_holdings"] = {"error": str(e)}

    return snapshot
