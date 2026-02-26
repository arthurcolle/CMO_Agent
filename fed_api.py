"""
NY Fed Markets Data API Client.
Provides live market data for SOFR, EFFR, AMBS operations, SOMA holdings, and Treasury operations.
"""
import json
from datetime import date, timedelta
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


BASE_URL = "https://markets.newyorkfed.org"


def _fetch_json(path: str) -> dict:
    """Fetch JSON from the NY Fed API."""
    url = f"{BASE_URL}{path}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError) as e:
        return {"error": str(e), "url": url}


# ─── Reference Rates (SOFR, EFFR, OBFR, BGCR, TGCR) ─────────────────────

def get_latest_rates() -> dict:
    """Get all latest reference rates (SOFR, EFFR, OBFR, etc.)."""
    return _fetch_json("/api/rates/all/latest.json")


def get_sofr_latest(n: int = 10) -> dict:
    """Get last n SOFR rates."""
    return _fetch_json(f"/api/rates/secured/sofr/last/{n}.json")


def get_sofr_averages_latest(n: int = 10) -> dict:
    """Get last n SOFR averages and index values."""
    return _fetch_json(f"/api/rates/secured/sofr-avg-ind/last/{n}.json")


def get_effr_latest(n: int = 10) -> dict:
    """Get last n Effective Federal Funds Rate values."""
    return _fetch_json(f"/api/rates/unsecured/effr/last/{n}.json")


def get_obfr_latest(n: int = 10) -> dict:
    """Get last n Overnight Bank Funding Rate values."""
    return _fetch_json(f"/api/rates/unsecured/obfr/last/{n}.json")


def search_rates(rate_type: str = "sofr", start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> dict:
    """Search rates by date range. rate_type: sofr, effr, obfr, bgcr, tgcr."""
    secured = rate_type in ("sofr", "bgcr", "tgcr", "sofr-avg-ind")
    category = "secured" if secured else "unsecured"
    path = f"/api/rates/{category}/{rate_type}/search.json"
    params = []
    if start_date:
        params.append(f"startDate={start_date}")
    if end_date:
        params.append(f"endDate={end_date}")
    if params:
        path += "?" + "&".join(params)
    return _fetch_json(path)


# ─── Agency MBS Operations ───────────────────────────────────────────────

def get_ambs_latest() -> dict:
    """Get latest Agency MBS operations (outrights, dollar rolls, coupon swaps)."""
    return _fetch_json("/api/ambs/all/latest/latest.json")


def get_ambs_results_last(n: int = 5) -> dict:
    """Get last n AMBS operation results with details."""
    return _fetch_json(f"/api/ambs/all/results/details/last/{n}.json")


def search_ambs(start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
    """Search AMBS operations by date range."""
    path = "/api/ambs/all/results/details/search.json"
    params = []
    if start_date:
        params.append(f"startDate={start_date}")
    if end_date:
        params.append(f"endDate={end_date}")
    if params:
        path += "?" + "&".join(params)
    return _fetch_json(path)


# ─── SOMA Holdings ───────────────────────────────────────────────────────

def get_soma_summary() -> dict:
    """Get SOMA holdings summary by security type."""
    return _fetch_json("/api/soma/summary.json")


def get_soma_agency_holdings(as_of_date: Optional[str] = None) -> dict:
    """Get SOMA agency securities holdings (MBS, Agency Debt, CMBS)."""
    if as_of_date:
        return _fetch_json(f"/api/soma/agency/get/asof/{as_of_date}.json")
    # Get latest as-of date first
    dates = _fetch_json("/api/soma/asofdates/latest.json")
    if "error" not in dates and "soma" in dates:
        latest = dates["soma"].get("asOfDate", "")
        if latest:
            return _fetch_json(f"/api/soma/agency/get/asof/{latest}.json")
    return dates


def get_soma_mbs_holdings(as_of_date: Optional[str] = None) -> dict:
    """Get SOMA MBS holdings specifically."""
    if as_of_date:
        return _fetch_json(f"/api/soma/agency/get/mbs/asof/{as_of_date}.json")
    dates = _fetch_json("/api/soma/asofdates/latest.json")
    if "error" not in dates and "soma" in dates:
        latest = dates["soma"].get("asOfDate", "")
        if latest:
            return _fetch_json(f"/api/soma/agency/get/mbs/asof/{latest}.json")
    return dates


def get_soma_treasury_holdings(as_of_date: Optional[str] = None) -> dict:
    """Get SOMA Treasury holdings."""
    if as_of_date:
        return _fetch_json(f"/api/soma/tsy/get/asof/{as_of_date}.json")
    dates = _fetch_json("/api/soma/asofdates/latest.json")
    if "error" not in dates and "soma" in dates:
        latest = dates["soma"].get("asOfDate", "")
        if latest:
            return _fetch_json(f"/api/soma/tsy/get/asof/{latest}.json")
    return dates


def get_soma_monthly_treasury() -> dict:
    """Get monthly summary of SOMA Treasury holdings."""
    return _fetch_json("/api/soma/tsy/get/monthly.json")


# ─── Repo/Reverse Repo Operations ───────────────────────────────────────

def get_repo_latest() -> dict:
    """Get latest repo operations."""
    return _fetch_json("/api/rp/repo/all/all/latest.json")


def get_reverse_repo_latest() -> dict:
    """Get latest reverse repo operations."""
    return _fetch_json("/api/rp/reverserepo/all/all/latest.json")


def get_reverse_repo_last(n: int = 10) -> dict:
    """Get last n reverse repo operation results."""
    return _fetch_json(f"/api/rp/reverserepo/all/results/last/{n}.json")


# ─── Treasury Securities Operations ─────────────────────────────────────

def get_treasury_ops_latest() -> dict:
    """Get latest Treasury securities operations."""
    return _fetch_json("/api/tsy/all/all/all/latest.json")


def get_treasury_ops_last(n: int = 5) -> dict:
    """Get last n Treasury operations results."""
    return _fetch_json(f"/api/tsy/all/results/details/last/{n}.json")


# ─── Primary Dealer Statistics ───────────────────────────────────────────

def get_primary_dealer_latest() -> dict:
    """Get latest primary dealer statistics survey."""
    return _fetch_json("/api/pd/latest/SBN2022.json")


def get_primary_dealer_market_share() -> dict:
    """Get latest quarterly primary dealer market share data."""
    return _fetch_json("/api/marketshare/qtrly/latest.json")


# ─── Securities Lending ─────────────────────────────────────────────────

def get_sec_lending_latest() -> dict:
    """Get latest securities lending operations."""
    return _fetch_json("/api/seclending/all/results/summary/latest.json")


# ─── Guide Sheets ───────────────────────────────────────────────────────

def get_guide_sheets_latest() -> dict:
    """Get latest guide sheets (FR 2004SI, WI, F-series)."""
    return _fetch_json("/api/guidesheets/si/latest.json")


# ─── Convenience Functions ──────────────────────────────────────────────

def get_market_data_snapshot() -> dict:
    """
    Get a comprehensive market data snapshot from the NY Fed.
    Returns SOFR, EFFR, SOMA summary, and latest AMBS operations.
    """
    snapshot = {
        "rates": {},
        "soma": {},
        "ambs": {},
        "repo": {},
    }

    # Reference rates
    rates = get_latest_rates()
    if "error" not in rates:
        snapshot["rates"] = rates

    # SOMA summary
    soma = get_soma_summary()
    if "error" not in soma:
        snapshot["soma"] = soma

    return snapshot


def extract_sofr_rate(data: dict) -> Optional[float]:
    """Extract the SOFR rate from API response."""
    try:
        if "refRates" in data:
            for r in data["refRates"]:
                if r.get("type") == "SOFR":
                    return float(r.get("percentRate", 0))
        if "rates" in data:
            for r in data["rates"]:
                if r.get("type") == "SOFR":
                    return float(r.get("percentRate", 0))
    except (KeyError, ValueError, TypeError):
        pass
    return None


def extract_effr_rate(data: dict) -> Optional[float]:
    """Extract the EFFR rate from API response."""
    try:
        if "refRates" in data:
            for r in data["refRates"]:
                if r.get("type") == "EFFR":
                    return float(r.get("percentRate", 0))
    except (KeyError, ValueError, TypeError):
        pass
    return None
