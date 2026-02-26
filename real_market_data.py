"""
Real Market Data Provider — Historical market scenarios from FRED + Treasury.gov.

Replaces synthetic MarketSimulator scenarios with actual historical data.
SQLite cache at data/market_history.db stores ~5,000 business days (2006-present).

Data sources:
- FRED API: 19 series (rates, money supply, housing, credit, macro)
- Treasury.gov CSV: Daily par yield curve (13 tenors)
- NY Fed: SOFR (post-2018)

Regime classification from actual data instead of random assignment.
"""
import os
import sqlite3
import time
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from .yield_curve import YieldCurve, YieldCurvePoint
from .spec_pool import SpecPool, AgencyType, CollateralType
from .tba import build_tba_price_grid
from .market_simulator import MarketScenario


# ═══════════════════════════════════════════════════════════════════════
# Historical Market Record
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HistoricalMarketRecord:
    """One business day's complete market data."""
    date: str  # YYYY-MM-DD

    # Treasury curve (13 tenors)
    dgs1mo: float = np.nan
    dgs3mo: float = np.nan
    dgs6mo: float = np.nan
    dgs1: float = np.nan
    dgs2: float = np.nan
    dgs3: float = np.nan
    dgs5: float = np.nan
    dgs7: float = np.nan
    dgs10: float = np.nan
    dgs20: float = np.nan
    dgs30: float = np.nan

    # Policy rates
    fed_funds: float = np.nan
    sofr: float = np.nan

    # Mortgage rates
    mortgage_30yr: float = np.nan
    mortgage_15yr: float = np.nan

    # Money supply / Fed balance sheet
    m2: float = np.nan              # M2 money supply ($B)
    fed_balance_sheet: float = np.nan  # WALCL ($M)
    rrp: float = np.nan             # Reverse repo (RRPONTSYD, $B)
    tga: float = np.nan             # Treasury General Account (WTREGEN, $M)

    # Housing
    housing_starts: float = np.nan   # HOUST (thousands, SAAR)
    building_permits: float = np.nan # PERMIT
    case_shiller: float = np.nan     # CSUSHPINSA (index)
    months_supply: float = np.nan    # MSACSR
    # HPI YoY computed as derived

    # Credit spreads
    hy_oas: float = np.nan           # BAMLH0A0HYM2 (bps)
    ig_oas: float = np.nan           # BAMLC0A0CM (bps)

    # MBS
    mbs_bank_holdings: float = np.nan  # WSHOMCB ($B)

    # Macro
    unemployment: float = np.nan     # UNRATE (%)
    cpi: float = np.nan              # CPIAUCSL (index)

    # Spreads (FRED pre-computed)
    t10y2y: float = np.nan
    t10y3m: float = np.nan

    def curve_tenors_yields(self) -> list[tuple[float, float]]:
        """Return (tenor, yield) pairs for non-NaN tenors."""
        mapping = [
            (1/12, self.dgs1mo), (3/12, self.dgs3mo), (6/12, self.dgs6mo),
            (1.0, self.dgs1), (2.0, self.dgs2), (3.0, self.dgs3),
            (5.0, self.dgs5), (7.0, self.dgs7), (10.0, self.dgs10),
            (20.0, self.dgs20), (30.0, self.dgs30),
        ]
        return [(t, y) for t, y in mapping if not np.isnan(y)]


# ═══════════════════════════════════════════════════════════════════════
# FRED Series Configuration
# ═══════════════════════════════════════════════════════════════════════

FRED_SERIES = {
    # Treasury yields (daily)
    "DGS1MO": "dgs1mo", "DGS3MO": "dgs3mo", "DGS6MO": "dgs6mo",
    "DGS1": "dgs1", "DGS2": "dgs2", "DGS3": "dgs3",
    "DGS5": "dgs5", "DGS7": "dgs7", "DGS10": "dgs10",
    "DGS20": "dgs20", "DGS30": "dgs30",
    # Rates
    "FEDFUNDS": "fed_funds",
    # Mortgage (weekly Thursdays)
    "MORTGAGE30US": "mortgage_30yr", "MORTGAGE15US": "mortgage_15yr",
    # Money supply / Fed
    "M2SL": "m2", "WALCL": "fed_balance_sheet",
    "RRPONTSYD": "rrp", "WTREGEN": "tga",
    # Housing (monthly)
    "HOUST": "housing_starts", "PERMIT": "building_permits",
    "CSUSHPINSA": "case_shiller", "MSACSR": "months_supply",
    # Credit (daily)
    "BAMLH0A0HYM2": "hy_oas", "BAMLC0A0CM": "ig_oas",
    # MBS (weekly)
    "WSHOMCB": "mbs_bank_holdings",
    # Macro (monthly)
    "UNRATE": "unemployment", "CPIAUCSL": "cpi",
    # Spreads (daily)
    "T10Y2Y": "t10y2y", "T10Y3M": "t10y3m",
}

# Series available from different start dates
_SERIES_START_OVERRIDE = {
    "RRPONTSYD": "2013-09-23",  # RRP facility started late 2013
    "WTREGEN": "2015-10-01",    # TGA data starts ~2015
    "BAMLH0A0HYM2": "1996-12-31",
    "BAMLC0A0CM": "1996-12-31",
}

DB_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "market_history.db"
)


# ═══════════════════════════════════════════════════════════════════════
# SQLite Schema & Helpers
# ═══════════════════════════════════════════════════════════════════════

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS daily_market (
    date TEXT PRIMARY KEY,
    dgs1mo REAL, dgs3mo REAL, dgs6mo REAL,
    dgs1 REAL, dgs2 REAL, dgs3 REAL,
    dgs5 REAL, dgs7 REAL, dgs10 REAL,
    dgs20 REAL, dgs30 REAL,
    fed_funds REAL, sofr REAL,
    mortgage_30yr REAL, mortgage_15yr REAL,
    m2 REAL, fed_balance_sheet REAL, rrp REAL, tga REAL,
    housing_starts REAL, building_permits REAL,
    case_shiller REAL, months_supply REAL,
    hy_oas REAL, ig_oas REAL,
    mbs_bank_holdings REAL,
    unemployment REAL, cpi REAL,
    t10y2y REAL, t10y3m REAL
);
"""

_CREATE_META = """
CREATE TABLE IF NOT EXISTS fetch_meta (
    series_id TEXT PRIMARY KEY,
    last_date TEXT,
    record_count INTEGER,
    updated_at TEXT
);
"""

_COLUMNS = [
    "date", "dgs1mo", "dgs3mo", "dgs6mo", "dgs1", "dgs2", "dgs3",
    "dgs5", "dgs7", "dgs10", "dgs20", "dgs30",
    "fed_funds", "sofr",
    "mortgage_30yr", "mortgage_15yr",
    "m2", "fed_balance_sheet", "rrp", "tga",
    "housing_starts", "building_permits", "case_shiller", "months_supply",
    "hy_oas", "ig_oas", "mbs_bank_holdings",
    "unemployment", "cpi", "t10y2y", "t10y3m",
]


def _record_to_row(rec: HistoricalMarketRecord) -> tuple:
    """Convert record to SQLite row tuple."""
    return tuple(
        getattr(rec, col) if col != "date" else rec.date
        for col in _COLUMNS
    )


def _row_to_record(row: tuple) -> HistoricalMarketRecord:
    """Convert SQLite row to record."""
    d = {}
    for i, col in enumerate(_COLUMNS):
        val = row[i]
        if col == "date":
            d[col] = val
        else:
            d[col] = float(val) if val is not None else np.nan
    return HistoricalMarketRecord(**d)


# ═══════════════════════════════════════════════════════════════════════
# Real Market Data Provider
# ═══════════════════════════════════════════════════════════════════════

class RealMarketDataProvider:
    """Provides historical market scenarios from SQLite cache.

    Usage:
        provider = RealMarketDataProvider()
        provider.build_database()  # one-time, ~5 min
        scenario = provider.get_scenario_for_date("2020-03-16")
    """

    def __init__(self, db_path: str = DB_PATH_DEFAULT, seed: Optional[int] = None):
        self.db_path = db_path
        self.rng = np.random.RandomState(seed)
        self._available_dates: Optional[list[str]] = None
        self._ensure_db()

    def _ensure_db(self):
        """Create tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_META)

    @property
    def available_dates(self) -> list[str]:
        """Get sorted list of available business days."""
        if self._available_dates is None:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT date FROM daily_market WHERE dgs10 IS NOT NULL ORDER BY date"
                ).fetchall()
                self._available_dates = [r[0] for r in rows]
        return self._available_dates

    @property
    def n_dates(self) -> int:
        return len(self.available_dates)

    def get_random_historical_date(self, rng: Optional[np.random.RandomState] = None) -> str:
        """Pick a random date from available business days."""
        r = rng or self.rng
        dates = self.available_dates
        if not dates:
            raise RuntimeError("No data in database. Run build_database() first.")
        return dates[r.randint(len(dates))]

    # ─── Database Building ────────────────────────────────────────────

    def build_database(
        self,
        start_date: str = "2006-01-01",
        end_date: Optional[str] = None,
        incremental: bool = True,
    ):
        """Bulk-fetch all FRED series and store in SQLite.

        Args:
            start_date: Earliest date to fetch.
            end_date: Latest date (default: today).
            incremental: If True, only fetch data newer than last cached date.
        """
        from .data_sources import fred_fetch_series_bulk

        if end_date is None:
            end_date = str(date.today())

        print(f"Building market database: {start_date} to {end_date}")
        print(f"  Database: {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            # Check existing data for incremental update
            if incremental:
                row = conn.execute(
                    "SELECT MAX(date) FROM daily_market WHERE dgs10 IS NOT NULL"
                ).fetchone()
                if row and row[0]:
                    last_date = row[0]
                    # Fetch from day after last cached date
                    fetch_start = str(
                        datetime.strptime(last_date, "%Y-%m-%d").date() + timedelta(days=1)
                    )
                    if fetch_start > end_date:
                        print(f"  Database already up to date (last: {last_date})")
                        return
                    print(f"  Incremental update from {fetch_start} (last cached: {last_date})")
                    start_date = fetch_start

            # Fetch all series
            all_data: dict[str, list[dict]] = {}
            series_list = list(FRED_SERIES.keys())

            print(f"  Fetching {len(series_list)} FRED series...")
            fetched = fred_fetch_series_bulk(
                series_list, start_date=start_date, end_date=end_date
            )
            all_data.update(fetched)

            # Also fetch SOFR from NY Fed (separate source)
            sofr_data = self._fetch_sofr_history(start_date, end_date)

            # Build date → record mapping
            records: dict[str, HistoricalMarketRecord] = {}

            # First pass: populate from FRED series
            for series_id, observations in all_data.items():
                field_name = FRED_SERIES.get(series_id)
                if not field_name:
                    continue
                for obs in observations:
                    d = obs["date"]
                    if d not in records:
                        records[d] = HistoricalMarketRecord(date=d)
                    setattr(records[d], field_name, obs["value"])

            # SOFR overlay
            for obs in sofr_data:
                d = obs["date"]
                if d not in records:
                    records[d] = HistoricalMarketRecord(date=d)
                records[d].sofr = obs["value"]

            # Forward-fill monthly/weekly series to daily
            if records:
                self._forward_fill_records(records)

            # Write to SQLite
            n_written = 0
            placeholders = ", ".join(["?"] * len(_COLUMNS))
            col_names = ", ".join(_COLUMNS)
            for d, rec in sorted(records.items()):
                # Skip if no curve data
                if np.isnan(rec.dgs10):
                    continue
                row = _record_to_row(rec)
                conn.execute(
                    f"INSERT OR REPLACE INTO daily_market ({col_names}) VALUES ({placeholders})",
                    row,
                )
                n_written += 1

            # Update metadata
            for series_id in series_list:
                obs_list = all_data.get(series_id, [])
                last = obs_list[0]["date"] if obs_list else ""
                conn.execute(
                    "INSERT OR REPLACE INTO fetch_meta (series_id, last_date, record_count, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (series_id, last, len(obs_list), str(datetime.now())),
                )

            conn.commit()

        # Reset cache
        self._available_dates = None
        print(f"  Written {n_written} daily records. Total available: {self.n_dates}")

    def _fetch_sofr_history(self, start_date: str, end_date: str) -> list[dict]:
        """Fetch SOFR history from NY Fed API."""
        try:
            from .fed_api import _fetch_json
            # NY Fed SOFR search endpoint
            result = _fetch_json(
                f"/api/rates/secured/sofr/search.json"
                f"?startDate={start_date.replace('-', '')}"
                f"&endDate={end_date.replace('-', '')}"
            )
            if "error" in result:
                # Fallback: use FRED EFFR as proxy for pre-SOFR period
                return []
            rates = result.get("refRates", [])
            return [
                {"date": r["effectiveDate"][:10], "value": float(r["percentRate"])}
                for r in rates
                if r.get("percentRate") is not None
            ]
        except Exception:
            return []

    def _forward_fill_records(self, records: dict[str, HistoricalMarketRecord]):
        """Forward-fill weekly/monthly series to daily granularity."""
        # Fields that are weekly or monthly (need forward fill)
        fill_fields = [
            "fed_funds", "mortgage_30yr", "mortgage_15yr",
            "m2", "fed_balance_sheet", "rrp", "tga",
            "housing_starts", "building_permits", "case_shiller", "months_supply",
            "mbs_bank_holdings", "unemployment", "cpi",
        ]

        sorted_dates = sorted(records.keys())
        prev_values: dict[str, float] = {}

        for d in sorted_dates:
            rec = records[d]
            for f in fill_fields:
                val = getattr(rec, f)
                if np.isnan(val):
                    if f in prev_values:
                        setattr(rec, f, prev_values[f])
                else:
                    prev_values[f] = val

            # Pre-SOFR: use fed_funds as proxy
            if np.isnan(rec.sofr) and not np.isnan(rec.fed_funds):
                rec.sofr = rec.fed_funds

            # Pre-2013 RRP/TGA: set to 0
            if np.isnan(rec.rrp):
                rec.rrp = 0.0
            if np.isnan(rec.tga):
                rec.tga = 0.0

    # ─── Scenario Generation ──────────────────────────────────────────

    def get_record_for_date(self, target_date: str) -> Optional[HistoricalMarketRecord]:
        """Get the historical record for a specific date."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM daily_market WHERE date = ?",
                (target_date,)
            ).fetchone()
            if row:
                return _row_to_record(row)

            # Try nearest prior business day (up to 5 days back)
            for i in range(1, 6):
                prior = str(
                    datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=i)
                )
                row = conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM daily_market WHERE date = ?",
                    (prior,)
                ).fetchone()
                if row:
                    return _row_to_record(row)
        return None

    def get_trailing_records(
        self, target_date: str, n_days: int = 63
    ) -> list[HistoricalMarketRecord]:
        """Get trailing N business days of records for derived calculations."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM daily_market "
                f"WHERE date <= ? AND dgs10 IS NOT NULL "
                f"ORDER BY date DESC LIMIT ?",
                (target_date, n_days)
            ).fetchall()
            return [_row_to_record(r) for r in rows]

    def classify_regime(
        self, record: HistoricalMarketRecord,
        trailing: Optional[list[HistoricalMarketRecord]] = None,
    ) -> str:
        """Classify market regime from actual data.

        Rules:
        - 10Y-2Y > 1.5 + short < 3% => steep
        - 10Y-2Y < 0 => inverted
        - |10Y-2Y| < 0.3 => flat
        - FF decreasing trailing 3mo => easing
        - FF increasing trailing 3mo => tightening
        - HY OAS > 500bps => crisis
        - else => normal
        """
        spread_10y2y = record.t10y2y if not np.isnan(record.t10y2y) else (
            record.dgs10 - record.dgs2 if not (np.isnan(record.dgs10) or np.isnan(record.dgs2)) else 0.0
        )
        short_rate = record.dgs3mo if not np.isnan(record.dgs3mo) else record.fed_funds

        # Crisis check first (highest priority)
        if not np.isnan(record.hy_oas) and record.hy_oas > 500:
            return "crisis"

        # Inverted
        if spread_10y2y < 0:
            return "inverted"

        # Steep
        if spread_10y2y > 1.5 and (not np.isnan(short_rate) and short_rate < 3.0):
            return "steep"

        # Flat
        if abs(spread_10y2y) < 0.3:
            return "flat"

        # Easing/tightening from trailing fed funds
        if trailing and len(trailing) >= 63:
            recent_ff = [r.fed_funds for r in trailing[:21] if not np.isnan(r.fed_funds)]
            older_ff = [r.fed_funds for r in trailing[42:63] if not np.isnan(r.fed_funds)]
            if recent_ff and older_ff:
                avg_recent = np.mean(recent_ff)
                avg_older = np.mean(older_ff)
                if avg_recent < avg_older - 0.25:
                    return "easing"
                if avg_recent > avg_older + 0.25:
                    return "tightening"

        return "normal"

    def compute_derived_fields(
        self, record: HistoricalMarketRecord,
        trailing: list[HistoricalMarketRecord],
    ) -> dict:
        """Compute derived ecosystem fields from trailing data.

        Returns dict with:
        - rate changes (1w, 1m, 3m for 2Y, 10Y, 30Y)
        - realized vol (10Y, 20-day)
        - curve slopes (2s10s, 5s30s, 2s5s10s butterfly)
        - mortgage-treasury spread
        - HPI YoY
        - CPI YoY
        - M2 growth rate
        - Fed balance sheet change
        """
        derived = {}

        # Trailing rate changes
        for tenor_name, field_name in [("2y", "dgs2"), ("10y", "dgs10"), ("30y", "dgs30")]:
            current = getattr(record, field_name)
            if np.isnan(current):
                derived[f"{tenor_name}_chg_1w"] = 0.0
                derived[f"{tenor_name}_chg_1m"] = 0.0
                derived[f"{tenor_name}_chg_3m"] = 0.0
                continue
            # 1 week (~5 biz days)
            if len(trailing) > 5:
                prev = getattr(trailing[5], field_name)
                derived[f"{tenor_name}_chg_1w"] = (current - prev) if not np.isnan(prev) else 0.0
            else:
                derived[f"{tenor_name}_chg_1w"] = 0.0
            # 1 month (~21 biz days)
            if len(trailing) > 21:
                prev = getattr(trailing[21], field_name)
                derived[f"{tenor_name}_chg_1m"] = (current - prev) if not np.isnan(prev) else 0.0
            else:
                derived[f"{tenor_name}_chg_1m"] = 0.0
            # 3 months (~63 biz days)
            if len(trailing) > 62:
                prev = getattr(trailing[62], field_name)
                derived[f"{tenor_name}_chg_3m"] = (current - prev) if not np.isnan(prev) else 0.0
            else:
                derived[f"{tenor_name}_chg_3m"] = 0.0

        # Realized vol (10Y, 20-day)
        ten_y_values = [getattr(r, "dgs10") for r in trailing[:20] if not np.isnan(getattr(r, "dgs10"))]
        if len(ten_y_values) >= 10:
            changes = np.diff(ten_y_values)
            derived["realized_vol_10y"] = float(np.std(changes) * np.sqrt(252) * 100)  # annualized bps
        else:
            derived["realized_vol_10y"] = 80.0  # default

        # Curve slopes
        if not np.isnan(record.dgs10) and not np.isnan(record.dgs2):
            derived["slope_2s10s"] = record.dgs10 - record.dgs2
        else:
            derived["slope_2s10s"] = 0.0

        if not np.isnan(record.dgs30) and not np.isnan(record.dgs5):
            derived["slope_5s30s"] = record.dgs30 - record.dgs5
        else:
            derived["slope_5s30s"] = 0.0

        # Butterfly: 2 * 5Y - 2Y - 10Y
        if not (np.isnan(record.dgs2) or np.isnan(record.dgs5) or np.isnan(record.dgs10)):
            derived["butterfly_2s5s10s"] = 2 * record.dgs5 - record.dgs2 - record.dgs10
        else:
            derived["butterfly_2s5s10s"] = 0.0

        # Mortgage-Treasury spread
        if not np.isnan(record.mortgage_30yr) and not np.isnan(record.dgs10):
            derived["mtg_tsy_spread"] = record.mortgage_30yr - record.dgs10
        else:
            derived["mtg_tsy_spread"] = 1.7  # historical average

        # HPI YoY (Case-Shiller, monthly, ~12 months lag = ~252 biz days)
        if not np.isnan(record.case_shiller) and len(trailing) > 250:
            prior_cs = [r.case_shiller for r in trailing[240:260] if not np.isnan(r.case_shiller)]
            if prior_cs:
                derived["hpi_yoy"] = (record.case_shiller / np.mean(prior_cs) - 1.0) * 100
            else:
                derived["hpi_yoy"] = 0.0
        else:
            derived["hpi_yoy"] = 0.0

        # CPI YoY
        if not np.isnan(record.cpi) and len(trailing) > 250:
            prior_cpi = [r.cpi for r in trailing[240:260] if not np.isnan(r.cpi)]
            if prior_cpi:
                derived["cpi_yoy"] = (record.cpi / np.mean(prior_cpi) - 1.0) * 100
            else:
                derived["cpi_yoy"] = 0.0
        else:
            derived["cpi_yoy"] = 0.0

        # M2 growth (YoY)
        if not np.isnan(record.m2) and len(trailing) > 250:
            prior_m2 = [r.m2 for r in trailing[240:260] if not np.isnan(r.m2)]
            if prior_m2:
                derived["m2_growth_yoy"] = (record.m2 / np.mean(prior_m2) - 1.0) * 100
            else:
                derived["m2_growth_yoy"] = 0.0
        else:
            derived["m2_growth_yoy"] = 0.0

        # Fed balance sheet change (3mo)
        if not np.isnan(record.fed_balance_sheet) and len(trailing) > 62:
            prior_bs = [r.fed_balance_sheet for r in trailing[58:66] if not np.isnan(r.fed_balance_sheet)]
            if prior_bs:
                derived["fed_bs_chg_3m"] = (record.fed_balance_sheet - np.mean(prior_bs)) / 1e6  # trillions
            else:
                derived["fed_bs_chg_3m"] = 0.0
        else:
            derived["fed_bs_chg_3m"] = 0.0

        return derived

    def get_scenario_for_date(
        self,
        target_date: str,
        collateral_balance: float = 100_000_000,
        rng: Optional[np.random.RandomState] = None,
    ) -> Optional[MarketScenario]:
        """Convert historical data to a MarketScenario.

        Args:
            target_date: YYYY-MM-DD
            collateral_balance: For pool generation
            rng: Random state for pool characteristics

        Returns:
            MarketScenario populated from real data, or None if date unavailable.
        """
        r = rng or self.rng
        record = self.get_record_for_date(target_date)
        if record is None:
            return None

        trailing = self.get_trailing_records(target_date, n_days=265)
        derived = self.compute_derived_fields(record, trailing)
        regime = self.classify_regime(record, trailing)

        # Build yield curve
        tenors_yields = record.curve_tenors_yields()
        if not tenors_yields:
            return None

        from datetime import date as datemod
        try:
            curve_date = datemod.fromisoformat(record.date)
        except (ValueError, TypeError):
            curve_date = datemod.today()

        points = [YieldCurvePoint(t, y) for t, y in tenors_yields]
        curve = YieldCurve(as_of_date=curve_date, points=points)

        # Key rates
        t10y = record.dgs10 if not np.isnan(record.dgs10) else 4.0
        mortgage_rate = record.mortgage_30yr if not np.isnan(record.mortgage_30yr) else t10y + 1.7
        sofr = record.sofr if not np.isnan(record.sofr) else record.fed_funds
        if np.isnan(sofr):
            sofr = t10y - 1.0
        ff = record.fed_funds if not np.isnan(record.fed_funds) else sofr

        # Build TBA grid
        tba_grid = build_tba_price_grid(t10y, "FNMA15", 30)

        # Generate collateral pools calibrated to actual mortgage rate
        pools = self._generate_calibrated_collateral(mortgage_rate, regime, collateral_balance, r)

        # Rate vol from realized data
        rate_vol = derived.get("realized_vol_10y", 80.0)

        # Financing rate
        financing_rate = sofr + r.uniform(-0.05, 0.15)

        # Song & Zhu desk state (still partially randomized but anchored to real data)
        from .market_simulator import MarketSimulator
        sim = MarketSimulator(seed=int(r.randint(1_000_000)))
        dealer_leverage, cpr_disp, cpr_signed, fed_roll, specialness = \
            sim._generate_desk_state(regime, mortgage_rate, rate_vol)

        scenario = MarketScenario(
            curve=curve,
            treasury_10y=t10y,
            mortgage_rate=mortgage_rate,
            sofr=sofr,
            fed_funds=ff,
            tba_grid=tba_grid,
            rate_vol_bps=rate_vol,
            collateral_pools=pools,
            deal_mode="AGENCY",
            financing_rate=financing_rate,
            dealer_leverage=dealer_leverage,
            cpr_dispersion=cpr_disp,
            cpr_signed_change=cpr_signed,
            fed_roll_indicator=fed_roll,
            dollar_roll_specialness=specialness,
            regime=regime,
            scenario_id=f"HIST_{record.date}",
            # Ecosystem fields
            m2=record.m2 if not np.isnan(record.m2) else 0.0,
            fed_balance_sheet=record.fed_balance_sheet if not np.isnan(record.fed_balance_sheet) else 0.0,
            rrp=record.rrp,
            tga=record.tga,
            housing_starts=record.housing_starts if not np.isnan(record.housing_starts) else 0.0,
            building_permits=record.building_permits if not np.isnan(record.building_permits) else 0.0,
            case_shiller=record.case_shiller if not np.isnan(record.case_shiller) else 0.0,
            months_supply=record.months_supply if not np.isnan(record.months_supply) else 0.0,
            hpi_yoy=derived.get("hpi_yoy", 0.0),
            hy_oas=record.hy_oas if not np.isnan(record.hy_oas) else 0.0,
            ig_oas=record.ig_oas if not np.isnan(record.ig_oas) else 0.0,
            mbs_bank_holdings=record.mbs_bank_holdings if not np.isnan(record.mbs_bank_holdings) else 0.0,
            unemployment=record.unemployment if not np.isnan(record.unemployment) else 0.0,
            cpi_yoy=derived.get("cpi_yoy", 0.0),
            mtg_tsy_spread=derived.get("mtg_tsy_spread", 1.7),
            rate_chg_2y_1w=derived.get("2y_chg_1w", 0.0),
            rate_chg_2y_1m=derived.get("2y_chg_1m", 0.0),
            rate_chg_2y_3m=derived.get("2y_chg_3m", 0.0),
            rate_chg_10y_1w=derived.get("10y_chg_1w", 0.0),
            rate_chg_10y_1m=derived.get("10y_chg_1m", 0.0),
            rate_chg_10y_3m=derived.get("10y_chg_3m", 0.0),
            rate_chg_30y_1w=derived.get("30y_chg_1w", 0.0),
            rate_chg_30y_1m=derived.get("30y_chg_1m", 0.0),
            rate_chg_30y_3m=derived.get("30y_chg_3m", 0.0),
            realized_vol_10y=derived.get("realized_vol_10y", 80.0),
            slope_2s10s=derived.get("slope_2s10s", 0.0),
            slope_5s30s=derived.get("slope_5s30s", 0.0),
            butterfly_2s5s10s=derived.get("butterfly_2s5s10s", 0.0),
            m2_growth_yoy=derived.get("m2_growth_yoy", 0.0),
            fed_bs_chg_3m=derived.get("fed_bs_chg_3m", 0.0),
        )

        return scenario

    def _generate_calibrated_collateral(
        self,
        mortgage_rate: float,
        regime: str,
        collateral_balance: float,
        rng: np.random.RandomState,
    ) -> list[SpecPool]:
        """Generate collateral pools calibrated to actual date's mortgage rate."""
        n_pools = rng.randint(1, 4)
        pools = []

        for i in range(n_pools):
            coupon = round(mortgage_rate - rng.uniform(0.3, 1.0), 1)
            coupon = round(coupon * 2) / 2
            coupon = max(2.0, min(8.0, coupon))

            wac = coupon + rng.uniform(0.3, 0.7)
            wam = int(rng.choice([357, 356, 355, 354, 350, 345, 340, 180, 178, 176]))
            wala = 360 - wam if wam <= 360 else 0

            balance = collateral_balance
            agency = rng.choice([AgencyType.FNMA, AgencyType.FHLMC, AgencyType.GNMA])
            coll_type = {
                AgencyType.FNMA: CollateralType.FN,
                AgencyType.FHLMC: CollateralType.FH,
                AgencyType.GNMA: CollateralType.G2,
            }[agency]

            # FICO/LTV vary by regime
            if regime == "crisis":
                avg_fico = rng.uniform(640, 720)
                avg_ltv = rng.uniform(80, 100)
            else:
                avg_fico = rng.uniform(700, 780)
                avg_ltv = rng.uniform(60, 90)

            pool = SpecPool(
                pool_id=f"HIST_POOL_{i+1}",
                agency=agency,
                collateral_type=coll_type,
                coupon=coupon,
                wac=round(wac, 3),
                wam=wam,
                wala=wala,
                original_balance=balance,
                current_balance=balance,
                original_term=360 if wam > 200 else 180,
                avg_fico=round(avg_fico),
                avg_ltv=round(avg_ltv, 1),
                avg_loan_size=round(rng.uniform(150000, 500000)),
            )
            pools.append(pool)

        return pools

    # ─── Utilities ────────────────────────────────────────────────────

    def get_date_range(self) -> tuple[str, str]:
        """Get first and last available dates."""
        dates = self.available_dates
        if not dates:
            return ("", "")
        return (dates[0], dates[-1])

    def summary(self) -> dict:
        """Get summary statistics of the database."""
        dates = self.available_dates
        if not dates:
            return {"n_dates": 0, "status": "empty"}

        with sqlite3.connect(self.db_path) as conn:
            # Count non-null fields
            counts = {}
            for col in _COLUMNS[1:]:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM daily_market WHERE {col} IS NOT NULL"
                ).fetchone()
                counts[col] = row[0] if row else 0

        return {
            "n_dates": len(dates),
            "date_range": (dates[0], dates[-1]),
            "field_coverage": counts,
            "db_path": self.db_path,
        }
