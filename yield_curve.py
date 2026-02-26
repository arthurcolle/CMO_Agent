"""
US Treasury Yield Curve modeling and interpolation.
Supports bootstrapping, cubic spline interpolation, and Nelson-Siegel fitting.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class YieldCurvePoint:
    maturity_years: float
    yield_pct: float  # in percent, e.g. 4.221 for 4.221%


@dataclass
class YieldCurve:
    """Represents a yield curve with interpolation capabilities."""
    as_of_date: date
    points: list[YieldCurvePoint] = field(default_factory=list)
    _spline: Optional[CubicSpline] = field(default=None, repr=False)
    _ns_params: Optional[tuple] = field(default=None, repr=False)

    def __post_init__(self):
        if self.points:
            self._fit()

    def _fit(self):
        """Fit cubic spline interpolation to the curve points."""
        self.points.sort(key=lambda p: p.maturity_years)
        mats = np.array([p.maturity_years for p in self.points])
        yields = np.array([p.yield_pct for p in self.points])
        if len(mats) >= 3:
            self._spline = CubicSpline(mats, yields, bc_type='natural')
        self._fit_nelson_siegel(mats, yields)

    def _nelson_siegel(self, t, beta0, beta1, beta2, tau):
        """Nelson-Siegel yield curve model."""
        t = np.maximum(t, 1e-6)
        factor1 = (1 - np.exp(-t / tau)) / (t / tau)
        factor2 = factor1 - np.exp(-t / tau)
        return beta0 + beta1 * factor1 + beta2 * factor2

    def _fit_nelson_siegel(self, mats, yields):
        """Fit Nelson-Siegel parameters to observed yields."""
        def objective(params):
            beta0, beta1, beta2, tau = params
            if tau <= 0:
                return 1e10
            fitted = self._nelson_siegel(mats, beta0, beta1, beta2, tau)
            return np.sum((fitted - yields) ** 2)

        # Initial guess: beta0=long rate, beta1=short-long spread, beta2=curvature
        y0 = yields[0] if len(yields) > 0 else 4.0
        yn = yields[-1] if len(yields) > 0 else 4.5
        x0 = [yn, y0 - yn, 0.0, 2.0]
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8})
        if result.success:
            self._ns_params = tuple(result.x)

    def get_yield(self, maturity_years: float) -> float:
        """Get interpolated yield for a given maturity (in years)."""
        if self._spline is not None:
            mats = [p.maturity_years for p in self.points]
            if mats[0] <= maturity_years <= mats[-1]:
                return float(self._spline(maturity_years))
        if self._ns_params is not None:
            return float(self._nelson_siegel(
                np.array([maturity_years]), *self._ns_params)[0])
        # Fallback: linear interpolation
        return self._linear_interp(maturity_years)

    def _linear_interp(self, t: float) -> float:
        mats = [p.maturity_years for p in self.points]
        yields = [p.yield_pct for p in self.points]
        if t <= mats[0]:
            return yields[0]
        if t >= mats[-1]:
            return yields[-1]
        for i in range(len(mats) - 1):
            if mats[i] <= t <= mats[i + 1]:
                w = (t - mats[i]) / (mats[i + 1] - mats[i])
                return yields[i] + w * (yields[i + 1] - yields[i])
        return yields[-1]

    def discount_factor(self, maturity_years: float) -> float:
        """Get discount factor for a given maturity."""
        y = self.get_yield(maturity_years) / 100.0
        return np.exp(-y * maturity_years)

    def spot_rate(self, maturity_years: float) -> float:
        """Get continuously compounded spot rate."""
        return self.get_yield(maturity_years) / 100.0

    def forward_rate(self, t1: float, t2: float) -> float:
        """Get forward rate between t1 and t2."""
        if t2 <= t1:
            return self.spot_rate(t1)
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return -np.log(df2 / df1) / (t2 - t1)

    def par_rate(self, maturity_years: float, freq: int = 2) -> float:
        """Calculate par swap rate for given maturity and payment frequency."""
        dt = 1.0 / freq
        n_periods = int(maturity_years * freq)
        annuity = sum(self.discount_factor(i * dt) * dt
                      for i in range(1, n_periods + 1))
        df_final = self.discount_factor(maturity_years)
        if annuity == 0:
            return 0.0
        return (1 - df_final) / annuity

    def to_dict(self) -> dict:
        return {
            'as_of_date': str(self.as_of_date),
            'points': [{'maturity': p.maturity_years, 'yield': p.yield_pct}
                       for p in self.points],
            'nelson_siegel_params': self._ns_params,
        }

    def shift(self, basis_points: float) -> 'YieldCurve':
        """Create a parallel-shifted yield curve."""
        shifted_points = [
            YieldCurvePoint(p.maturity_years, p.yield_pct + basis_points / 100.0)
            for p in self.points
        ]
        return YieldCurve(as_of_date=self.as_of_date, points=shifted_points)


def build_us_treasury_curve(as_of: Optional[date] = None, live: bool = True) -> YieldCurve:
    """
    Build US Treasury yield curve.

    Args:
        as_of: Date for the curve. Defaults to today.
        live: If True, attempt to fetch live data from FRED/Treasury.gov
              before falling back to hardcoded values.
    """
    if as_of is None:
        as_of = date.today()

    # Try live data first
    if live:
        try:
            from .data_sources import build_live_treasury_curve
            as_of_str = str(as_of) if as_of != date.today() else None
            result = build_live_treasury_curve(as_of_str)
            if result.get("yields") and result.get("source") != "hardcoded_fallback":
                points = [
                    YieldCurvePoint(float(m), float(y))
                    for m, y in result["yields"].items()
                ]
                curve = YieldCurve(as_of_date=as_of, points=points)
                curve._data_source = result.get("source", "live")
                return curve
        except Exception:
            pass

    # Hardcoded fallback (Feb 2026)
    us_treasury_data = [
        (1/12, 3.699),   # 1M
        (2/12, 3.701),   # 2M
        (3/12, 3.690),   # 3M
        (4/12, 3.684),   # 4M
        (6/12, 3.640),   # 6M
        (1.0, 3.460),    # 1Y
        (2.0, 3.506),    # 2Y
        (3.0, 3.579),    # 3Y
        (5.0, 3.769),    # 5Y
        (7.0, 3.988),    # 7Y
        (10.0, 4.221),   # 10Y
        (20.0, 4.812),   # 20Y
        (30.0, 4.867),   # 30Y
    ]

    points = [YieldCurvePoint(m, y) for m, y in us_treasury_data]
    curve = YieldCurve(as_of_date=as_of, points=points)
    curve._data_source = "hardcoded_feb2026"
    return curve


def build_curve_from_dict(data: dict) -> YieldCurve:
    """Build yield curve from a dictionary of maturity->yield pairs."""
    points = [YieldCurvePoint(float(m), float(y)) for m, y in data.items()]
    return YieldCurve(as_of_date=date.today(), points=points)
