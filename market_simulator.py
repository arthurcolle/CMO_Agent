"""
Market Simulator for RL Training.

Generates realistic randomized market scenarios for CMO structuring:
- Yield curves (normal, flat, inverted, steep)
- TBA prices consistent with rates
- Collateral pools with varying characteristics
- Prepayment speed assumptions
- Vol environments

Used to create diverse training episodes for the RL agent.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .yield_curve import YieldCurve, YieldCurvePoint
from .spec_pool import SpecPool, AgencyType, CollateralType, PoolCharacteristic
from .tba import TBAPriceGrid, build_tba_price_grid


@dataclass
class MarketScenario:
    """A complete market scenario for one RL episode."""
    # Rates
    curve: YieldCurve
    treasury_10y: float
    mortgage_rate: float
    sofr: float
    fed_funds: float

    # Swap curve (spread over treasuries)
    swap_spreads: dict[float, float] = field(default_factory=dict)

    # TBA prices
    tba_grid: Optional[TBAPriceGrid] = None

    # Vol
    rate_vol_bps: float = 80.0  # Interest rate vol
    prepay_vol: float = 0.15     # Prepayment model uncertainty

    # Collateral available
    collateral_pools: list[SpecPool] = field(default_factory=list)

    # Deal mode
    deal_mode: str = "AGENCY"  # AGENCY or NON-AGENCY
    financing_rate: float = 0.0  # Repo/financing rate

    # ─── Full Desk P&L State (Song & Zhu 2019) ───────────────────────────
    # Dollar roll specialness determinants
    dealer_leverage: float = 20.0       # Squared asset/equity ratio of PD HoldCos
    cpr_dispersion: float = 0.03        # DispCPR: std of prepay speeds within cohort
    cpr_signed_change: float = 0.0      # CPR^{Signed,Change}: prepay exposure transfer
    fed_roll_indicator: float = 0.0     # 1 if Fed selling dollar rolls this month
    dollar_roll_specialness: float = 0.005  # Repo rate - implied DR financing rate (%)

    # Pool selection state (set by agent's SELECT_POOL action)
    selected_pool_type: int = 0         # 0=TBA, 1=LLB, 2=HLTV, 3=NY, 4=LFICO, 5=INV, 6=GEO
    selected_pool_payup: float = 0.0    # Payup in 32nds over TBA

    # Metadata
    scenario_id: str = ""
    regime: str = ""  # "normal", "crisis", "easing", "tightening"

    # ─── Ecosystem Fields (real data) ─────────────────────────────────
    # Money supply / Fed balance sheet
    m2: float = 0.0                   # M2 money supply ($B)
    fed_balance_sheet: float = 0.0    # WALCL ($M)
    rrp: float = 0.0                  # Reverse repo ($B)
    tga: float = 0.0                  # Treasury General Account ($M)

    # Housing
    housing_starts: float = 0.0       # HOUST (thousands, SAAR)
    building_permits: float = 0.0     # PERMIT
    case_shiller: float = 0.0         # CSUSHPINSA (index)
    months_supply: float = 0.0        # MSACSR
    hpi_yoy: float = 0.0             # House Price Index YoY %

    # Credit spreads
    hy_oas: float = 0.0              # HY OAS (bps)
    ig_oas: float = 0.0              # IG OAS (bps)

    # MBS / Macro
    mbs_bank_holdings: float = 0.0   # WSHOMCB ($B)
    unemployment: float = 0.0        # UNRATE (%)
    cpi_yoy: float = 0.0            # CPI YoY %
    mtg_tsy_spread: float = 0.0     # Mortgage rate - 10Y Treasury

    # Derived: trailing rate changes (percentage points)
    rate_chg_2y_1w: float = 0.0
    rate_chg_2y_1m: float = 0.0
    rate_chg_2y_3m: float = 0.0
    rate_chg_10y_1w: float = 0.0
    rate_chg_10y_1m: float = 0.0
    rate_chg_10y_3m: float = 0.0
    rate_chg_30y_1w: float = 0.0
    rate_chg_30y_1m: float = 0.0
    rate_chg_30y_3m: float = 0.0

    # Derived: realized vol, curve shape
    realized_vol_10y: float = 80.0   # Annualized bps
    slope_2s10s: float = 0.0         # 10Y - 2Y
    slope_5s30s: float = 0.0         # 30Y - 5Y
    butterfly_2s5s10s: float = 0.0   # 2*5Y - 2Y - 10Y

    # Derived: growth/flow
    m2_growth_yoy: float = 0.0       # M2 YoY growth %
    fed_bs_chg_3m: float = 0.0       # Fed balance sheet change (3mo, $T)

    def to_observation(self) -> dict:
        """Convert to a flat observation dict for the RL environment."""
        # Yield curve as array
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        curve_yields = [self.curve.get_yield(t) for t in tenors]

        # TBA prices for standard coupons
        tba_coupons = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
        tba_prices = []
        if self.tba_grid:
            for c in tba_coupons:
                tba_prices.append(self.tba_grid.get_price(c))
        else:
            tba_prices = [100.0] * len(tba_coupons)

        # Collateral summary
        pool_data = []
        for p in self.collateral_pools[:5]:  # Max 5 pools
            pool_data.extend([
                p.coupon, p.wac, p.wam / 360.0, p.wala / 360.0,
                p.original_balance / 1e9,  # Normalize to billions
                p.avg_fico / 850.0, p.avg_ltv / 100.0,
            ])
        # Pad if fewer pools
        while len(pool_data) < 35:
            pool_data.append(0.0)

        return {
            "curve_yields": curve_yields,
            "tba_prices": tba_prices,
            "rates": [self.treasury_10y, self.mortgage_rate, self.sofr,
                     self.fed_funds, self.rate_vol_bps],
            "pools": pool_data,
            "deal_mode": 1.0 if self.deal_mode == "AGENCY" else 0.0,
            "financing_rate": self.financing_rate,
            # Song & Zhu desk P&L state (6 dims)
            "desk_state": [
                self.dealer_leverage / 50.0,             # normalized (range ~10-50)
                self.cpr_dispersion * 10.0,              # amplified (range ~0.01-0.10)
                self.fed_roll_indicator,                  # binary 0/1
                self.dollar_roll_specialness * 100.0,    # pct -> bps scale
                self.selected_pool_payup / 100.0,        # 32nds / 100
                self.selected_pool_type / 7.0,           # pool type normalized
            ],
            # Ecosystem vector (40 dims: 16 primary + 24 derived)
            "ecosystem": [
                # 16 primary fields (normalized)
                self.m2 / 25000.0,                        # M2 ~$21T, normalize to ~1
                self.fed_balance_sheet / 9_000_000.0,     # WALCL ~$7-9T in $M
                self.rrp / 2500.0,                        # RRP peaked ~$2.5T
                self.tga / 1_000_000.0,                   # TGA ~$500B-$1T in $M
                self.housing_starts / 2000.0,             # HOUST ~1000-1800K
                self.building_permits / 2000.0,           # PERMIT similar
                self.case_shiller / 400.0,                # Index ~100-350
                self.months_supply / 12.0,                # MSACSR ~2-12 months
                self.hpi_yoy / 20.0,                      # YoY % ~-20 to +20
                self.hy_oas / 1000.0,                     # HY OAS ~300-2000bps
                self.ig_oas / 300.0,                      # IG OAS ~50-300bps
                self.mbs_bank_holdings / 3000.0,          # WSHOMCB ~$1-3T
                self.unemployment / 15.0,                 # UNRATE ~3-15%
                self.cpi_yoy / 10.0,                      # CPI YoY ~0-10%
                self.mtg_tsy_spread / 3.0,                # Spread ~1-3%
                0.0,                                      # reserved/padding
                # 9 trailing rate changes (normalized to ~1)
                self.rate_chg_2y_1w * 5.0,
                self.rate_chg_2y_1m * 2.0,
                self.rate_chg_2y_3m,
                self.rate_chg_10y_1w * 5.0,
                self.rate_chg_10y_1m * 2.0,
                self.rate_chg_10y_3m,
                self.rate_chg_30y_1w * 5.0,
                self.rate_chg_30y_1m * 2.0,
                self.rate_chg_30y_3m,
                # 5 derived shape/vol (normalized)
                self.realized_vol_10y / 200.0,            # Vol ~40-200bps
                self.slope_2s10s / 3.0,                   # Slope ~-1 to +3
                self.slope_5s30s / 2.0,                   # Slope ~-0.5 to +2
                self.butterfly_2s5s10s * 5.0,             # Butterfly ~-0.3 to +0.3
                # 2 growth/flow
                self.m2_growth_yoy / 30.0,                # M2 growth ~-5 to +30%
                self.fed_bs_chg_3m / 2.0,                 # BS change ~-1 to +2T
                # 9 padding to reach 40 dims total
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        }


class MarketSimulator:
    """Generates realistic market scenarios for RL training."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self._regime_probs = {
            "normal": 0.4,
            "steep": 0.15,
            "flat": 0.15,
            "inverted": 0.1,
            "crisis": 0.05,
            "easing": 0.1,
            "tightening": 0.05,
        }

    def generate_scenario(self, regime: Optional[str] = None) -> MarketScenario:
        """Generate a random market scenario."""
        if regime is None:
            regimes = list(self._regime_probs.keys())
            probs = list(self._regime_probs.values())
            regime = self.rng.choice(regimes, p=probs)

        curve, t10y, sofr, ff = self._generate_rates(regime)
        mortgage_rate = t10y + self.rng.uniform(1.3, 2.1)  # MBS-Treasury spread
        financing_rate = sofr + self.rng.uniform(-0.05, 0.15)

        tba_grid = build_tba_price_grid(t10y, "FNMA15", 30)

        # Generate collateral pools
        pools = self._generate_collateral(mortgage_rate, regime)

        vol = self._generate_vol(regime)
        deal_mode = "AGENCY" if self.rng.random() < 0.7 else "NON-AGENCY"

        # ─── Song & Zhu (2019) Desk State ─────────────────────────────────
        dealer_leverage, cpr_disp, cpr_signed, fed_roll, specialness = \
            self._generate_desk_state(regime, mortgage_rate, vol)

        return MarketScenario(
            curve=curve,
            treasury_10y=t10y,
            mortgage_rate=mortgage_rate,
            sofr=sofr,
            fed_funds=ff,
            tba_grid=tba_grid,
            rate_vol_bps=vol,
            collateral_pools=pools,
            deal_mode=deal_mode,
            financing_rate=financing_rate,
            dealer_leverage=dealer_leverage,
            cpr_dispersion=cpr_disp,
            cpr_signed_change=cpr_signed,
            fed_roll_indicator=fed_roll,
            dollar_roll_specialness=specialness,
            regime=regime,
        )

    def _generate_rates(self, regime: str) -> tuple:
        """Generate yield curve for a given regime."""
        if regime == "normal":
            short = self.rng.uniform(3.0, 5.0)
            long_spread = self.rng.uniform(0.5, 2.0)
            t10y = short + long_spread * 0.7
        elif regime == "steep":
            short = self.rng.uniform(1.0, 3.0)
            long_spread = self.rng.uniform(2.0, 3.5)
            t10y = short + long_spread * 0.7
        elif regime == "flat":
            short = self.rng.uniform(3.0, 5.5)
            long_spread = self.rng.uniform(-0.3, 0.3)
            t10y = short + long_spread
        elif regime == "inverted":
            short = self.rng.uniform(4.5, 6.5)
            long_spread = self.rng.uniform(-1.5, -0.3)
            t10y = short + long_spread * 0.7
        elif regime == "crisis":
            short = self.rng.uniform(0.0, 1.0)
            long_spread = self.rng.uniform(1.5, 3.5)
            t10y = short + long_spread * 0.7
        elif regime == "easing":
            short = self.rng.uniform(2.0, 4.0)
            long_spread = self.rng.uniform(1.0, 2.5)
            t10y = short + long_spread * 0.7
        else:  # tightening
            short = self.rng.uniform(4.0, 6.0)
            long_spread = self.rng.uniform(-0.5, 1.0)
            t10y = short + long_spread * 0.7

        sofr = short + self.rng.uniform(-0.25, 0.0)
        ff = short + self.rng.uniform(-0.15, 0.15)

        # Build curve points
        tenors = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
        points = []
        for t in tenors:
            if t <= 1:
                y = short + self.rng.normal(0, 0.05)
            else:
                # Interpolate to long end
                w = min(1.0, (t - 1) / 29.0)
                y = short + long_spread * w + self.rng.normal(0, 0.03)
            points.append(YieldCurvePoint(t, max(0.01, y)))

        from datetime import date
        curve = YieldCurve(as_of_date=date.today(), points=points)
        return curve, t10y, sofr, ff

    def _generate_collateral(self, mortgage_rate: float, regime: str) -> list[SpecPool]:
        """Generate a set of collateral pools."""
        n_pools = self.rng.randint(1, 4)
        pools = []

        for i in range(n_pools):
            # Coupon near current coupon (within 100bps)
            coupon = round(mortgage_rate - self.rng.uniform(0.3, 1.0), 1)
            coupon = round(coupon * 2) / 2  # Round to nearest 0.5
            coupon = max(2.0, min(8.0, coupon))

            wac = coupon + self.rng.uniform(0.3, 0.7)
            wam = int(self.rng.choice([357, 356, 355, 354, 350, 345, 340, 180, 178, 176]))
            wala = 360 - wam if wam <= 360 else 0

            balance = self.rng.uniform(50, 500) * 1_000_000
            balance = round(balance, -3)

            agency = self.rng.choice([AgencyType.FNMA, AgencyType.FHLMC, AgencyType.GNMA])
            coll_type = {
                AgencyType.FNMA: CollateralType.FN,
                AgencyType.FHLMC: CollateralType.FH,
                AgencyType.GNMA: CollateralType.G2,
            }[agency]

            avg_fico = self.rng.uniform(680, 780)
            avg_ltv = self.rng.uniform(60, 97)
            avg_loan = self.rng.uniform(150000, 500000)

            pool = SpecPool(
                pool_id=f"POOL_{i+1}",
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
                avg_loan_size=round(avg_loan),
            )
            pools.append(pool)

        return pools

    def _generate_vol(self, regime: str) -> float:
        """Generate rate volatility for the regime."""
        base_vols = {
            "normal": 70, "steep": 85, "flat": 60,
            "inverted": 90, "crisis": 150, "easing": 100,
            "tightening": 95,
        }
        base = base_vols.get(regime, 80)
        return base + self.rng.normal(0, 10)

    def _generate_desk_state(self, regime: str, mortgage_rate: float,
                              vol: float) -> tuple[float, float, float, float, float]:
        """
        Generate Song & Zhu (2019) dollar roll specialness determinants.

        From Table 6 multivariate regression (no dummies):
            Specialness = 1.168 * DispCPR - 0.035 * Leverage - 1.550 * CPR^{Signed,Change} - 0.514

        From Table 9: Fed dollar roll sales → -50bps specialness change.

        Returns: (dealer_leverage, cpr_dispersion, cpr_signed_change, fed_roll, specialness)
        """
        # Dealer leverage: squared asset/equity of PD HoldCos
        # Mean ~20, SD ~4.86 (Table 3, Song & Zhu)
        regime_leverage = {
            "normal": (20.0, 3.0), "steep": (18.0, 3.0), "flat": (22.0, 4.0),
            "inverted": (25.0, 5.0), "crisis": (35.0, 8.0), "easing": (15.0, 3.0),
            "tightening": (28.0, 5.0),
        }
        lev_mu, lev_sd = regime_leverage.get(regime, (20.0, 3.0))
        dealer_leverage = max(8.0, self.rng.normal(lev_mu, lev_sd))

        # CPR dispersion: std of prepay speeds within coupon cohort
        # Mean ~0.026, SD ~0.024 (Table 3)
        # Higher vol → higher dispersion
        vol_factor = max(0.5, vol / 80.0)
        cpr_disp = max(0.005, self.rng.normal(0.026, 0.015) * vol_factor)

        # CPR^{Signed,Change}: prepay exposure transfer measure
        # Mean ~0, SD ~0.149 (Table 3)
        # Premium MBS: positive = faster-than-expected prepays (bad for roll buyer)
        moneyness = mortgage_rate - 5.5  # rough premium/discount indicator
        cpr_signed = self.rng.normal(0.0, 0.10)
        if moneyness > 0:
            cpr_signed += self.rng.uniform(0, 0.05)  # premium → slightly positive

        # Fed dollar roll sales indicator
        # More likely during easing/crisis (QE periods)
        fed_roll_probs = {
            "normal": 0.05, "steep": 0.10, "flat": 0.05,
            "inverted": 0.08, "crisis": 0.40, "easing": 0.50,
            "tightening": 0.02,
        }
        fed_roll = 1.0 if self.rng.random() < fed_roll_probs.get(regime, 0.05) else 0.0

        # Song & Zhu Table 6 Column (2) with moneyness FEs:
        # DispCPR=1.893***, Leverage=-0.019*, CPR_signed=-1.312***
        # Unconditional mean specialness ≈ 0.70% (Table 3 Panel B)
        # Center leverage around its sample mean (20.24) to match observed levels
        specialness = (
            0.70                                          # unconditional mean
            + 1.893 * (cpr_disp - 0.026)                 # DispCPR effect (centered)
            - 0.019 * (dealer_leverage - 20.24)           # Leverage effect (centered)
            - 1.312 * cpr_signed                          # CPR exposure transfer
        )
        # Fed effect: Table 9, d_roll coefficient ≈ -0.50
        if fed_roll > 0:
            specialness -= 0.50

        # Add noise (residual std ≈ 0.40 from R²=0.28 with moneyness FEs)
        specialness += self.rng.normal(0, 0.25)

        # Specialness typically 0-5% annualized, can go negative
        specialness = float(np.clip(specialness, -1.0, 6.0))

        return dealer_leverage, cpr_disp, cpr_signed, fed_roll, specialness

    def generate_batch(self, n: int = 100) -> list[MarketScenario]:
        """Generate a batch of market scenarios for training."""
        return [self.generate_scenario() for _ in range(n)]


def song_zhu_specialness(
    cpr_dispersion: float,
    dealer_leverage: float,
    cpr_signed_change: float,
    fed_roll_sales: bool = False,
) -> float:
    """
    Compute dollar roll specialness using Song & Zhu (RFS 2019) Table 6.

    Specialness_it = 1.168 * DispCPR_it - 0.035 * Leverage_{t-1}
                     - 1.550 * CPR^{Signed,Change}_it - 0.514

    Fed effect (Table 9): d_roll → -50bps

    Args:
        cpr_dispersion: Dispersion of realized prepay rates within cohort (%)
        dealer_leverage: Squared asset/equity ratio of primary dealer HoldCos
        cpr_signed_change: Signed prepay shock (positive = unfavorable to roll buyer)
        fed_roll_sales: Whether Fed is selling dollar rolls this month

    Returns:
        Dollar roll specialness in percentage points (annualized)
    """
    specialness = (
        0.70                                          # unconditional mean
        + 1.893 * (cpr_dispersion - 0.026)            # DispCPR effect (centered)
        - 0.019 * (dealer_leverage - 20.24)           # Leverage effect (centered)
        - 1.312 * cpr_signed_change                   # CPR exposure transfer
    )
    if fed_roll_sales:
        specialness -= 0.50  # Table 9: β1 ≈ -0.50

    return float(np.clip(specialness, -1.0, 6.0))


def specialness_to_roll_income_ticks(
    specialness_pct: float,
    collateral_balance: float,
    tba_price: float = 100.0,
) -> float:
    """
    Convert dollar roll specialness to monthly income in ticks (32nds).

    Roll income = specialness (annualized %) / 12 * collateral_notional
    Then express as ticks per $100M collateral.

    From Song & Zhu Table 1:
    - FN 5% avg specialness 0.48% → ~1.3 ticks/month per $100M
    - FN 2.5% peak specialness 5.13% → ~13.7 ticks/month per $100M
    - During QE: 30-50+ ticks/month at peak
    """
    monthly_specialness = specialness_pct / 100.0 / 12.0  # annual % → monthly decimal
    roll_income_dollars = monthly_specialness * collateral_balance * (tba_price / 100.0)
    # Convert to ticks: income / collateral * 32 * 100
    roll_ticks = roll_income_dollars / collateral_balance * 32 * 100
    return float(roll_ticks)
