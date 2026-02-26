"""
Investment Grade Credit Trading Desk RL Environment.

Gymnasium-compatible environment for IG corporate bond and CDS trading.
Covers:
  - Cash bond trading (on/off-the-run, new issue)
  - CDX index trading and basis
  - Single-name CDS
  - Sector rotation
  - New issue concession capture
  - Curve trades (2s5s, 5s10s in credit)
  - Cross-over credits (BBB-/BB+)

References:
  - Collin-Dufresne, Goldstein & Martin (2001) "Determinants of Credit Spread Changes" JF
  - Longstaff, Mithal & Neis (2005) "Corporate Yield Spreads" JF
  - Bao, Pan & Wang (2011) "Illiquidity of Corporate Bonds" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Universe ─────────────────────────────────────────────────────────────

N_SECTORS = 8  # Fins, Tech, Industrials, Utilities, Energy, Healthcare, Consumer, Telecom
SECTOR_NAMES = ["Fins", "Tech", "Industrials", "Utilities", "Energy",
                "Healthcare", "Consumer", "Telecom"]
N_MATURITIES = 5  # 2Y, 3Y, 5Y, 7Y, 10Y
MATURITY_POINTS = [2, 3, 5, 7, 10]
N_RATINGS = 4  # AAA/AA, A, BBB, BBB-/Crossover


@dataclass
class IGScenario:
    """Market state for IG credit desk."""
    # IG CDX index spread (bps)
    cdx_ig_spread: float = 60.0
    # Sector spreads (bps) by sector x maturity
    sector_spreads: np.ndarray = field(
        default_factory=lambda: np.full((N_SECTORS, N_MATURITIES), 80.0))
    # Spread vol by sector
    spread_vol: np.ndarray = field(default_factory=lambda: np.full(N_SECTORS, 5.0))
    # New issue calendar ($B this week)
    new_issue_volume: float = 20.0
    # New issue concession (bps)
    new_issue_concession: float = 5.0
    # CDS-bond basis (bps, negative = bonds cheap)
    cds_bond_basis: float = -5.0
    # Treasury curve (for total return)
    tsy_2y: float = 4.5
    tsy_5y: float = 4.3
    tsy_10y: float = 4.2
    # Macro
    vix: float = 15.0
    ig_fund_flows_bn: float = 2.0  # Weekly inflows (+) / outflows (-)
    # Rating migration rate (annual %, down)
    downgrade_rate: float = 3.0
    # Cross-sector correlation
    sector_correlation: float = 0.6
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            [self.cdx_ig_spread / 200.0],              # 1
            self.sector_spreads.flatten() / 300.0,      # 40
            self.spread_vol / 20.0,                     # 8
            [self.new_issue_volume / 50.0],             # 1
            [self.new_issue_concession / 20.0],         # 1
            [self.cds_bond_basis / 50.0],               # 1
            [self.tsy_2y / 10.0],                       # 1
            [self.tsy_5y / 10.0],                       # 1
            [self.tsy_10y / 10.0],                      # 1
            [self.vix / 50.0],                          # 1
            [self.ig_fund_flows_bn / 20.0],             # 1
            [self.downgrade_rate / 10.0],               # 1
            [self.sector_correlation],                   # 1
        ])  # Total: 59

    @property
    def obs_dim(self) -> int:
        return 59


@dataclass
class IGBook:
    """Position book for IG credit desk."""
    # Cash bond positions by sector x maturity ($M face)
    bond_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((N_SECTORS, N_MATURITIES)))
    # CDS positions by sector ($M notional)
    cds_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_SECTORS))
    # CDX index position ($M notional)
    index_position: float = 0.0
    # New issue allocation pending ($M)
    new_issue_pending: float = 0.0
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    carry_earned: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.bond_positions.flatten() / 100.0,   # 40
            self.cds_positions / 100.0,              # 8
            [self.index_position / 500.0],           # 1
            [self.new_issue_pending / 100.0],        # 1
            [self.realized_pnl / 5000.0],            # 1
            [self.unrealized_pnl / 5000.0],          # 1
            [self.carry_earned / 1000.0],            # 1
        ])  # Total: 53

    @property
    def state_dim(self) -> int:
        return 53

    @property
    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.bond_positions)) +
                     np.sum(np.abs(self.cds_positions)) + abs(self.index_position))

    @property
    def spread_dv01(self) -> float:
        """Approximate spread DV01 ($K/bp)."""
        dv01 = 0.0
        for i in range(N_SECTORS):
            for j in range(N_MATURITIES):
                dv01 += self.bond_positions[i, j] * MATURITY_POINTS[j] * 0.01
        return float(dv01)


class IGAction(IntEnum):
    NOOP = 0
    BUY_BOND = 1
    SELL_BOND = 2
    BUY_CDS = 3       # Buy protection
    SELL_CDS = 4       # Sell protection
    BUY_INDEX = 5      # Buy CDX protection
    SELL_INDEX = 6     # Sell CDX protection
    BASIS_TRADE = 7    # Cash-CDS basis (long bond, buy CDS)
    NEW_ISSUE = 8      # Bid on new issue
    SECTOR_ROTATE = 9  # Sell one sector, buy another
    CREDIT_CURVE = 10  # 2s5s or 5s10s credit curve trade
    FLATTEN = 11
    CLOSE_DAY = 12


class IGScenarioGenerator:
    REGIMES = ["normal", "tightening", "widening", "new_issue_flood", "risk_off"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> IGScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = IGScenario(regime=regime)

        if regime == "tightening":
            s.cdx_ig_spread = self.rng.uniform(40, 65)
            base_spread = self.rng.uniform(50, 80)
            s.vix = self.rng.uniform(10, 18)
            s.ig_fund_flows_bn = self.rng.uniform(2, 8)
        elif regime == "widening":
            s.cdx_ig_spread = self.rng.uniform(80, 150)
            base_spread = self.rng.uniform(100, 180)
            s.vix = self.rng.uniform(18, 35)
            s.ig_fund_flows_bn = self.rng.uniform(-8, 0)
        elif regime == "risk_off":
            s.cdx_ig_spread = self.rng.uniform(100, 200)
            base_spread = self.rng.uniform(120, 250)
            s.vix = self.rng.uniform(25, 45)
            s.ig_fund_flows_bn = self.rng.uniform(-15, -3)
        elif regime == "new_issue_flood":
            s.cdx_ig_spread = self.rng.uniform(55, 90)
            base_spread = self.rng.uniform(65, 110)
            s.new_issue_volume = self.rng.uniform(30, 60)
            s.new_issue_concession = self.rng.uniform(8, 20)
            s.vix = self.rng.uniform(12, 22)
            s.ig_fund_flows_bn = self.rng.uniform(0, 5)
        else:
            s.cdx_ig_spread = self.rng.uniform(50, 90)
            base_spread = self.rng.uniform(60, 110)
            s.vix = self.rng.uniform(12, 22)
            s.ig_fund_flows_bn = self.rng.uniform(-3, 5)

        # Sector spreads with term structure
        sector_betas = [1.2, 0.8, 1.0, 0.6, 1.3, 0.9, 0.85, 1.1]
        for i in range(N_SECTORS):
            for j in range(N_MATURITIES):
                curve_slope = (MATURITY_POINTS[j] - 2) * 2.5  # upward sloping
                s.sector_spreads[i, j] = max(20,
                    base_spread * sector_betas[i] + curve_slope +
                    self.rng.normal(0, 5))

        # Spread vol
        for i in range(N_SECTORS):
            s.spread_vol[i] = s.cdx_ig_spread * 0.05 * sector_betas[i] + \
                self.rng.uniform(1, 4)

        s.new_issue_volume = max(0, s.new_issue_volume + self.rng.normal(0, 3))
        s.new_issue_concession = max(1, s.new_issue_concession + self.rng.normal(0, 1))
        s.cds_bond_basis = self.rng.normal(-5, 10)
        s.tsy_2y = self.rng.uniform(2.0, 6.0)
        s.tsy_5y = s.tsy_2y + self.rng.uniform(-0.5, 1.0)
        s.tsy_10y = s.tsy_5y + self.rng.uniform(-0.3, 0.8)
        s.downgrade_rate = self.rng.uniform(1.0, 8.0)
        s.sector_correlation = self.rng.uniform(0.4, 0.85)

        return s

    def step_scenario(self, s: IGScenario) -> IGScenario:
        new = IGScenario(regime=s.regime)

        # CDX index
        new.cdx_ig_spread = max(20, s.cdx_ig_spread + self.rng.normal(0, 2))

        # Sector spreads
        systematic = self.rng.normal(0, 1)  # Common factor
        for i in range(N_SECTORS):
            for j in range(N_MATURITIES):
                idio = self.rng.normal(0, s.spread_vol[i] * 0.2)
                sys_move = systematic * s.spread_vol[i] * s.sector_correlation * 0.2
                new.sector_spreads[i, j] = max(15,
                    s.sector_spreads[i, j] + idio + sys_move)

        new.spread_vol = np.maximum(1.0, s.spread_vol + self.rng.normal(0, 0.5, N_SECTORS))
        new.new_issue_volume = max(0, s.new_issue_volume + self.rng.normal(0, 3))
        new.new_issue_concession = max(1, s.new_issue_concession + self.rng.normal(0, 0.5))
        new.cds_bond_basis = s.cds_bond_basis + self.rng.normal(0, 1)
        new.tsy_2y = s.tsy_2y + self.rng.normal(0, 0.03)
        new.tsy_5y = s.tsy_5y + self.rng.normal(0, 0.03)
        new.tsy_10y = s.tsy_10y + self.rng.normal(0, 0.03)
        new.vix = max(8, s.vix + self.rng.normal(0, 1))
        new.ig_fund_flows_bn = s.ig_fund_flows_bn + self.rng.normal(0, 1.5)
        new.downgrade_rate = max(0.5, s.downgrade_rate + self.rng.normal(0, 0.2))
        new.sector_correlation = np.clip(s.sector_correlation + self.rng.normal(0, 0.02), 0.2, 0.95)

        return new


class IGCreditEnv(gym.Env):
    """
    Gymnasium environment for IG credit trading desk.

    Action space: MultiDiscrete([13, 8, 5, 10])
      - action_type (13): NOOP through CLOSE_DAY
      - sector_idx (8): which sector
      - maturity_idx (5): which maturity point
      - size_bucket (10): position size

    Observation: 112-dim
      - Market (59) + Book (53)

    Reward: Daily P&L from carry + spread MTM + new issue capture
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 20,
        max_actions_per_day: int = 10,
        gross_limit: float = 2000.0,  # $M
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.gross_limit = gross_limit
        self.gen = IGScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([13, 8, 5, 10])

        self._market_dim = 59
        self._book_dim = 53
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._size_buckets = np.linspace(5, 100, 10)  # $M face

        self.scenario: Optional[IGScenario] = None
        self.prev_scenario: Optional[IGScenario] = None
        self.book: Optional[IGBook] = None
        self.day = 0
        self.actions_today = 0

    def _get_obs(self) -> np.ndarray:
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([
            self.scenario.to_observation(),
            self.book.to_vector(),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.gen = IGScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = IGBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        act = IGAction(action[0])
        sector = min(action[1], N_SECTORS - 1)
        mat = min(action[2], N_MATURITIES - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == IGAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == IGAction.NOOP:
            pass
        elif act == IGAction.BUY_BOND:
            reward = self._trade_bond(sector, mat, size)
        elif act == IGAction.SELL_BOND:
            reward = self._trade_bond(sector, mat, -size)
        elif act == IGAction.BUY_CDS:
            reward = self._trade_cds(sector, size)
        elif act == IGAction.SELL_CDS:
            reward = self._trade_cds(sector, -size)
        elif act == IGAction.BUY_INDEX:
            reward = self._trade_index(size)
        elif act == IGAction.SELL_INDEX:
            reward = self._trade_index(-size)
        elif act == IGAction.BASIS_TRADE:
            reward = self._basis_trade(sector, mat, size)
        elif act == IGAction.NEW_ISSUE:
            reward = self._new_issue_bid(sector, mat, size)
        elif act == IGAction.SECTOR_ROTATE:
            reward = self._sector_rotate(sector, mat, size)
        elif act == IGAction.CREDIT_CURVE:
            reward = self._credit_curve(sector, mat, size)
        elif act == IGAction.FLATTEN:
            reward = self._flatten()

        if self.book.gross_exposure > self.gross_limit:
            reward -= 3.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {
            "day": self.day,
            "gross": self.book.gross_exposure,
            "spread_dv01": self.book.spread_dv01,
            "realized_pnl": self.book.realized_pnl,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _trade_bond(self, sector: int, mat: int, size: float) -> float:
        self.book.bond_positions[sector, mat] += size
        # IG bid/ask ~0.5-2pts depending on maturity/liquidity
        spread = self.scenario.sector_spreads[sector, mat]
        bid_ask = 0.3 + spread / 500 + MATURITY_POINTS[mat] * 0.05
        cost = abs(size) * bid_ask / 100
        return -cost * 1000  # $K

    def _trade_cds(self, sector: int, size: float) -> float:
        self.book.cds_positions[sector] += size
        cost = abs(size) * 0.5 / 10000 * 5 * 10  # ~0.5bp running on 5Y
        return -cost

    def _trade_index(self, size: float) -> float:
        self.book.index_position += size
        cost = abs(size) * 0.25 / 10000 * 5 * 10  # CDX very liquid
        return -cost

    def _basis_trade(self, sector: int, mat: int, size: float) -> float:
        self.book.bond_positions[sector, mat] += size
        self.book.cds_positions[sector] += size  # Buy protection
        cost = abs(size) * 1.0 / 100 * 1000  # ~1pt to execute both legs
        return -cost

    def _new_issue_bid(self, sector: int, mat: int, size: float) -> float:
        if self.scenario.new_issue_volume < 5:
            return -0.1  # No issuance
        allocation = size * self.gen.rng.uniform(0.2, 0.8)
        self.book.bond_positions[sector, mat] += allocation
        # Capture concession
        concession = self.scenario.new_issue_concession
        pnl = allocation * concession / 100 * MATURITY_POINTS[mat] * 0.01 * 1000
        return float(pnl) * 0.5  # Partial capture

    def _sector_rotate(self, sector: int, mat: int, size: float) -> float:
        # Find widest sector to buy, sell current
        spreads_at_mat = self.scenario.sector_spreads[:, mat]
        buy_sector = int(np.argmax(spreads_at_mat))
        if buy_sector == sector:
            buy_sector = int(np.argmin(spreads_at_mat))

        cost1 = abs(self._trade_bond(sector, mat, -size))
        cost2 = abs(self._trade_bond(buy_sector, mat, size))
        return -(cost1 + cost2)

    def _credit_curve(self, sector: int, mat: int, size: float) -> float:
        short_mat = max(0, mat - 1)
        long_mat = min(N_MATURITIES - 1, mat + 1)
        # Steepener: sell short, buy long
        self.book.bond_positions[sector, short_mat] -= size * 0.5
        self.book.bond_positions[sector, long_mat] += size * 0.5
        cost = abs(size) * 0.6 / 100 * 1000
        return -cost

    def _flatten(self) -> float:
        cost = 0.0
        for i in range(N_SECTORS):
            for j in range(N_MATURITIES):
                if abs(self.book.bond_positions[i, j]) > 0.1:
                    spread = self.scenario.sector_spreads[i, j]
                    tc = abs(self.book.bond_positions[i, j]) * (0.3 + spread / 500) / 100 * 1000
                    cost += tc
                    self.book.bond_positions[i, j] *= 0.1
            if abs(self.book.cds_positions[i]) > 0.1:
                cost += abs(self.book.cds_positions[i]) * 0.3
                self.book.cds_positions[i] *= 0.1
        self.book.index_position *= 0.1
        return -cost

    def _end_of_day(self) -> float:
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        daily_pnl = 0.0

        # Carry
        for i in range(N_SECTORS):
            for j in range(N_MATURITIES):
                pos = self.book.bond_positions[i, j]
                if abs(pos) > 0.01:
                    spread = self.prev_scenario.sector_spreads[i, j]
                    carry = pos * spread / 10000 / 252 * 1000  # $K
                    daily_pnl += carry
            # CDS carry
            cds = self.book.cds_positions[i]
            if abs(cds) > 0.01:
                avg_spread = np.mean(self.prev_scenario.sector_spreads[i])
                daily_pnl -= cds * avg_spread / 10000 / 252 * 1000

        # Index carry
        if abs(self.book.index_position) > 0.01:
            daily_pnl -= self.book.index_position * \
                self.prev_scenario.cdx_ig_spread / 10000 / 252 * 1000

        self.book.carry_earned += daily_pnl

        # Spread MTM
        mtm = 0.0
        for i in range(N_SECTORS):
            for j in range(N_MATURITIES):
                pos = self.book.bond_positions[i, j]
                if abs(pos) > 0.01:
                    old_s = self.prev_scenario.sector_spreads[i, j]
                    new_s = self.scenario.sector_spreads[i, j]
                    dur = MATURITY_POINTS[j] * 0.9
                    price_move = -(new_s - old_s) / 100 * dur  # pts
                    mtm += pos * price_move / 100 * 1000

            cds = self.book.cds_positions[i]
            if abs(cds) > 0.01:
                old_s = np.mean(self.prev_scenario.sector_spreads[i])
                new_s = np.mean(self.scenario.sector_spreads[i])
                mtm += cds * (new_s - old_s) / 100 * 4 * 10

        # Index MTM
        if abs(self.book.index_position) > 0.01:
            idx_change = self.scenario.cdx_ig_spread - self.prev_scenario.cdx_ig_spread
            mtm += self.book.index_position * idx_change / 100 * 4 * 10

        self.book.unrealized_pnl += mtm
        daily_pnl += mtm
        self.book.realized_pnl += daily_pnl

        return float(np.clip(daily_pnl, -50, 50))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        print(f"CDX IG: {self.scenario.cdx_ig_spread:.0f}bps | VIX: {self.scenario.vix:.1f}")
        print(f"Gross: ${self.book.gross_exposure:.0f}M | DV01: ${self.book.spread_dv01:.0f}K")
        print(f"P&L: ${self.book.realized_pnl:.0f}K (carry: ${self.book.carry_earned:.0f}K)")


def make_ig_credit_env(seed: Optional[int] = None, **kwargs) -> IGCreditEnv:
    return IGCreditEnv(seed=seed, **kwargs)
