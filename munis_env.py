"""
Municipal Bonds Trading Desk RL Environment.

Gymnasium-compatible environment for municipal bond trading.
Covers:
  - Tax-exempt GO and Revenue bonds
  - Muni/Treasury ratio trading
  - New issue calendar management
  - Credit quality (AAA MMD to BBB)
  - Taxable munis and Build America Bonds
  - State-level spread differentiation
  - Muni ETF hedging (MUB, HYD)
  - Advance refunding and pre-refunded munis

References:
  - Ang, Bhatt & Sun (2011) "Taxes on Tax-Exempt Bonds" JF
  - Green, Hollifield & Schurhoff (2007) "Dealer Intermediation" JFE
  - Schwert (2017) "Municipal Bond Liquidity" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Market Structure ─────────────────────────────────────────────────────

N_STATES = 8  # CA, NY, TX, IL, FL, NJ, PA, OH (top issuers)
STATE_NAMES = ["CA", "NY", "TX", "IL", "FL", "NJ", "PA", "OH"]
N_MATURITIES = 6  # 2, 5, 10, 15, 20, 30
MAT_POINTS = [2, 5, 10, 15, 20, 30]
N_QUALITY = 3  # AAA, AA, A/BBB


@dataclass
class MuniScenario:
    """Market state for municipal bond desk."""
    # MMD AAA curve (yields, %)
    mmd_curve: np.ndarray = field(default_factory=lambda: np.zeros(N_MATURITIES))
    # Treasury curve for ratio calculation
    tsy_curve: np.ndarray = field(default_factory=lambda: np.zeros(N_MATURITIES))
    # Muni/Treasury ratios
    muni_tsy_ratios: np.ndarray = field(default_factory=lambda: np.ones(N_MATURITIES))
    # State credit spreads to MMD (bps) by state x quality
    state_spreads: np.ndarray = field(
        default_factory=lambda: np.zeros((N_STATES, N_QUALITY)))
    # New issue supply ($B this week)
    new_issue_supply: float = 8.0
    # Fund flows ($B this week)
    muni_fund_flows: float = 0.5
    # Dealer inventory ($B)
    dealer_inventory: float = 12.0
    # Tax rate environment
    top_marginal_rate: float = 37.0  # %
    salt_cap: float = 10000.0  # SALT deduction cap ($)
    # Taxable muni spread to corp (bps)
    taxable_muni_spread: float = 20.0
    # ETF NAV premium/discount (%)
    mub_premium: float = 0.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.mmd_curve / 10.0,                     # 6
            self.tsy_curve / 10.0,                     # 6
            self.muni_tsy_ratios / 2.0,                # 6
            self.state_spreads.flatten() / 200.0,      # 24
            [self.new_issue_supply / 20.0],            # 1
            [self.muni_fund_flows / 10.0],             # 1
            [self.dealer_inventory / 30.0],            # 1
            [self.top_marginal_rate / 50.0],           # 1
            [self.taxable_muni_spread / 100.0],        # 1
            [self.mub_premium / 5.0],                  # 1
        ])  # Total: 48

    @property
    def obs_dim(self) -> int:
        return 48


@dataclass
class MuniBook:
    """Position book for muni desk."""
    # Tax-exempt positions by state x maturity ($M par)
    te_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((N_STATES, N_MATURITIES)))
    # Taxable muni positions ($M par) by maturity
    taxable_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_MATURITIES))
    # Treasury hedge ($M DV01)
    tsy_hedge: np.ndarray = field(default_factory=lambda: np.zeros(N_MATURITIES))
    # ETF hedge (shares, rough $M equiv)
    etf_hedge: float = 0.0
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    carry_earned: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.te_positions.flatten() / 50.0,    # 48
            self.taxable_positions / 50.0,          # 6
            self.tsy_hedge / 50.0,                  # 6
            [self.etf_hedge / 100.0],               # 1
            [self.realized_pnl / 5000.0],           # 1
            [self.unrealized_pnl / 5000.0],         # 1
            [self.carry_earned / 1000.0],           # 1
        ])  # Total: 64

    @property
    def state_dim(self) -> int:
        return 64

    @property
    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.te_positions)) +
                     np.sum(np.abs(self.taxable_positions)))


class MuniAction(IntEnum):
    NOOP = 0
    BUY_TE = 1           # Buy tax-exempt
    SELL_TE = 2          # Sell tax-exempt
    BUY_TAXABLE = 3      # Buy taxable muni
    SELL_TAXABLE = 4
    RATIO_TRADE = 5      # Muni/Treasury ratio trade
    REVERSE_RATIO = 6
    BID_NEW_ISSUE = 7    # Bid on new issue
    HEDGE_TSY = 8        # Hedge with Treasuries
    HEDGE_ETF = 9        # Hedge with MUB ETF
    STATE_ROTATE = 10    # Rotate between states
    FLATTEN = 11
    CLOSE_DAY = 12


class MuniScenarioGenerator:
    REGIMES = ["normal", "tax_hike", "supply_glut", "risk_off", "rate_rally"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> MuniScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = MuniScenario(regime=regime)

        # Treasury curve
        short_rate = self.rng.uniform(2.0, 5.5)
        slope = self.rng.uniform(0.0, 2.0) if regime != "rate_rally" else self.rng.uniform(1.0, 3.0)
        for i, t in enumerate(MAT_POINTS):
            s.tsy_curve[i] = short_rate + slope * (1 - np.exp(-t / 7.0)) + \
                self.rng.normal(0, 0.03)

        # Muni/Treasury ratios
        if regime == "tax_hike":
            base_ratio = self.rng.uniform(0.55, 0.70)  # Munis rich
            s.top_marginal_rate = self.rng.uniform(39, 45)
        elif regime == "supply_glut":
            base_ratio = self.rng.uniform(0.85, 1.05)  # Munis cheap
            s.new_issue_supply = self.rng.uniform(12, 25)
        elif regime == "risk_off":
            base_ratio = self.rng.uniform(0.80, 1.00)
            s.muni_fund_flows = self.rng.uniform(-5, -1)
        else:
            base_ratio = self.rng.uniform(0.65, 0.85)

        for i in range(N_MATURITIES):
            # Ratios are higher (munis cheaper) at the long end
            mat_adj = (MAT_POINTS[i] - 10) * 0.005
            s.muni_tsy_ratios[i] = base_ratio + mat_adj + self.rng.normal(0, 0.02)
            s.mmd_curve[i] = s.tsy_curve[i] * s.muni_tsy_ratios[i]

        # State spreads
        # IL and NJ wider (fiscal stress)
        state_betas = [1.0, 0.9, 0.7, 2.0, 0.6, 1.8, 1.2, 1.1]
        for i in range(N_STATES):
            for j in range(N_QUALITY):
                quality_add = [0, 15, 50][j]
                s.state_spreads[i, j] = state_betas[i] * 10 + quality_add + \
                    self.rng.normal(0, 3)

        s.muni_fund_flows = s.muni_fund_flows if regime == "risk_off" else \
            self.rng.uniform(-3, 5)
        s.dealer_inventory = self.rng.uniform(8, 20)
        s.taxable_muni_spread = self.rng.uniform(-10, 40)
        s.mub_premium = self.rng.normal(0, 0.5)
        s.salt_cap = 10000.0

        return s

    def step_scenario(self, s: MuniScenario) -> MuniScenario:
        new = MuniScenario(regime=s.regime)

        # Evolve Treasury curve
        for i in range(N_MATURITIES):
            new.tsy_curve[i] = s.tsy_curve[i] + self.rng.normal(0, 0.02)

        # Evolve ratios
        for i in range(N_MATURITIES):
            new.muni_tsy_ratios[i] = np.clip(
                s.muni_tsy_ratios[i] + self.rng.normal(0, 0.01), 0.4, 1.3)
            new.mmd_curve[i] = new.tsy_curve[i] * new.muni_tsy_ratios[i]

        # State spreads
        for i in range(N_STATES):
            for j in range(N_QUALITY):
                new.state_spreads[i, j] = max(0,
                    s.state_spreads[i, j] + self.rng.normal(0, 1))

        new.new_issue_supply = max(0, s.new_issue_supply + self.rng.normal(0, 2))
        new.muni_fund_flows = s.muni_fund_flows + self.rng.normal(0, 0.5)
        new.dealer_inventory = max(5, s.dealer_inventory + self.rng.normal(0, 0.5))
        new.top_marginal_rate = s.top_marginal_rate
        new.taxable_muni_spread = s.taxable_muni_spread + self.rng.normal(0, 2)
        new.mub_premium = s.mub_premium + self.rng.normal(0, 0.1)
        new.salt_cap = s.salt_cap

        return new


class MunisEnv(gym.Env):
    """
    Gymnasium environment for municipal bond trading desk.

    Action space: MultiDiscrete([13, 8, 6, 10])
      - action_type (13): NOOP through CLOSE_DAY
      - state_idx (8): which state
      - maturity_idx (6): which maturity point
      - size_bucket (10): position size

    Observation: 112-dim
      - Market (48) + Book (64)

    Reward: Daily P&L from tax-advantaged carry + ratio moves + new issue
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 20,
        max_actions_per_day: int = 10,
        gross_limit: float = 500.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.gross_limit = gross_limit
        self.gen = MuniScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([13, 8, 6, 10])

        self._market_dim = 48
        self._book_dim = 64
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._size_buckets = np.linspace(2, 50, 10)

        self.scenario: Optional[MuniScenario] = None
        self.prev_scenario: Optional[MuniScenario] = None
        self.book: Optional[MuniBook] = None
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
            self.gen = MuniScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = MuniBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        act = MuniAction(action[0])
        state_idx = min(action[1], N_STATES - 1)
        mat_idx = min(action[2], N_MATURITIES - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == MuniAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == MuniAction.NOOP:
            pass
        elif act == MuniAction.BUY_TE:
            reward = self._trade_te(state_idx, mat_idx, size)
        elif act == MuniAction.SELL_TE:
            reward = self._trade_te(state_idx, mat_idx, -size)
        elif act == MuniAction.BUY_TAXABLE:
            reward = self._trade_taxable(mat_idx, size)
        elif act == MuniAction.SELL_TAXABLE:
            reward = self._trade_taxable(mat_idx, -size)
        elif act == MuniAction.RATIO_TRADE:
            reward = self._ratio_trade(mat_idx, size)
        elif act == MuniAction.REVERSE_RATIO:
            reward = self._ratio_trade(mat_idx, -size)
        elif act == MuniAction.BID_NEW_ISSUE:
            reward = self._bid_new_issue(state_idx, mat_idx, size)
        elif act == MuniAction.HEDGE_TSY:
            reward = self._hedge_tsy(mat_idx, size)
        elif act == MuniAction.HEDGE_ETF:
            reward = self._hedge_etf(size)
        elif act == MuniAction.STATE_ROTATE:
            reward = self._state_rotate(state_idx, mat_idx, size)
        elif act == MuniAction.FLATTEN:
            reward = self._flatten()

        if self.book.gross_exposure > self.gross_limit:
            reward -= 3.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {"day": self.day, "gross": self.book.gross_exposure,
                "realized_pnl": self.book.realized_pnl}
        return self._get_obs(), reward, terminated, truncated, info

    def _trade_te(self, state: int, mat: int, size: float) -> float:
        self.book.te_positions[state, mat] += size
        # Munis: wide bid/ask (Green et al 2007 ~1-2pts for retail, 0.5 institutional)
        bid_ask = 0.5 + MAT_POINTS[mat] * 0.03
        return -abs(size) * bid_ask / 100 * 1000

    def _trade_taxable(self, mat: int, size: float) -> float:
        self.book.taxable_positions[mat] += size
        bid_ask = 0.4 + MAT_POINTS[mat] * 0.02
        return -abs(size) * bid_ask / 100 * 1000

    def _ratio_trade(self, mat: int, size: float) -> float:
        # Long muni, short Treasury (size > 0 means bet on ratio tightening)
        self.book.te_positions[0, mat] += size  # Use CA as proxy
        self.book.tsy_hedge[mat] -= size
        cost = abs(size) * 0.6 / 100 * 1000
        return -cost

    def _bid_new_issue(self, state: int, mat: int, size: float) -> float:
        if self.scenario.new_issue_supply < 2:
            return -0.1
        alloc = size * self.gen.rng.uniform(0.3, 0.9)
        self.book.te_positions[state, mat] += alloc
        # New issue concession ~5-15bps
        concession = self.rng.uniform(3, 12) if hasattr(self, 'rng') else 7.0
        pnl = alloc * concession / 10000 * MAT_POINTS[mat] * 10
        return float(pnl)

    def _hedge_tsy(self, mat: int, size: float) -> float:
        self.book.tsy_hedge[mat] += size
        return -abs(size) * 0.2 / 100 * 1000

    def _hedge_etf(self, size: float) -> float:
        self.book.etf_hedge += size
        return -abs(size) * 0.05 / 100 * 1000

    def _state_rotate(self, state: int, mat: int, size: float) -> float:
        # Find cheapest state (widest spread)
        spreads = self.scenario.state_spreads[:, 1]  # AA quality
        buy_state = int(np.argmax(spreads))
        if buy_state == state:
            buy_state = (state + 1) % N_STATES
        self.book.te_positions[state, mat] -= size
        self.book.te_positions[buy_state, mat] += size
        cost = abs(size) * 1.0 / 100 * 1000  # 2 legs
        return -cost

    def _flatten(self) -> float:
        cost = 0.0
        for i in range(N_STATES):
            for j in range(N_MATURITIES):
                if abs(self.book.te_positions[i, j]) > 0.1:
                    cost += abs(self.book.te_positions[i, j]) * 0.5 / 100 * 1000
                    self.book.te_positions[i, j] *= 0.1
        for j in range(N_MATURITIES):
            cost += abs(self.book.taxable_positions[j]) * 0.4 / 100 * 1000
            self.book.taxable_positions[j] *= 0.1
        self.book.tsy_hedge *= 0.1
        self.book.etf_hedge *= 0.1
        return -cost

    def _end_of_day(self) -> float:
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        daily_pnl = 0.0

        # Carry: tax-exempt yield (multiply by 1/(1-tax_rate) for pre-tax equivalent)
        tax_mult = 1.0 / (1.0 - self.prev_scenario.top_marginal_rate / 100.0)
        for i in range(N_STATES):
            for j in range(N_MATURITIES):
                pos = self.book.te_positions[i, j]
                if abs(pos) > 0.01:
                    te_yield = self.prev_scenario.mmd_curve[j] + \
                        self.prev_scenario.state_spreads[i, 1] / 100
                    # Tax-equivalent carry advantage
                    daily_carry = pos * te_yield * tax_mult / 100 / 252 * 1000
                    daily_pnl += daily_carry

        # Taxable muni carry
        for j in range(N_MATURITIES):
            pos = self.book.taxable_positions[j]
            if abs(pos) > 0.01:
                txbl_yield = self.prev_scenario.tsy_curve[j] + \
                    self.prev_scenario.taxable_muni_spread / 100
                daily_pnl += pos * txbl_yield / 100 / 252 * 1000

        # Treasury hedge carry (negative - paying tsy yield)
        for j in range(N_MATURITIES):
            pos = self.book.tsy_hedge[j]
            if abs(pos) > 0.01:
                daily_pnl -= pos * self.prev_scenario.tsy_curve[j] / 100 / 252 * 1000

        self.book.carry_earned += daily_pnl

        # MTM from yield/ratio changes
        mtm = 0.0
        for i in range(N_STATES):
            for j in range(N_MATURITIES):
                pos = self.book.te_positions[i, j]
                if abs(pos) > 0.01:
                    old_y = self.prev_scenario.mmd_curve[j] + \
                        self.prev_scenario.state_spreads[i, 1] / 100
                    new_y = self.scenario.mmd_curve[j] + \
                        self.scenario.state_spreads[i, 1] / 100
                    dur = MAT_POINTS[j] * 0.9
                    price_change = -(new_y - old_y) * dur  # points
                    mtm += pos * price_change / 100 * 1000

        # Tsy hedge MTM
        for j in range(N_MATURITIES):
            pos = self.book.tsy_hedge[j]
            if abs(pos) > 0.01:
                dy = self.scenario.tsy_curve[j] - self.prev_scenario.tsy_curve[j]
                dur = MAT_POINTS[j] * 0.95
                mtm -= pos * dy * dur / 100 * 1000

        self.book.unrealized_pnl += mtm
        daily_pnl += mtm
        self.book.realized_pnl += daily_pnl

        return float(np.clip(daily_pnl, -50, 50))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        print(f"MMD 10Y: {self.scenario.mmd_curve[2]:.2f}% | "
              f"Ratio: {self.scenario.muni_tsy_ratios[2]:.1%}")
        print(f"Gross: ${self.book.gross_exposure:.0f}M | "
              f"P&L: ${self.book.realized_pnl:.0f}K")


def make_munis_env(seed: Optional[int] = None, **kwargs) -> MunisEnv:
    return MunisEnv(seed=seed, **kwargs)
