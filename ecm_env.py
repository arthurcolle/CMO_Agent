"""
ECM (Equity Capital Markets) / IPO Desk RL Environment.

Covers:
  - IPO pricing and book-building
  - Allocation optimization (institutional vs retail)
  - Greenshoe option management
  - Aftermarket stabilization
  - Follow-on offerings (FPO/secondary)
  - ATM (at-the-market) programs
  - Block trades / accelerated bookbuilds
  - SPAC / de-SPAC transactions

References:
  - Ritter (2003) "Investment Banking and Securities Issuance" Handbook of Economics of Finance
  - Ljungqvist (2007) "IPO Underpricing" Handbook of Empirical Corporate Finance
  - Benveniste & Spindt (1989) "How Investment Bankers Determine the Offer Price" JFE
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


N_DEALS = 5  # Concurrent deals in pipeline
N_INVESTOR_TYPES = 4  # Long-only, Hedge fund, Retail, Sovereign wealth


@dataclass
class ECMScenario:
    """Market state for ECM desk."""
    # Pipeline
    deal_sizes_m: np.ndarray = field(default_factory=lambda: np.full(N_DEALS, 500.0))
    deal_sectors: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS, dtype=int))
    deal_stages: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS, dtype=int))
    # 0=origination, 1=due_diligence, 2=marketing, 3=bookbuild, 4=pricing, 5=aftermarket
    # Book demand (x oversubscription by investor type)
    book_demand: np.ndarray = field(
        default_factory=lambda: np.ones((N_DEALS, N_INVESTOR_TYPES)))
    # Market conditions
    spx_return_1m: float = 2.0  # %
    ipo_index_return_1m: float = 3.0  # %
    vix: float = 18.0
    avg_first_day_return: float = 15.0  # % (IPO pop)
    pipeline_volume_bn: float = 10.0  # Monthly ECM volume
    # Competitor deals
    deals_in_market: int = 5
    # Fee environment (bps)
    avg_ipo_fee: float = 700.0  # 7% for IPOs
    avg_fpo_fee: float = 350.0  # 3.5% for follow-ons
    avg_block_fee: float = 200.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.deal_sizes_m / 5000.0,                # 5
            self.deal_stages.astype(float) / 5.0,      # 5
            self.book_demand.flatten() / 10.0,         # 20
            [self.spx_return_1m / 10.0],               # 1
            [self.ipo_index_return_1m / 20.0],         # 1
            [self.vix / 50.0],                         # 1
            [self.avg_first_day_return / 50.0],        # 1
            [self.pipeline_volume_bn / 30.0],          # 1
            [self.deals_in_market / 15.0],             # 1
            [self.avg_ipo_fee / 1000.0],               # 1
        ])  # Total: 37

    @property
    def obs_dim(self) -> int:
        return 37


@dataclass
class ECMBook:
    """Position book for ECM desk."""
    # Revenue pipeline ($M)
    fee_pipeline: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS))
    # Greenshoe position ($M face)
    greenshoe_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS))
    # Stabilization cost ($M)
    stabilization_cost: float = 0.0
    # Realized fees ($M)
    realized_fees: float = 0.0
    # Trading P&L from greenshoe/stabilization ($M)
    trading_pnl: float = 0.0
    # Deals completed
    deals_completed: int = 0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.fee_pipeline / 50.0,              # 5
            self.greenshoe_positions / 100.0,      # 5
            [self.stabilization_cost / 20.0],      # 1
            [self.realized_fees / 100.0],          # 1
            [self.trading_pnl / 50.0],             # 1
            [self.deals_completed / 20.0],         # 1
        ])  # Total: 14

    @property
    def state_dim(self) -> int:
        return 14


class ECMAction(IntEnum):
    NOOP = 0
    ADVANCE_DEAL = 1       # Move deal to next stage
    SET_PRICE_HIGH = 2     # Price at top of range
    SET_PRICE_MID = 3      # Price at midpoint
    SET_PRICE_LOW = 4      # Price at bottom (leave money on table)
    ALLOC_INSTITUTIONAL = 5  # Heavy institutional allocation
    ALLOC_BALANCED = 6     # Balanced allocation
    ALLOC_RETAIL = 7       # Heavy retail
    EXERCISE_GREENSHOE = 8  # Exercise overallotment
    STABILIZE = 9           # Aftermarket stabilization (buy stock)
    PULL_DEAL = 10          # Withdraw deal from market
    CLOSE_WEEK = 11


class ECMScenarioGenerator:
    REGIMES = ["hot_market", "normal", "cold_market", "window_open", "risk_off"]

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime=None):
        if regime is None: regime = self.rng.choice(self.REGIMES)
        s = ECMScenario(regime=regime)

        s.deal_sizes_m = self.rng.uniform(100, 3000, N_DEALS)
        s.deal_sectors = self.rng.randint(0, 11, N_DEALS)
        s.deal_stages = self.rng.randint(0, 6, N_DEALS)

        if regime == "hot_market":
            s.book_demand = self.rng.uniform(3, 15, (N_DEALS, N_INVESTOR_TYPES))
            s.avg_first_day_return = self.rng.uniform(15, 40)
            s.vix = self.rng.uniform(10, 18)
        elif regime == "cold_market":
            s.book_demand = self.rng.uniform(0.3, 2, (N_DEALS, N_INVESTOR_TYPES))
            s.avg_first_day_return = self.rng.uniform(-5, 5)
            s.vix = self.rng.uniform(22, 40)
        else:
            s.book_demand = self.rng.uniform(1, 6, (N_DEALS, N_INVESTOR_TYPES))
            s.avg_first_day_return = self.rng.uniform(5, 20)
            s.vix = self.rng.uniform(14, 25)

        s.spx_return_1m = self.rng.normal(1, 3)
        s.ipo_index_return_1m = self.rng.normal(2, 5)
        s.pipeline_volume_bn = self.rng.uniform(3, 25)
        s.deals_in_market = self.rng.randint(1, 12)

        return s

    def step_scenario(self, s):
        new = ECMScenario(regime=s.regime)
        new.deal_sizes_m = s.deal_sizes_m.copy()
        new.deal_sectors = s.deal_sectors.copy()
        new.deal_stages = s.deal_stages.copy()
        new.book_demand = np.maximum(0.1, s.book_demand + self.rng.normal(0, 0.3, s.book_demand.shape))
        new.spx_return_1m = s.spx_return_1m + self.rng.normal(0, 0.5)
        new.ipo_index_return_1m = s.ipo_index_return_1m + self.rng.normal(0, 0.8)
        new.vix = max(8, s.vix + self.rng.normal(0, 1))
        new.avg_first_day_return = max(-10, s.avg_first_day_return + self.rng.normal(0, 2))
        new.pipeline_volume_bn = max(1, s.pipeline_volume_bn + self.rng.normal(0, 1))
        new.deals_in_market = max(0, s.deals_in_market + self.rng.randint(-1, 2))
        new.avg_ipo_fee = s.avg_ipo_fee
        new.avg_fpo_fee = s.avg_fpo_fee
        new.avg_block_fee = s.avg_block_fee
        return new


class ECMEnv(gym.Env):
    """ECM/IPO desk. Action: MultiDiscrete([12, 5, 10])"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_weeks=26, max_actions_per_week=10, seed=None):
        super().__init__()
        self.max_weeks = max_weeks
        self.max_actions_per_week = max_actions_per_week
        self.gen = ECMScenarioGenerator(seed=seed)
        self.action_space = spaces.MultiDiscrete([12, 5, 10])
        self._market_dim = 37
        self._book_dim = 14
        self.observation_space = spaces.Box(-10, 10, (self._market_dim + self._book_dim,), np.float32)

        self.scenario: Optional[ECMScenario] = None
        self.book: Optional[ECMBook] = None
        self.week = 0; self.actions_this_week = 0

    def _get_obs(self):
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([self.scenario.to_observation(),
                               self.book.to_vector()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.gen = ECMScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.book = ECMBook()
        self.week = 0; self.actions_this_week = 0
        return self._get_obs(), {"week": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None
        act = ECMAction(action[0])
        deal = min(action[1], N_DEALS - 1)
        size_idx = min(action[2], 9)

        reward = 0.0; terminated = False
        self.actions_this_week += 1

        if act == ECMAction.CLOSE_WEEK:
            reward = self._end_of_week()
            self.week += 1; self.actions_this_week = 0
            if self.week >= self.max_weeks: terminated = True
        elif act == ECMAction.ADVANCE_DEAL:
            if self.scenario.deal_stages[deal] < 5:
                self.scenario.deal_stages[deal] += 1
        elif act == ECMAction.SET_PRICE_HIGH:
            if self.scenario.deal_stages[deal] == 4:
                demand = float(np.mean(self.scenario.book_demand[deal]))
                fee = self.scenario.deal_sizes_m[deal] * self.scenario.avg_ipo_fee / 10000
                if demand > 3:
                    self.book.fee_pipeline[deal] = fee
                    self.scenario.deal_stages[deal] = 5
                    reward = fee * 10  # $M -> reward units
                else:
                    reward = -5  # Overpriced, deal may break
        elif act == ECMAction.SET_PRICE_MID:
            if self.scenario.deal_stages[deal] == 4:
                fee = self.scenario.deal_sizes_m[deal] * self.scenario.avg_ipo_fee / 10000
                self.book.fee_pipeline[deal] = fee * 0.9
                self.scenario.deal_stages[deal] = 5
                reward = fee * 0.9 * 8
        elif act == ECMAction.SET_PRICE_LOW:
            if self.scenario.deal_stages[deal] == 4:
                fee = self.scenario.deal_sizes_m[deal] * self.scenario.avg_ipo_fee / 10000
                self.book.fee_pipeline[deal] = fee * 0.8
                self.scenario.deal_stages[deal] = 5
                reward = fee * 0.8 * 5  # Lower fee but safer
        elif act == ECMAction.EXERCISE_GREENSHOE:
            if self.scenario.deal_stages[deal] == 5:
                greenshoe = self.scenario.deal_sizes_m[deal] * 0.15
                first_day = self.scenario.avg_first_day_return / 100
                self.book.greenshoe_positions[deal] = greenshoe
                reward = greenshoe * first_day * 10
        elif act == ECMAction.STABILIZE:
            cost = self.scenario.deal_sizes_m[deal] * 0.01
            self.book.stabilization_cost += cost
            reward = -cost
        elif act == ECMAction.PULL_DEAL:
            self.scenario.deal_stages[deal] = 0
            self.book.fee_pipeline[deal] = 0
            reward = -2  # Reputational cost

        if self.actions_this_week >= self.max_actions_per_week:
            reward += self._end_of_week()
            self.week += 1; self.actions_this_week = 0
            if self.week >= self.max_weeks: terminated = True

        return self._get_obs(), reward, terminated, False, {
            "week": self.week, "fees": self.book.realized_fees,
            "deals": self.book.deals_completed}

    def _end_of_week(self):
        if self.scenario is None or self.book is None: return 0.0
        self.scenario = self.gen.step_scenario(self.scenario)
        pnl = 0.0
        # Collect fees from completed deals
        for i in range(N_DEALS):
            if self.scenario.deal_stages[i] == 5 and self.book.fee_pipeline[i] > 0:
                self.book.realized_fees += self.book.fee_pipeline[i]
                pnl += self.book.fee_pipeline[i]
                self.book.deals_completed += 1
                self.book.fee_pipeline[i] = 0
                # Reset deal slot
                self.scenario.deal_sizes_m[i] = self.gen.rng.uniform(100, 3000)
                self.scenario.deal_stages[i] = 0
        pnl -= self.book.stabilization_cost * 0.1
        self.book.trading_pnl += pnl
        return float(np.clip(pnl, -50, 200))


def make_ecm_env(seed=None, **kwargs): return ECMEnv(seed=seed, **kwargs)
