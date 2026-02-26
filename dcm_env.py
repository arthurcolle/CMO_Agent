"""
DCM (Debt Capital Markets) / Leveraged Finance Desk RL Environment.

Covers:
  - Investment-grade bond underwriting
  - Leveraged loans (Term Loan B)
  - High-yield bond underwriting
  - CLO warehouse / securitization
  - Bridge financing
  - Market flex (repricing risk)
  - Syndication and distribution
  - Private placements

References:
  - Ivashina & Sun (2011) "Institutional Demand Pressure and the Cost of Corporate Loans" JFE
  - Benmelech, Dlugosz & Ivashina (2012) "Securitization without Adverse Selection" JFE
  - Bruche, Malherbe & Meisenzahl (2020) "Pipeline Risk in Leveraged Loan Syndication" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


N_DEALS = 6  # Pipeline capacity
N_DEAL_TYPES = 4  # IG bond, HY bond, Lev loan, CLO


@dataclass
class DCMScenario:
    """Market state for DCM desk."""
    # Pipeline
    deal_sizes_m: np.ndarray = field(default_factory=lambda: np.full(N_DEALS, 500.0))
    deal_types: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS, dtype=int))
    deal_stages: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS, dtype=int))
    deal_spreads: np.ndarray = field(default_factory=lambda: np.full(N_DEALS, 200.0))
    # Market
    ig_spread: float = 100.0
    hy_spread: float = 400.0
    lev_loan_spread: float = 350.0
    clo_spread: float = 150.0
    # Supply/demand
    weekly_issuance_bn: float = 15.0
    clo_creation_pace_bn: float = 5.0  # Monthly
    bank_appetite: float = 0.7  # 0-1
    # Flex risk
    flex_risk: float = 0.3  # Probability of adverse flex
    avg_flex_bps: float = 25.0
    # Bridge commitments ($M)
    bridge_outstanding: float = 0.0
    # Fees
    ig_fee_bps: float = 65.0
    hy_fee_bps: float = 200.0
    loan_fee_bps: float = 250.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.deal_sizes_m / 5000.0,            # 6
            self.deal_types.astype(float) / 4.0,   # 6
            self.deal_stages.astype(float) / 5.0,  # 6
            self.deal_spreads / 1000.0,            # 6
            [self.ig_spread / 300.0],              # 1
            [self.hy_spread / 1000.0],             # 1
            [self.lev_loan_spread / 800.0],        # 1
            [self.clo_spread / 500.0],             # 1
            [self.weekly_issuance_bn / 40.0],      # 1
            [self.clo_creation_pace_bn / 15.0],    # 1
            [self.bank_appetite],                   # 1
            [self.flex_risk],                       # 1
            [self.avg_flex_bps / 100.0],           # 1
            [self.bridge_outstanding / 5000.0],    # 1
        ])  # Total: 34

    @property
    def obs_dim(self) -> int:
        return 34


@dataclass
class DCMBook:
    fee_pipeline: np.ndarray = field(default_factory=lambda: np.zeros(N_DEALS))
    warehouse: float = 0.0  # CLO warehouse ($M)
    bridge_exposure: float = 0.0  # Bridge financing ($M)
    realized_fees: float = 0.0
    syndication_pnl: float = 0.0
    deals_completed: int = 0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.fee_pipeline / 50.0,              # 6
            [self.warehouse / 2000.0],             # 1
            [self.bridge_exposure / 3000.0],       # 1
            [self.realized_fees / 200.0],          # 1
            [self.syndication_pnl / 100.0],        # 1
            [self.deals_completed / 30.0],         # 1
        ])  # Total: 11

    @property
    def state_dim(self) -> int:
        return 11


class DCMAction(IntEnum):
    NOOP = 0
    ADVANCE_DEAL = 1
    PRICE_TIGHT = 2       # Price at tight end (borrower-friendly)
    PRICE_MID = 3
    PRICE_WIDE = 4        # Price at wide end (investor-friendly)
    SYNDICATE = 5          # Distribute risk to other banks
    FLEX_UP = 6            # Invoke upward flex (widen spread)
    FLEX_DOWN = 7          # Offer reverse flex (tighten)
    BUILD_CLO = 8          # Accumulate CLO warehouse
    SELL_CLO = 9           # Issue CLO from warehouse
    BRIDGE_COMMIT = 10     # Provide bridge financing
    PULL_DEAL = 11
    CLOSE_WEEK = 12


class DCMScenarioGenerator:
    REGIMES = ["normal", "leveraged_boom", "credit_crunch", "clo_wave", "risk_off"]

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime=None):
        if regime is None: regime = self.rng.choice(self.REGIMES)
        s = DCMScenario(regime=regime)

        s.deal_sizes_m = self.rng.uniform(200, 5000, N_DEALS)
        s.deal_types = self.rng.randint(0, N_DEAL_TYPES, N_DEALS)
        s.deal_stages = self.rng.randint(0, 5, N_DEALS)

        if regime == "leveraged_boom":
            s.hy_spread = self.rng.uniform(250, 400)
            s.lev_loan_spread = self.rng.uniform(250, 350)
            s.weekly_issuance_bn = self.rng.uniform(15, 30)
            s.flex_risk = 0.1
            s.bank_appetite = 0.85
        elif regime == "credit_crunch":
            s.hy_spread = self.rng.uniform(500, 900)
            s.lev_loan_spread = self.rng.uniform(500, 700)
            s.weekly_issuance_bn = self.rng.uniform(2, 8)
            s.flex_risk = 0.7
            s.bank_appetite = 0.3
        elif regime == "clo_wave":
            s.clo_creation_pace_bn = self.rng.uniform(8, 15)
            s.clo_spread = self.rng.uniform(100, 200)
        else:
            s.hy_spread = self.rng.uniform(350, 550)
            s.lev_loan_spread = self.rng.uniform(300, 450)

        for i in range(N_DEALS):
            base = [s.ig_spread, s.hy_spread, s.lev_loan_spread, s.clo_spread][s.deal_types[i]]
            s.deal_spreads[i] = base + self.rng.normal(0, 20)

        return s

    def step_scenario(self, s):
        new = DCMScenario(regime=s.regime)
        new.deal_sizes_m = s.deal_sizes_m.copy()
        new.deal_types = s.deal_types.copy()
        new.deal_stages = s.deal_stages.copy()
        new.deal_spreads = np.maximum(50, s.deal_spreads + self.rng.normal(0, 5, N_DEALS))
        new.ig_spread = max(30, s.ig_spread + self.rng.normal(0, 3))
        new.hy_spread = max(100, s.hy_spread + self.rng.normal(0, 10))
        new.lev_loan_spread = max(100, s.lev_loan_spread + self.rng.normal(0, 8))
        new.clo_spread = max(50, s.clo_spread + self.rng.normal(0, 5))
        new.weekly_issuance_bn = max(1, s.weekly_issuance_bn + self.rng.normal(0, 2))
        new.clo_creation_pace_bn = max(1, s.clo_creation_pace_bn + self.rng.normal(0, 0.5))
        new.bank_appetite = np.clip(s.bank_appetite + self.rng.normal(0, 0.03), 0.1, 0.95)
        new.flex_risk = np.clip(s.flex_risk + self.rng.normal(0, 0.05), 0.05, 0.9)
        new.avg_flex_bps = max(5, s.avg_flex_bps + self.rng.normal(0, 3))
        new.bridge_outstanding = s.bridge_outstanding
        new.ig_fee_bps = s.ig_fee_bps; new.hy_fee_bps = s.hy_fee_bps; new.loan_fee_bps = s.loan_fee_bps
        return new


class DCMEnv(gym.Env):
    """DCM/LevFin desk. Action: MultiDiscrete([13, 6, 10])"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_weeks=26, max_actions_per_week=12, seed=None):
        super().__init__()
        self.max_weeks = max_weeks
        self.max_actions_per_week = max_actions_per_week
        self.gen = DCMScenarioGenerator(seed=seed)
        self.action_space = spaces.MultiDiscrete([13, 6, 10])
        self._market_dim = 34
        self._book_dim = 11
        self.observation_space = spaces.Box(-10, 10, (self._market_dim + self._book_dim,), np.float32)
        self._size_buckets = np.linspace(50, 1000, 10)

        self.scenario: Optional[DCMScenario] = None
        self.book: Optional[DCMBook] = None
        self.week = 0; self.actions_this_week = 0

    def _get_obs(self):
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([self.scenario.to_observation(),
                               self.book.to_vector()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.gen = DCMScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.book = DCMBook()
        self.week = 0; self.actions_this_week = 0
        return self._get_obs(), {"week": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None
        act = DCMAction(action[0])
        deal = min(action[1], N_DEALS - 1)
        size_idx = min(action[2], 9)
        size = float(self._size_buckets[size_idx])

        reward = 0.0; terminated = False
        self.actions_this_week += 1

        if act == DCMAction.CLOSE_WEEK:
            reward = self._end_of_week()
            self.week += 1; self.actions_this_week = 0
            if self.week >= self.max_weeks: terminated = True
        elif act == DCMAction.ADVANCE_DEAL:
            if self.scenario.deal_stages[deal] < 5:
                self.scenario.deal_stages[deal] += 1
        elif act in (DCMAction.PRICE_TIGHT, DCMAction.PRICE_MID, DCMAction.PRICE_WIDE):
            if self.scenario.deal_stages[deal] >= 3:
                dt = self.scenario.deal_types[deal]
                fee_bps = [self.scenario.ig_fee_bps, self.scenario.hy_fee_bps,
                           self.scenario.loan_fee_bps, self.scenario.ig_fee_bps][dt]
                fee = self.scenario.deal_sizes_m[deal] * fee_bps / 10000
                # Tight pricing = higher fee risk, wider = lower
                disc = {DCMAction.PRICE_TIGHT: 1.0, DCMAction.PRICE_MID: 0.85,
                        DCMAction.PRICE_WIDE: 0.7}[act]
                if act == DCMAction.PRICE_TIGHT and self.gen.rng.random() < self.scenario.flex_risk:
                    reward = -fee * 0.3  # Flex event, lose money
                else:
                    self.book.fee_pipeline[deal] = fee * disc
                    self.scenario.deal_stages[deal] = 5
                    reward = fee * disc * 5
        elif act == DCMAction.SYNDICATE:
            if self.book.fee_pipeline[deal] > 0:
                syndicated = self.book.fee_pipeline[deal] * 0.5
                self.book.fee_pipeline[deal] *= 0.5
                self.book.syndication_pnl += syndicated * 0.2
                reward = syndicated * 0.2
        elif act == DCMAction.FLEX_UP:
            self.scenario.deal_spreads[deal] += self.scenario.avg_flex_bps
            reward = -1  # Market perceives weakness
        elif act == DCMAction.BUILD_CLO:
            self.book.warehouse += size
            reward = -size * 0.001  # Carry cost
        elif act == DCMAction.SELL_CLO:
            if self.book.warehouse > 200:
                arb = self.book.warehouse * (self.scenario.lev_loan_spread -
                       self.scenario.clo_spread) / 10000
                self.book.syndication_pnl += arb
                reward = arb * 10
                self.book.warehouse = 0
        elif act == DCMAction.BRIDGE_COMMIT:
            self.book.bridge_exposure += size
            fee = size * 0.01  # 1% commitment fee
            reward = fee
        elif act == DCMAction.PULL_DEAL:
            self.scenario.deal_stages[deal] = 0
            self.book.fee_pipeline[deal] = 0
            reward = -3

        if self.actions_this_week >= self.max_actions_per_week:
            reward += self._end_of_week()
            self.week += 1; self.actions_this_week = 0
            if self.week >= self.max_weeks: terminated = True

        return self._get_obs(), reward, terminated, False, {
            "week": self.week, "fees": self.book.realized_fees,
            "warehouse": self.book.warehouse, "deals": self.book.deals_completed}

    def _end_of_week(self):
        if self.scenario is None or self.book is None: return 0.0
        self.scenario = self.gen.step_scenario(self.scenario)
        pnl = 0.0
        for i in range(N_DEALS):
            if self.scenario.deal_stages[i] == 5 and self.book.fee_pipeline[i] > 0:
                self.book.realized_fees += self.book.fee_pipeline[i]
                pnl += self.book.fee_pipeline[i]
                self.book.deals_completed += 1
                self.book.fee_pipeline[i] = 0
                self.scenario.deal_sizes_m[i] = self.gen.rng.uniform(200, 5000)
                self.scenario.deal_stages[i] = 0

        # Warehouse carry cost
        if self.book.warehouse > 0:
            carry = self.book.warehouse * self.scenario.lev_loan_spread / 10000 / 52
            pnl += carry * 0.5  # Net of funding

        # Bridge risk
        if self.book.bridge_exposure > 0:
            if self.gen.rng.random() < 0.05:
                loss = self.book.bridge_exposure * 0.02
                pnl -= loss
        self.book.syndication_pnl += pnl
        return float(np.clip(pnl, -100, 200))


def make_dcm_env(seed=None, **kwargs): return DCMEnv(seed=seed, **kwargs)
