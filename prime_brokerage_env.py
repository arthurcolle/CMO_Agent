"""
Prime Brokerage / Securities Lending Desk RL Environment.

Covers:
  - Margin lending (financing hedge fund positions)
  - Securities lending (hard-to-borrow, general collateral)
  - Synthetic prime (total return swaps)
  - Short interest monetization
  - Balance sheet optimization
  - Client risk management (margin calls, liquidation)

References:
  - Duffie, Garleanu & Pedersen (2002) "Securities Lending" JFE
  - Kolasinski, Reed & Ringgenberg (2013) "A Multiple Lender Approach" RFS
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


N_CLIENTS = 8
N_COLLATERAL_TYPES = 4  # Equity, UST, IG Corp, HY Corp
N_BORROW_TIERS = 3  # Easy, Medium, Hard-to-borrow


@dataclass
class PBScenario:
    """Market state for prime brokerage."""
    # Client balances ($B long, $B short)
    client_longs: np.ndarray = field(default_factory=lambda: np.zeros(N_CLIENTS))
    client_shorts: np.ndarray = field(default_factory=lambda: np.zeros(N_CLIENTS))
    client_margin_util: np.ndarray = field(default_factory=lambda: np.full(N_CLIENTS, 0.5))
    # Lending rates (bps annualized)
    gc_lending_rate: float = 25.0  # General collateral
    htb_rates: np.ndarray = field(default_factory=lambda: np.full(N_BORROW_TIERS, 50.0))
    # Rebate rates (bps)
    rebate_rate: float = 10.0
    # Margin rates
    margin_rate: float = 150.0  # bps over funding
    # Funding cost
    funding_rate: float = 5.25
    # Securities lending fee pool ($M/day)
    sec_lending_fee_pool: float = 2.0
    # Short interest in market (% of float)
    market_short_interest: float = 4.0
    # Utilization (% of lendable inventory out)
    utilization_rate: float = 0.65
    # VIX
    vix: float = 18.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.client_longs / 10.0,              # 8
            self.client_shorts / 5.0,              # 8
            self.client_margin_util,               # 8
            [self.gc_lending_rate / 100.0],        # 1
            self.htb_rates / 500.0,                # 3
            [self.rebate_rate / 50.0],             # 1
            [self.margin_rate / 500.0],            # 1
            [self.funding_rate / 10.0],            # 1
            [self.sec_lending_fee_pool / 10.0],    # 1
            [self.market_short_interest / 15.0],   # 1
            [self.utilization_rate],               # 1
            [self.vix / 50.0],                     # 1
        ])  # Total: 35

    @property
    def obs_dim(self) -> int:
        return 35


@dataclass
class PBBook:
    """Position book for prime brokerage."""
    # Margin loans outstanding ($B) by client
    margin_loans: np.ndarray = field(default_factory=lambda: np.zeros(N_CLIENTS))
    # Securities on loan ($B) by borrow tier
    sec_on_loan: np.ndarray = field(default_factory=lambda: np.zeros(N_BORROW_TIERS))
    # Synthetic prime (TRS notional, $B)
    trs_notional: float = 0.0
    # Cash collateral received ($B)
    cash_collateral: float = 0.0
    # P&L ($M)
    realized_pnl: float = 0.0
    fee_income: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.margin_loans / 10.0,              # 8
            self.sec_on_loan / 5.0,                # 3
            [self.trs_notional / 20.0],            # 1
            [self.cash_collateral / 10.0],         # 1
            [self.realized_pnl / 100.0],           # 1
            [self.fee_income / 50.0],              # 1
        ])  # Total: 15

    @property
    def state_dim(self) -> int:
        return 15

    @property
    def total_exposure(self) -> float:
        return float(np.sum(self.margin_loans) + np.sum(self.sec_on_loan) + abs(self.trs_notional))


class PBAction(IntEnum):
    NOOP = 0
    EXTEND_MARGIN = 1      # Extend margin loan to client
    RECALL_MARGIN = 2      # Recall margin
    LEND_SECURITIES = 3    # Put securities on loan
    RECALL_SECURITIES = 4  # Recall borrowed securities
    ENTER_TRS = 5          # Enter total return swap
    EXIT_TRS = 6
    RAISE_MARGIN_REQ = 7   # Increase margin requirement for client
    LOWER_MARGIN_REQ = 8
    MARGIN_CALL = 9        # Issue margin call to specific client
    OPTIMIZE_BS = 10       # Balance sheet optimization
    FLATTEN = 11
    CLOSE_DAY = 12


class PBScenarioGenerator:
    REGIMES = ["normal", "vol_spike", "squeeze", "deleveraging", "year_end"]

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime=None):
        if regime is None:
            regime = self.rng.choice(self.REGIMES)
        s = PBScenario(regime=regime)

        s.client_longs = self.rng.uniform(1, 8, N_CLIENTS)
        s.client_shorts = self.rng.uniform(0.5, 3, N_CLIENTS)

        if regime == "vol_spike":
            s.vix = self.rng.uniform(25, 50)
            s.client_margin_util = self.rng.uniform(0.6, 0.95, N_CLIENTS)
        elif regime == "squeeze":
            s.market_short_interest = self.rng.uniform(8, 15)
            s.htb_rates = self.rng.uniform(200, 1000, N_BORROW_TIERS)
        elif regime == "deleveraging":
            s.client_shorts *= 0.5
            s.client_margin_util = self.rng.uniform(0.8, 0.98, N_CLIENTS)
        else:
            s.vix = self.rng.uniform(12, 25)
            s.client_margin_util = self.rng.uniform(0.3, 0.7, N_CLIENTS)

        s.gc_lending_rate = self.rng.uniform(10, 50)
        s.htb_rates = np.sort(self.rng.uniform(30, 500, N_BORROW_TIERS))
        s.rebate_rate = self.rng.uniform(5, 30)
        s.margin_rate = self.rng.uniform(100, 300)
        s.funding_rate = self.rng.uniform(3, 6)
        s.sec_lending_fee_pool = self.rng.uniform(0.5, 5)
        s.market_short_interest = max(1, s.market_short_interest + self.rng.normal(0, 1))
        s.utilization_rate = self.rng.uniform(0.4, 0.85)

        return s

    def step_scenario(self, s):
        new = PBScenario(regime=s.regime)
        new.client_longs = np.maximum(0.1, s.client_longs + self.rng.normal(0, 0.2, N_CLIENTS))
        new.client_shorts = np.maximum(0.1, s.client_shorts + self.rng.normal(0, 0.1, N_CLIENTS))
        new.client_margin_util = np.clip(s.client_margin_util + self.rng.normal(0, 0.03, N_CLIENTS), 0.1, 0.99)
        new.gc_lending_rate = max(5, s.gc_lending_rate + self.rng.normal(0, 2))
        new.htb_rates = np.maximum(20, s.htb_rates + self.rng.normal(0, 10, N_BORROW_TIERS))
        new.rebate_rate = max(1, s.rebate_rate + self.rng.normal(0, 1))
        new.margin_rate = max(50, s.margin_rate + self.rng.normal(0, 5))
        new.funding_rate = s.funding_rate
        new.sec_lending_fee_pool = max(0.1, s.sec_lending_fee_pool + self.rng.normal(0, 0.2))
        new.market_short_interest = max(1, s.market_short_interest + self.rng.normal(0, 0.3))
        new.utilization_rate = np.clip(s.utilization_rate + self.rng.normal(0, 0.02), 0.2, 0.95)
        new.vix = max(8, s.vix + self.rng.normal(0, 1))
        return new


class PrimeBrokerageEnv(gym.Env):
    """Prime brokerage desk. Action: MultiDiscrete([13, 8, 3, 10])"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_days=30, max_actions_per_day=10,
                 exposure_limit=50.0, seed=None):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.exposure_limit = exposure_limit
        self.gen = PBScenarioGenerator(seed=seed)
        self.action_space = spaces.MultiDiscrete([13, 8, 3, 10])
        self._market_dim = 35
        self._book_dim = 15
        self.observation_space = spaces.Box(-10, 10, (self._market_dim + self._book_dim,), np.float32)
        self._size_buckets = np.linspace(0.1, 3.0, 10)

        self.scenario: Optional[PBScenario] = None
        self.book: Optional[PBBook] = None
        self.day = 0
        self.actions_today = 0

    def _get_obs(self):
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([self.scenario.to_observation(),
                               self.book.to_vector()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.gen = PBScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.book = PBBook()
        self.day = 0; self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None
        act = PBAction(action[0])
        client = min(action[1], N_CLIENTS - 1)
        tier = min(action[2], N_BORROW_TIERS - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0; terminated = False
        self.actions_today += 1

        if act == PBAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1; self.actions_today = 0
            if self.day >= self.max_days: terminated = True
        elif act == PBAction.NOOP: pass
        elif act == PBAction.EXTEND_MARGIN:
            self.book.margin_loans[client] += size
        elif act == PBAction.RECALL_MARGIN:
            recalled = min(size, self.book.margin_loans[client])
            self.book.margin_loans[client] -= recalled
        elif act == PBAction.LEND_SECURITIES:
            self.book.sec_on_loan[tier] += size
        elif act == PBAction.RECALL_SECURITIES:
            recalled = min(size, self.book.sec_on_loan[tier])
            self.book.sec_on_loan[tier] -= recalled
        elif act == PBAction.ENTER_TRS:
            self.book.trs_notional += size
        elif act == PBAction.EXIT_TRS:
            self.book.trs_notional = max(0, self.book.trs_notional - size)
        elif act == PBAction.MARGIN_CALL:
            if self.scenario.client_margin_util[client] > 0.8:
                recovered = size * 0.5
                self.book.margin_loans[client] = max(0, self.book.margin_loans[client] - recovered)
                self.book.cash_collateral += recovered
                reward = recovered * 10  # $M reward
        elif act == PBAction.FLATTEN:
            self.book.margin_loans *= 0.5
            self.book.sec_on_loan *= 0.5
            self.book.trs_notional *= 0.5

        if self.book.total_exposure > self.exposure_limit: reward -= 3.0
        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1; self.actions_today = 0
            if self.day >= self.max_days: terminated = True

        return self._get_obs(), reward, terminated, False, {
            "day": self.day, "exposure": self.book.total_exposure,
            "fee_income": self.book.fee_income}

    def _end_of_day(self):
        if self.scenario is None or self.book is None: return 0.0
        prev = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        pnl = 0.0

        # Margin lending income
        for i in range(N_CLIENTS):
            daily_income = self.book.margin_loans[i] * prev.margin_rate / 10000 / 365 * 1000
            pnl += daily_income

        # Securities lending income
        for t in range(N_BORROW_TIERS):
            rate = prev.htb_rates[t] if t > 0 else prev.gc_lending_rate
            daily_income = self.book.sec_on_loan[t] * rate / 10000 / 365 * 1000
            pnl += daily_income

        # TRS income (financing spread scales with funding environment)
        trs_spread = prev.margin_rate * 0.3  # TRS spread ~30% of margin rate
        pnl += self.book.trs_notional * trs_spread / 10000 / 365 * 1000

        # Funding cost
        total_funded = float(np.sum(self.book.margin_loans)) + float(np.sum(self.book.sec_on_loan))
        funding_cost = total_funded * prev.funding_rate / 100 / 365 * 1000
        pnl -= funding_cost

        # Client default risk (low prob, high impact)
        for i in range(N_CLIENTS):
            if self.scenario.client_margin_util[i] > 0.95 and self.gen.rng.random() < 0.01:
                loss = self.book.margin_loans[i] * 0.1 * 1000
                pnl -= loss

        self.book.fee_income += max(0, pnl)
        self.book.realized_pnl += pnl
        return float(np.clip(pnl, -50, 50))


def make_prime_brokerage_env(seed=None, **kwargs): return PrimeBrokerageEnv(seed=seed, **kwargs)
