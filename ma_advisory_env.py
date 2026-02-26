"""
M&A Advisory Desk RL Environment.

Covers:
  - Buy-side advisory (helping acquirers)
  - Sell-side advisory (running auctions)
  - Fairness opinions
  - Hostile defense (poison pills, white knights)
  - Merger arbitrage (proprietary)
  - LBO advisory
  - Cross-border M&A
  - Restructuring advisory

References:
  - DePamphilis (2019) "Mergers, Acquisitions, and Other Restructuring Activities"
  - Officer (2003) "Termination Fees in Mergers and Acquisitions" JFE
  - Mitchell & Pulvino (2001) "Characteristics of Risk and Return in Risk Arbitrage" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


N_MANDATES = 6  # Active advisory mandates
N_DEAL_TYPES = 5  # Strategic, PE buyout, hostile, restructuring, cross-border


@dataclass
class MAScenario:
    """Market state for M&A advisory."""
    # Mandate pipeline
    mandate_sizes_bn: np.ndarray = field(default_factory=lambda: np.full(N_MANDATES, 5.0))
    mandate_types: np.ndarray = field(default_factory=lambda: np.zeros(N_MANDATES, dtype=int))
    mandate_stages: np.ndarray = field(default_factory=lambda: np.zeros(N_MANDATES, dtype=int))
    # 0=pitch, 1=engaged, 2=diligence, 3=negotiation, 4=signing, 5=closed
    mandate_probabilities: np.ndarray = field(default_factory=lambda: np.full(N_MANDATES, 0.5))
    # Market
    m_a_volume_bn_quarter: float = 200.0
    avg_premium: float = 30.0  # % over unaffected price
    deal_break_rate: float = 0.10  # % of announced deals that fail
    pe_dry_powder_bn: float = 1000.0
    spx_level: float = 4500.0
    credit_spread: float = 400.0  # HY spread for LBO financing
    # Fee environment
    advisory_fee_bps: float = 30.0  # ~30bps of deal value
    fairness_opinion_fee_m: float = 5.0
    # Competition
    league_table_rank: int = 5  # 1-10
    active_pitches: int = 10
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.mandate_sizes_bn / 50.0,              # 6
            self.mandate_types.astype(float) / 5.0,    # 6
            self.mandate_stages.astype(float) / 5.0,   # 6
            self.mandate_probabilities,                 # 6
            [self.m_a_volume_bn_quarter / 500.0],      # 1
            [self.avg_premium / 50.0],                 # 1
            [self.deal_break_rate / 0.3],              # 1
            [self.pe_dry_powder_bn / 2000.0],          # 1
            [self.spx_level / 6000.0],                 # 1
            [self.credit_spread / 1000.0],             # 1
            [self.advisory_fee_bps / 100.0],           # 1
            [self.league_table_rank / 10.0],           # 1
            [self.active_pitches / 20.0],              # 1
        ])  # Total: 33

    @property
    def obs_dim(self) -> int:
        return 33


@dataclass
class MABook:
    fee_pipeline: np.ndarray = field(default_factory=lambda: np.zeros(N_MANDATES))
    # Merger arb positions ($M notional)
    arb_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_MANDATES))
    realized_fees: float = 0.0
    arb_pnl: float = 0.0
    deals_closed: int = 0
    pitches_won: int = 0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.fee_pipeline / 100.0,             # 6
            self.arb_positions / 100.0,            # 6
            [self.realized_fees / 500.0],          # 1
            [self.arb_pnl / 100.0],               # 1
            [self.deals_closed / 20.0],            # 1
            [self.pitches_won / 30.0],             # 1
        ])  # Total: 16

    @property
    def state_dim(self) -> int:
        return 16


class MAAction(IntEnum):
    NOOP = 0
    PITCH_NEW = 1          # Pitch for new mandate
    ADVANCE_MANDATE = 2    # Move mandate forward
    NEGOTIATE_FEE = 3      # Negotiate higher fee
    FAIRNESS_OPINION = 4   # Provide fairness opinion
    RECOMMEND_BID = 5      # Advise client to bid higher
    RECOMMEND_HOLD = 6     # Advise client to hold/walk
    HOSTILE_DEFENSE = 7    # Deploy defense tactics
    MERGER_ARB_LONG = 8    # Proprietary merger arb (long target)
    MERGER_ARB_SHORT = 9   # Short acquirer
    CLOSE_MANDATE = 10     # Close out completed mandate
    CLOSE_MONTH = 11


class MAScenarioGenerator:
    REGIMES = ["normal", "m_a_boom", "m_a_drought", "hostile_wave", "pe_boom"]

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime=None):
        if regime is None: regime = self.rng.choice(self.REGIMES)
        s = MAScenario(regime=regime)

        s.mandate_sizes_bn = self.rng.uniform(0.5, 30, N_MANDATES)
        s.mandate_types = self.rng.randint(0, N_DEAL_TYPES, N_MANDATES)
        s.mandate_stages = self.rng.randint(0, 6, N_MANDATES)
        s.mandate_probabilities = self.rng.uniform(0.2, 0.8, N_MANDATES)

        if regime == "m_a_boom":
            s.m_a_volume_bn_quarter = self.rng.uniform(300, 600)
            s.pe_dry_powder_bn = self.rng.uniform(1200, 2000)
            s.deal_break_rate = 0.05
            s.avg_premium = self.rng.uniform(25, 45)
        elif regime == "m_a_drought":
            s.m_a_volume_bn_quarter = self.rng.uniform(50, 150)
            s.deal_break_rate = 0.20
            s.avg_premium = self.rng.uniform(15, 25)
        elif regime == "hostile_wave":
            s.avg_premium = self.rng.uniform(35, 60)
            s.mandate_types[::2] = 2  # hostile
        elif regime == "pe_boom":
            s.pe_dry_powder_bn = self.rng.uniform(1500, 2500)
            s.credit_spread = self.rng.uniform(300, 450)
            s.mandate_types[::2] = 1  # PE buyout

        s.spx_level = self.rng.uniform(3800, 5500)
        s.credit_spread = max(200, s.credit_spread + self.rng.normal(0, 50))
        s.league_table_rank = self.rng.randint(1, 11)
        s.active_pitches = self.rng.randint(3, 20)

        return s

    def step_scenario(self, s):
        new = MAScenario(regime=s.regime)
        new.mandate_sizes_bn = s.mandate_sizes_bn.copy()
        new.mandate_types = s.mandate_types.copy()
        new.mandate_stages = s.mandate_stages.copy()
        new.mandate_probabilities = np.clip(
            s.mandate_probabilities + self.rng.normal(0, 0.05, N_MANDATES), 0.05, 0.95)
        new.m_a_volume_bn_quarter = max(30, s.m_a_volume_bn_quarter + self.rng.normal(0, 20))
        new.avg_premium = max(10, s.avg_premium + self.rng.normal(0, 2))
        new.deal_break_rate = np.clip(s.deal_break_rate + self.rng.normal(0, 0.01), 0.02, 0.30)
        new.pe_dry_powder_bn = max(200, s.pe_dry_powder_bn + self.rng.normal(0, 30))
        new.spx_level = s.spx_level * (1 + self.rng.normal(0, 0.02))
        new.credit_spread = max(100, s.credit_spread + self.rng.normal(0, 10))
        new.advisory_fee_bps = s.advisory_fee_bps
        new.fairness_opinion_fee_m = s.fairness_opinion_fee_m
        new.league_table_rank = s.league_table_rank
        new.active_pitches = max(1, s.active_pitches + self.rng.randint(-2, 3))
        return new


class MAAdvisoryEnv(gym.Env):
    """M&A advisory desk. Action: MultiDiscrete([12, 6, 10])"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_months=12, max_actions_per_month=15, seed=None):
        super().__init__()
        self.max_months = max_months
        self.max_actions_per_month = max_actions_per_month
        self.gen = MAScenarioGenerator(seed=seed)
        self.action_space = spaces.MultiDiscrete([12, 6, 10])
        self._market_dim = 33
        self._book_dim = 16
        self.observation_space = spaces.Box(-10, 10, (self._market_dim + self._book_dim,), np.float32)

        self.scenario: Optional[MAScenario] = None
        self.book: Optional[MABook] = None
        self.month = 0; self.actions_this_month = 0

    def _get_obs(self):
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([self.scenario.to_observation(),
                               self.book.to_vector()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.gen = MAScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.book = MABook()
        self.month = 0; self.actions_this_month = 0
        return self._get_obs(), {"month": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None
        act = MAAction(action[0])
        mandate = min(action[1], N_MANDATES - 1)
        size_idx = min(action[2], 9)

        reward = 0.0; terminated = False
        self.actions_this_month += 1

        if act == MAAction.CLOSE_MONTH:
            reward = self._end_of_month()
            self.month += 1; self.actions_this_month = 0
            if self.month >= self.max_months: terminated = True
        elif act == MAAction.PITCH_NEW:
            # Win new mandate
            if self.gen.rng.random() < 0.3:
                # Find empty slot
                for i in range(N_MANDATES):
                    if self.scenario.mandate_stages[i] == 0:
                        self.scenario.mandate_stages[i] = 1
                        self.scenario.mandate_sizes_bn[i] = self.gen.rng.uniform(0.5, 20)
                        self.book.pitches_won += 1
                        reward = 2  # Won mandate
                        break
        elif act == MAAction.ADVANCE_MANDATE:
            if self.scenario.mandate_stages[mandate] < 5:
                self.scenario.mandate_stages[mandate] += 1
                reward = 1  # Progress
        elif act == MAAction.NEGOTIATE_FEE:
            if self.gen.rng.random() < 0.4:
                self.scenario.advisory_fee_bps *= 1.1
                reward = 1
            else:
                reward = -0.5  # Client pushback
        elif act == MAAction.FAIRNESS_OPINION:
            if self.scenario.mandate_stages[mandate] >= 3:
                self.book.fee_pipeline[mandate] += self.scenario.fairness_opinion_fee_m
                reward = self.scenario.fairness_opinion_fee_m
        elif act == MAAction.RECOMMEND_BID:
            prob = self.scenario.mandate_probabilities[mandate]
            if prob > 0.5:
                self.scenario.mandate_probabilities[mandate] += 0.1
                reward = 2
            else:
                reward = -1  # Bad advice
        elif act == MAAction.RECOMMEND_HOLD:
            self.scenario.mandate_probabilities[mandate] = min(0.95,
                self.scenario.mandate_probabilities[mandate] + 0.05)
        elif act == MAAction.HOSTILE_DEFENSE:
            if self.scenario.mandate_types[mandate] == 2:
                self.book.fee_pipeline[mandate] += 3  # Defense fees
                reward = 3
        elif act == MAAction.MERGER_ARB_LONG:
            arb_size = size_idx * 5 + 5
            self.book.arb_positions[mandate] += arb_size
            reward = -arb_size * 0.001  # Transaction cost
        elif act == MAAction.MERGER_ARB_SHORT:
            arb_size = size_idx * 5 + 5
            self.book.arb_positions[mandate] -= arb_size
            reward = -arb_size * 0.001
        elif act == MAAction.CLOSE_MANDATE:
            if self.scenario.mandate_stages[mandate] >= 4:
                fee = self.scenario.mandate_sizes_bn[mandate] * 1000 * \
                    self.scenario.advisory_fee_bps / 10000
                self.book.fee_pipeline[mandate] += fee
                self.scenario.mandate_stages[mandate] = 5
                reward = fee * 2

        if self.actions_this_month >= self.max_actions_per_month:
            reward += self._end_of_month()
            self.month += 1; self.actions_this_month = 0
            if self.month >= self.max_months: terminated = True

        return self._get_obs(), reward, terminated, False, {
            "month": self.month, "fees": self.book.realized_fees,
            "deals": self.book.deals_closed}

    def _end_of_month(self):
        if self.scenario is None or self.book is None: return 0.0
        self.scenario = self.gen.step_scenario(self.scenario)
        pnl = 0.0

        for i in range(N_MANDATES):
            if self.scenario.mandate_stages[i] == 5 and self.book.fee_pipeline[i] > 0:
                # Check if deal closes
                if self.gen.rng.random() < self.scenario.mandate_probabilities[i]:
                    self.book.realized_fees += self.book.fee_pipeline[i]
                    pnl += self.book.fee_pipeline[i]
                    self.book.deals_closed += 1
                    # Arb profit on close
                    if self.book.arb_positions[i] > 0:
                        arb_profit = self.book.arb_positions[i] * \
                            self.scenario.avg_premium / 100 * 0.1
                        self.book.arb_pnl += arb_profit
                        pnl += arb_profit
                else:
                    # Deal breaks
                    if self.book.arb_positions[i] > 0:
                        arb_loss = self.book.arb_positions[i] * 0.05
                        self.book.arb_pnl -= arb_loss
                        pnl -= arb_loss

                self.book.fee_pipeline[i] = 0
                self.book.arb_positions[i] = 0
                self.scenario.mandate_stages[i] = 0
                self.scenario.mandate_sizes_bn[i] = self.gen.rng.uniform(0.5, 20)

        return float(np.clip(pnl, -50, 300))


def make_ma_advisory_env(seed=None, **kwargs): return MAAdvisoryEnv(seed=seed, **kwargs)
