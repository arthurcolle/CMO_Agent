"""
Distressed Credit Trading Desk RL Environment.

Gymnasium-compatible environment for distressed/high-yield credit trading.
Covers:
  - CDS and cash bond trading
  - Recovery rate estimation and fulcrum security analysis
  - Capital structure arbitrage (senior vs sub)
  - Event-driven trading (restructuring, Ch11, exchange offers)
  - Index vs single-name basis
  - Correlation and contagion dynamics

References:
  - Altman & Kishore (1996) "Almost Everything You Wanted to Know About Recoveries"
  - Moyer (2004) "Distressed Debt Analysis"
  - Longstaff, Mithal & Neis (2005) "Corporate Yield Spreads" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Credit Universe ──────────────────────────────────────────────────────

N_CREDITS = 10   # Number of credits in the universe
N_SECTORS = 5    # Energy, Real Estate, Retail, Tech, Healthcare

SENIORITY_LEVELS = ["secured_1L", "secured_2L", "senior_unsecured", "subordinated", "equity"]
N_SENIORITY = len(SENIORITY_LEVELS)

# Altman recovery rates by seniority (mean, std)
RECOVERY_BY_SENIORITY = {
    "secured_1L": (0.65, 0.15),
    "secured_2L": (0.45, 0.18),
    "senior_unsecured": (0.35, 0.20),
    "subordinated": (0.20, 0.15),
    "equity": (0.05, 0.05),
}


@dataclass
class CreditEntity:
    """A single credit/issuer in the universe."""
    name: str
    sector: int  # 0-4
    rating: float = 3.0  # 0=D, 1=CCC, 2=B, 3=BB, 4=BBB, 5=A
    leverage: float = 4.0  # Debt/EBITDA
    interest_coverage: float = 3.0  # EBITDA/Interest
    cash_burn_months: float = 36.0  # Months until cash runs out (if negative EBITDA)
    # Spread levels by seniority (bps)
    spreads: np.ndarray = field(default_factory=lambda: np.zeros(N_SENIORITY))
    # Recovery estimates (%)
    recovery_est: np.ndarray = field(default_factory=lambda: np.zeros(N_SENIORITY))
    # Event flags
    has_near_maturity: bool = False  # Maturity wall within 12 months
    in_restructuring: bool = False
    covenant_breach: bool = False
    recent_downgrade: bool = False
    # Default probability (1yr, %)
    default_prob: float = 0.0


@dataclass
class DistressedScenario:
    """Market state for the distressed credit desk."""
    credits: list[CreditEntity] = field(default_factory=list)
    # Index levels
    hy_index_spread: float = 400.0     # HY CDX spread (bps)
    ig_index_spread: float = 100.0     # IG CDX spread (bps)
    # Macro
    vix: float = 20.0
    hy_default_rate: float = 3.0       # Trailing 12m default rate (%)
    recovery_rate_market: float = 40.0 # Market-implied recovery (%)
    risk_free_rate: float = 4.5        # Treasury 5Y (%)
    # Funding
    cds_bond_basis: float = -20.0      # Negative = bonds cheap vs CDS
    repo_haircut: float = 15.0         # Repo haircut for HY bonds (%)
    funding_spread: float = 150.0      # Desk funding cost (bps over risk-free)
    # Distress ratio
    distress_ratio: float = 5.0        # % of HY universe trading >1000bps

    def to_observation(self) -> np.ndarray:
        """Flatten market state to observation vector."""
        # Per-credit features: rating, leverage, coverage, spreads, recovery, events
        credit_vecs = []
        for c in self.credits:
            credit_vecs.append(np.concatenate([
                [c.rating / 5.0],
                [c.leverage / 15.0],
                [c.interest_coverage / 10.0],
                [c.cash_burn_months / 60.0],
                c.spreads / 3000.0,              # 5
                c.recovery_est / 100.0,          # 5
                [float(c.has_near_maturity)],
                [float(c.in_restructuring)],
                [float(c.covenant_breach)],
                [float(c.recent_downgrade)],
                [c.default_prob / 50.0],
            ]))  # 19 per credit

        credit_flat = np.concatenate(credit_vecs)  # 190

        market_vec = np.array([
            self.hy_index_spread / 1000.0,
            self.ig_index_spread / 300.0,
            self.vix / 50.0,
            self.hy_default_rate / 15.0,
            self.recovery_rate_market / 100.0,
            self.risk_free_rate / 10.0,
            self.cds_bond_basis / 100.0,
            self.repo_haircut / 50.0,
            self.funding_spread / 500.0,
            self.distress_ratio / 30.0,
        ])  # 10

        return np.concatenate([credit_flat, market_vec])

    @property
    def obs_dim(self) -> int:
        return N_CREDITS * 19 + 10  # 200


# ─── Position Book ────────────────────────────────────────────────────────

@dataclass
class DistressedBook:
    """Position book for distressed credit desk."""
    # Position by credit and seniority ($M face)
    positions: np.ndarray = field(
        default_factory=lambda: np.zeros((N_CREDITS, N_SENIORITY))
    )
    # CDS positions ($M notional, positive = protection bought)
    cds_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_CREDITS))
    # Index hedge ($M notional)
    index_hedge: float = 0.0
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    carry_earned: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.positions.flatten() / 50.0,  # 50
            self.cds_positions / 50.0,        # 10
            [self.index_hedge / 100.0],       # 1
            [self.realized_pnl / 5000.0],     # 1
            [self.unrealized_pnl / 5000.0],   # 1
            [self.carry_earned / 1000.0],     # 1
        ])  # Total: 64

    @property
    def state_dim(self) -> int:
        return 64

    @property
    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.positions)) + np.sum(np.abs(self.cds_positions)))

    @property
    def net_exposure(self) -> float:
        return float(np.sum(self.positions) - np.sum(self.cds_positions) - self.index_hedge)


# ─── Actions ──────────────────────────────────────────────────────────────

class DistressedAction(IntEnum):
    NOOP = 0
    BUY_BOND = 1           # Buy cash bond (specific credit + seniority)
    SELL_BOND = 2          # Sell cash bond
    BUY_CDS = 3            # Buy CDS protection
    SELL_CDS = 4           # Sell CDS protection
    CAP_STRUCT_ARB = 5     # Capital structure arb (long senior, short sub)
    REVERSE_CAP_ARB = 6   # Reverse cap struct arb
    BASIS_TRADE = 7        # Cash-CDS basis trade (buy bond, buy CDS)
    INDEX_HEDGE = 8        # Hedge with HY index
    TAKE_RECOVERY = 9      # Position for recovery event
    EVENT_TRADE = 10       # Trade around event (restructuring/covenant)
    FLATTEN = 11           # Reduce positions
    END_WEEK = 12          # End trading week (terminal)


# ─── Scenario Generator ──────────────────────────────────────────────────

class DistressedScenarioGenerator:
    """Generate realistic distressed credit scenarios."""

    SECTORS = ["Energy", "RealEstate", "Retail", "Tech", "Healthcare"]
    NAMES = [
        "ClearwaterEnergy", "SunsetRealty", "GalaxyRetail", "QubitTech",
        "MeridianHealth", "AzureOil", "VanguardProp", "NexusMall",
        "QuantumChip", "LifelinePharma",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self) -> DistressedScenario:
        scenario = DistressedScenario()

        # Market regime
        cycle = self.rng.choice(["benign", "stress", "crisis"])
        if cycle == "crisis":
            scenario.hy_index_spread = self.rng.uniform(600, 1200)
            scenario.vix = self.rng.uniform(25, 50)
            scenario.hy_default_rate = self.rng.uniform(5, 15)
            scenario.distress_ratio = self.rng.uniform(10, 30)
        elif cycle == "stress":
            scenario.hy_index_spread = self.rng.uniform(400, 700)
            scenario.vix = self.rng.uniform(18, 30)
            scenario.hy_default_rate = self.rng.uniform(3, 7)
            scenario.distress_ratio = self.rng.uniform(5, 15)
        else:
            scenario.hy_index_spread = self.rng.uniform(250, 450)
            scenario.vix = self.rng.uniform(12, 22)
            scenario.hy_default_rate = self.rng.uniform(1, 4)
            scenario.distress_ratio = self.rng.uniform(2, 8)

        scenario.ig_index_spread = scenario.hy_index_spread * self.rng.uniform(0.15, 0.30)
        scenario.risk_free_rate = self.rng.uniform(2.0, 5.5)
        scenario.cds_bond_basis = self.rng.normal(-20, 30)
        scenario.funding_spread = self.rng.uniform(80, 250)

        # Generate credits
        for i in range(N_CREDITS):
            sector = i % N_SECTORS
            credit = CreditEntity(
                name=self.NAMES[i],
                sector=sector,
            )

            # Mix of investment-grade-fallen and original HY
            is_distressed = self.rng.random() < 0.4
            if is_distressed:
                credit.rating = self.rng.uniform(0.0, 1.5)  # CCC or below
                credit.leverage = self.rng.uniform(6.0, 15.0)
                credit.interest_coverage = self.rng.uniform(0.3, 1.5)
                credit.cash_burn_months = self.rng.uniform(3, 24)
                credit.default_prob = self.rng.uniform(10, 50)
                credit.has_near_maturity = self.rng.random() < 0.4
                credit.in_restructuring = self.rng.random() < 0.2
                credit.covenant_breach = self.rng.random() < 0.3
            else:
                credit.rating = self.rng.uniform(2.0, 4.0)  # B to BBB
                credit.leverage = self.rng.uniform(2.0, 6.0)
                credit.interest_coverage = self.rng.uniform(2.0, 8.0)
                credit.cash_burn_months = self.rng.uniform(24, 60)
                credit.default_prob = self.rng.uniform(0.5, 5.0)
                credit.has_near_maturity = self.rng.random() < 0.1

            credit.recent_downgrade = self.rng.random() < 0.15

            # Generate spreads by seniority
            base_spread = 100 + (5.0 - credit.rating) * 200 + \
                credit.leverage * 30 + scenario.hy_index_spread * 0.3
            for j, seniority in enumerate(SENIORITY_LEVELS):
                multiplier = [0.5, 0.7, 1.0, 1.5, 3.0][j]
                credit.spreads[j] = max(50, base_spread * multiplier + \
                    self.rng.normal(0, 30))

            # Recovery estimates
            for j, seniority in enumerate(SENIORITY_LEVELS):
                mean_r, std_r = RECOVERY_BY_SENIORITY[seniority]
                # Adjust for leverage (more leverage = lower recovery)
                leverage_adj = max(0.5, 1.0 - (credit.leverage - 4.0) * 0.05)
                credit.recovery_est[j] = np.clip(
                    (mean_r * leverage_adj + self.rng.normal(0, std_r * 0.3)) * 100,
                    2.0, 95.0,
                )

            scenario.credits.append(credit)

        return scenario

    def step_scenario(self, scenario: DistressedScenario) -> DistressedScenario:
        """Evolve scenario by one week."""
        new = DistressedScenario()
        new.hy_index_spread = max(100, scenario.hy_index_spread + \
            self.rng.normal(0, 20))
        new.ig_index_spread = max(30, scenario.ig_index_spread + \
            self.rng.normal(0, 5))
        new.vix = max(10, scenario.vix + self.rng.normal(0, 2))
        new.hy_default_rate = max(0.5, scenario.hy_default_rate + \
            self.rng.normal(0, 0.3))
        new.recovery_rate_market = np.clip(
            scenario.recovery_rate_market + self.rng.normal(0, 2), 15, 70)
        new.risk_free_rate = scenario.risk_free_rate + self.rng.normal(0, 0.05)
        new.cds_bond_basis = scenario.cds_bond_basis + self.rng.normal(0, 5)
        new.funding_spread = max(50, scenario.funding_spread + self.rng.normal(0, 10))
        new.distress_ratio = max(1, scenario.distress_ratio + self.rng.normal(0, 1))
        new.repo_haircut = scenario.repo_haircut

        # Evolve each credit
        for old_c in scenario.credits:
            c = CreditEntity(
                name=old_c.name,
                sector=old_c.sector,
                rating=old_c.rating,
                leverage=old_c.leverage,
                interest_coverage=old_c.interest_coverage,
                cash_burn_months=max(0, old_c.cash_burn_months - 1),
                has_near_maturity=old_c.has_near_maturity,
                in_restructuring=old_c.in_restructuring,
                covenant_breach=old_c.covenant_breach,
                recent_downgrade=old_c.recent_downgrade,
                default_prob=old_c.default_prob,
            )

            # Random events
            event_roll = self.rng.random()
            if event_roll < 0.02:
                # Default event
                c.rating = 0.0
                c.default_prob = 100.0
                c.in_restructuring = True
            elif event_roll < 0.05:
                # Downgrade
                c.rating = max(0, c.rating - self.rng.uniform(0.5, 1.5))
                c.recent_downgrade = True
                c.default_prob = min(50, c.default_prob * 1.5)
            elif event_roll < 0.08:
                # Upgrade / positive event
                c.rating = min(5, c.rating + self.rng.uniform(0.3, 1.0))
                c.default_prob = max(0.5, c.default_prob * 0.7)
                c.recent_downgrade = False
            elif event_roll < 0.10:
                # Covenant breach
                c.covenant_breach = True
                c.default_prob = min(40, c.default_prob * 1.3)

            # Evolve spreads
            beta_market = 0.3  # correlation to HY index
            market_move = (new.hy_index_spread - scenario.hy_index_spread) / \
                scenario.hy_index_spread
            for j in range(N_SENIORITY):
                idio = self.rng.normal(0, old_c.spreads[j] * 0.03)
                systematic = old_c.spreads[j] * beta_market * market_move
                c.spreads[j] = max(20, old_c.spreads[j] + idio + systematic)

                # Events affect spreads
                if c.rating == 0.0:  # Default
                    c.spreads[j] = max(c.spreads[j], 2000 + j * 500)
                elif c.recent_downgrade and not old_c.recent_downgrade:
                    c.spreads[j] *= 1.2

            c.recovery_est = old_c.recovery_est.copy()
            new.credits.append(c)

        return new


# ─── Environment ──────────────────────────────────────────────────────────

class DistressedCreditEnv(gym.Env):
    """
    Gymnasium environment for distressed credit trading desk.

    The agent identifies mispriced credits, trades bonds and CDS,
    manages event risk, and hedges with the index.

    Action space: MultiDiscrete([13, 10, 5, 10])
      - action_type (13): NOOP through END_WEEK
      - credit_idx (10): which credit
      - seniority (5): which level of capital structure
      - size_bucket (10): position size

    Observation space: 264-dim continuous
      - Market state (200 dims)
      - Position book (64 dims)

    Reward: Weekly P&L in $K from carry + spread moves + events
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_weeks: int = 26,
        max_actions_per_week: int = 15,
        gross_limit: float = 500.0,  # $M gross exposure limit
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_weeks = max_weeks
        self.max_actions_per_week = max_actions_per_week
        self.gross_limit = gross_limit

        self.gen = DistressedScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([13, 10, 5, 10])

        self._market_dim = N_CREDITS * 19 + 10  # 200
        self._book_dim = 64
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32,
        )

        self._size_buckets = np.linspace(1.0, 20.0, 10)  # $M face

        self.scenario: Optional[DistressedScenario] = None
        self.prev_scenario: Optional[DistressedScenario] = None
        self.book: Optional[DistressedBook] = None
        self.week = 0
        self.actions_this_week = 0

    def _get_obs(self) -> np.ndarray:
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        market = self.scenario.to_observation()
        book = self.book.to_vector()
        return np.concatenate([market, book]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.gen = DistressedScenarioGenerator(seed=seed)

        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = DistressedBook()
        self.week = 0
        self.actions_this_week = 0

        return self._get_obs(), {"week": 0}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        action_type = DistressedAction(action[0])
        credit_idx = min(action[1], N_CREDITS - 1)
        seniority_idx = min(action[2], N_SENIORITY - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_this_week += 1

        if action_type == DistressedAction.END_WEEK:
            reward = self._end_of_week()
            self.week += 1
            self.actions_this_week = 0
            if self.week >= self.max_weeks:
                terminated = True
        elif action_type == DistressedAction.NOOP:
            pass
        elif action_type == DistressedAction.BUY_BOND:
            reward = self._trade_bond(credit_idx, seniority_idx, size)
        elif action_type == DistressedAction.SELL_BOND:
            reward = self._trade_bond(credit_idx, seniority_idx, -size)
        elif action_type == DistressedAction.BUY_CDS:
            reward = self._trade_cds(credit_idx, size)
        elif action_type == DistressedAction.SELL_CDS:
            reward = self._trade_cds(credit_idx, -size)
        elif action_type == DistressedAction.CAP_STRUCT_ARB:
            reward = self._cap_struct_arb(credit_idx, size)
        elif action_type == DistressedAction.REVERSE_CAP_ARB:
            reward = self._cap_struct_arb(credit_idx, -size)
        elif action_type == DistressedAction.BASIS_TRADE:
            reward = self._basis_trade(credit_idx, seniority_idx, size)
        elif action_type == DistressedAction.INDEX_HEDGE:
            reward = self._index_hedge(size * np.sign(self.book.net_exposure) * -1)
        elif action_type == DistressedAction.TAKE_RECOVERY:
            reward = self._recovery_trade(credit_idx, seniority_idx, size)
        elif action_type == DistressedAction.EVENT_TRADE:
            reward = self._event_trade(credit_idx, seniority_idx, size)
        elif action_type == DistressedAction.FLATTEN:
            reward = self._flatten()

        # Risk limit
        if self.book.gross_exposure > self.gross_limit:
            reward -= 3.0

        # Max actions per week
        if self.actions_this_week >= self.max_actions_per_week:
            eow = self._end_of_week()
            reward += eow
            self.week += 1
            self.actions_this_week = 0
            if self.week >= self.max_weeks:
                terminated = True

        info = {
            "week": self.week,
            "gross_exposure": self.book.gross_exposure,
            "net_exposure": self.book.net_exposure,
            "realized_pnl": self.book.realized_pnl,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _trade_bond(self, credit_idx: int, seniority_idx: int, size: float) -> float:
        """Buy/sell a cash bond."""
        credit = self.scenario.credits[credit_idx]
        spread = credit.spreads[seniority_idx]

        # Transaction cost: wider for more distressed, less liquid
        bid_ask = 0.5 + spread / 500.0  # points
        cost = abs(size) * bid_ask / 100.0  # $M -> cost

        self.book.positions[credit_idx, seniority_idx] += size
        return -cost * 1000  # Convert to $K

    def _trade_cds(self, credit_idx: int, size: float) -> float:
        """Buy/sell CDS protection."""
        credit = self.scenario.credits[credit_idx]
        # CDS bid/ask
        bid_ask = max(5, credit.spreads[2] * 0.05)  # 5% of spread
        cost = abs(size) * bid_ask / 10000.0  # running cost

        self.book.cds_positions[credit_idx] += size
        return -cost * 1000

    def _cap_struct_arb(self, credit_idx: int, size: float) -> float:
        """Capital structure arbitrage: long senior, short subordinated."""
        # Long 1L secured, short sub
        self.book.positions[credit_idx, 0] += size  # Long secured
        self.book.positions[credit_idx, 3] -= size * 0.5  # Short sub (less notional)

        credit = self.scenario.credits[credit_idx]
        cost = abs(size) * (credit.spreads[0] + credit.spreads[3]) * 0.03 / 10000
        return -cost * 1000

    def _basis_trade(self, credit_idx: int, seniority_idx: int, size: float) -> float:
        """Cash-CDS basis trade: buy bond + buy CDS protection."""
        self.book.positions[credit_idx, seniority_idx] += size
        self.book.cds_positions[credit_idx] += size  # Buy protection

        credit = self.scenario.credits[credit_idx]
        cost = abs(size) * 1.0 / 100.0  # ~1pt execution
        return -cost * 1000

    def _index_hedge(self, size: float) -> float:
        """Hedge with HY CDX index."""
        self.book.index_hedge += size
        cost = abs(size) * 0.1 / 100.0  # Tight market
        return -cost * 1000

    def _recovery_trade(self, credit_idx: int, seniority_idx: int, size: float) -> float:
        """Position for recovery value in a distressed credit."""
        credit = self.scenario.credits[credit_idx]
        if not credit.in_restructuring and credit.rating > 1.0:
            return -0.5  # Not distressed enough

        # Buy at distressed levels, targeting recovery
        price = max(10, credit.recovery_est[seniority_idx] * 0.8)  # Buy below recovery
        cost = abs(size) * (100 - price) / 100.0 * 0.02  # 2% of discount to par
        self.book.positions[credit_idx, seniority_idx] += size
        return -cost * 1000

    def _event_trade(self, credit_idx: int, seniority_idx: int, size: float) -> float:
        """Trade around a credit event."""
        credit = self.scenario.credits[credit_idx]
        has_event = credit.covenant_breach or credit.in_restructuring or \
            credit.has_near_maturity or credit.recent_downgrade

        if not has_event:
            return -0.3  # No event to trade

        self.book.positions[credit_idx, seniority_idx] += size
        cost = abs(size) * 0.5 / 100.0
        return -cost * 1000

    def _flatten(self) -> float:
        """Reduce all positions."""
        cost = 0.0
        for i in range(N_CREDITS):
            for j in range(N_SENIORITY):
                if abs(self.book.positions[i, j]) > 0.1:
                    spread = self.scenario.credits[i].spreads[j]
                    tc = abs(self.book.positions[i, j]) * (0.5 + spread / 500.0) / 100.0
                    cost += tc
                    self.book.positions[i, j] *= 0.1

            if abs(self.book.cds_positions[i]) > 0.1:
                cost += abs(self.book.cds_positions[i]) * 0.05 / 100.0
                self.book.cds_positions[i] *= 0.1

        self.book.index_hedge *= 0.1
        return -cost * 1000

    def _end_of_week(self) -> float:
        """Process end-of-week: carry, MTM, events."""
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)

        weekly_pnl = 0.0

        # ─── Carry (weekly) ───────────────────────────────────────────
        for i in range(N_CREDITS):
            for j in range(N_SENIORITY):
                pos = self.book.positions[i, j]
                if abs(pos) > 0.01:
                    spread_bps = self.prev_scenario.credits[i].spreads[j]
                    # Weekly carry = spread * position / 52
                    carry = pos * spread_bps / 10000.0 / 52.0 * 1000  # $K
                    # Subtract funding cost
                    funding = abs(pos) * self.prev_scenario.funding_spread / 10000.0 / 52.0 * 1000
                    weekly_pnl += carry - funding

            # CDS carry (protection buyer pays spread)
            cds_pos = self.book.cds_positions[i]
            if abs(cds_pos) > 0.01:
                spread = self.prev_scenario.credits[i].spreads[2]  # senior unsecured
                cds_carry = -cds_pos * spread / 10000.0 / 52.0 * 1000
                weekly_pnl += cds_carry

        # Index hedge carry
        if abs(self.book.index_hedge) > 0.01:
            index_carry = -self.book.index_hedge * \
                self.prev_scenario.hy_index_spread / 10000.0 / 52.0 * 1000
            weekly_pnl += index_carry

        self.book.carry_earned += weekly_pnl

        # ─── Mark-to-Market ───────────────────────────────────────────
        mtm = 0.0
        for i in range(N_CREDITS):
            for j in range(N_SENIORITY):
                pos = self.book.positions[i, j]
                if abs(pos) > 0.01:
                    old_spread = self.prev_scenario.credits[i].spreads[j]
                    new_spread = self.scenario.credits[i].spreads[j]
                    # Spread tightening = gain for long, loss for short
                    spread_change = old_spread - new_spread  # bps
                    # Duration approximation (higher spread = shorter duration)
                    dur = max(1.0, 5.0 - old_spread / 500.0)
                    price_change = spread_change / 100.0 * dur  # points
                    mtm += pos * price_change / 100.0 * 1000  # $K

            # CDS MTM
            cds_pos = self.book.cds_positions[i]
            if abs(cds_pos) > 0.01:
                old_s = self.prev_scenario.credits[i].spreads[2]
                new_s = self.scenario.credits[i].spreads[2]
                cds_mtm = cds_pos * (new_s - old_s) / 100.0 * 4.0 * 10  # $K (risky DV01)
                mtm += cds_mtm

        # Index hedge MTM
        if abs(self.book.index_hedge) > 0.01:
            idx_change = self.scenario.hy_index_spread - self.prev_scenario.hy_index_spread
            idx_mtm = self.book.index_hedge * idx_change / 100.0 * 4.0 * 10
            mtm += idx_mtm

        self.book.unrealized_pnl += mtm
        weekly_pnl += mtm

        # ─── Default Events ──────────────────────────────────────────
        for i in range(N_CREDITS):
            old_c = self.prev_scenario.credits[i]
            new_c = self.scenario.credits[i]
            if new_c.rating == 0.0 and old_c.rating > 0.0:
                # Default! Apply recovery
                for j in range(N_SENIORITY):
                    pos = self.book.positions[i, j]
                    if abs(pos) > 0.01:
                        recovery = new_c.recovery_est[j] / 100.0
                        # Loss = position * (1 - recovery) for longs
                        if pos > 0:
                            loss = pos * (1 - recovery) * 1000  # $K
                            weekly_pnl -= loss
                        else:
                            gain = abs(pos) * (1 - recovery) * 1000
                            weekly_pnl += gain

                # CDS pays out
                cds_pos = self.book.cds_positions[i]
                if cds_pos > 0:  # Bought protection
                    recovery = new_c.recovery_est[2] / 100.0
                    payout = cds_pos * (1 - recovery) * 1000
                    weekly_pnl += payout
                elif cds_pos < 0:  # Sold protection
                    recovery = new_c.recovery_est[2] / 100.0
                    payout = abs(cds_pos) * (1 - recovery) * 1000
                    weekly_pnl -= payout

        self.book.realized_pnl += weekly_pnl
        return float(np.clip(weekly_pnl, -500.0, 500.0))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Week {self.week}/{self.max_weeks}")
        print(f"HY Index: {self.scenario.hy_index_spread:.0f}bps | "
              f"VIX: {self.scenario.vix:.1f} | "
              f"Default Rate: {self.scenario.hy_default_rate:.1f}%")
        print(f"Gross: ${self.book.gross_exposure:.0f}M | "
              f"Net: ${self.book.net_exposure:.0f}M")
        print(f"P&L: ${self.book.realized_pnl:.0f}K")
        for i, c in enumerate(self.scenario.credits):
            pos_sum = np.sum(np.abs(self.book.positions[i]))
            if pos_sum > 0.1 or abs(self.book.cds_positions[i]) > 0.1:
                print(f"  {c.name}: rating={c.rating:.1f} "
                      f"spread={c.spreads[2]:.0f}bps "
                      f"pos=${pos_sum:.0f}M cds=${self.book.cds_positions[i]:.0f}M")


def make_distressed_credit_env(seed: Optional[int] = None, **kwargs) -> DistressedCreditEnv:
    return DistressedCreditEnv(seed=seed, **kwargs)
