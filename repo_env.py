"""
Repo / Securities Financing Desk RL Environment.

Gymnasium-compatible environment for repo and securities lending.
Covers:
  - Overnight and term repo (GC and specials)
  - Reverse repo (cash lending)
  - Tri-party vs bilateral
  - Securities lending (equities and HY bonds)
  - Matched book management
  - Collateral transformation
  - Balance sheet optimization (SLR, LCR)
  - Quarter-end/year-end window dressing

References:
  - Copeland, Martin & Walker (2014) "Repo Runs" JF
  - Krishnamurthy, Nagel & Orlov (2014) "Sizing Up Repo" JF
  - Infante (2019) "Liquidity Windfalls" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Collateral Classes ──────────────────────────────────────────────────

N_COLLATERAL = 7  # UST, Agency, AgencyMBS, IG Corp, HY Corp, Equity, Muni
COLLATERAL_NAMES = ["UST", "Agency", "AgencyMBS", "IG_Corp", "HY_Corp", "Equity", "Muni"]

# Typical haircuts (%)
BASE_HAIRCUTS = [2.0, 3.0, 4.0, 5.0, 12.0, 15.0, 5.0]

# ─── Term Buckets ────────────────────────────────────────────────────────

N_TERMS = 5  # O/N, 1W, 2W, 1M, 3M
TERM_NAMES = ["ON", "1W", "2W", "1M", "3M"]
TERM_DAYS = [1, 7, 14, 30, 90]


@dataclass
class RepoScenario:
    """Market state for repo/financing desk."""
    # GC repo rates by term (%)
    gc_rates: np.ndarray = field(default_factory=lambda: np.zeros(N_TERMS))
    # Special rates by collateral type (% below GC)
    specials_spread: np.ndarray = field(default_factory=lambda: np.zeros(N_COLLATERAL))
    # Fed funds rate
    fed_funds: float = 5.25
    # RRP facility rate and usage ($B)
    rrp_rate: float = 5.30
    rrp_usage_bn: float = 500.0
    # Securities lending fee rates (bps annualized) by collateral
    sec_lending_fees: np.ndarray = field(default_factory=lambda: np.zeros(N_COLLATERAL))
    # Fails ($B in market)
    fails_volume_bn: float = 50.0
    # Balance sheet metrics
    days_to_quarter_end: int = 45
    slr_utilization: float = 0.70  # 0-1 (how close to SLR limit)
    lcr_ratio: float = 1.20  # >1.0 required
    # Collateral demand/supply imbalance (positive = collateral scarce)
    collateral_scarcity: np.ndarray = field(default_factory=lambda: np.zeros(N_COLLATERAL))
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.gc_rates / 10.0,                      # 5
            self.specials_spread / 2.0,                # 7
            [self.fed_funds / 10.0],                   # 1
            [self.rrp_rate / 10.0],                    # 1
            [self.rrp_usage_bn / 2500.0],              # 1
            self.sec_lending_fees / 500.0,             # 7
            [self.fails_volume_bn / 300.0],            # 1
            [self.days_to_quarter_end / 90.0],         # 1
            [self.slr_utilization],                     # 1
            [self.lcr_ratio / 2.0],                    # 1
            self.collateral_scarcity / 5.0,            # 7
        ])  # Total: 33

    @property
    def obs_dim(self) -> int:
        return 33


@dataclass
class RepoBook:
    """Position book for repo desk."""
    # Repo (borrowing cash = lending collateral) by collateral x term ($B)
    repo_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((N_COLLATERAL, N_TERMS)))
    # Reverse repo (lending cash = borrowing collateral) by collateral x term ($B)
    reverse_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((N_COLLATERAL, N_TERMS)))
    # Securities lending ($B) by collateral
    sec_lending: np.ndarray = field(default_factory=lambda: np.zeros(N_COLLATERAL))
    # Net cash position ($B)
    net_cash: float = 0.0
    # P&L ($M)
    realized_pnl: float = 0.0
    net_interest_income: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.repo_positions.flatten() / 20.0,      # 35
            self.reverse_positions.flatten() / 20.0,   # 35
            self.sec_lending / 10.0,                   # 7
            [self.net_cash / 50.0],                    # 1
            [self.realized_pnl / 100.0],               # 1
            [self.net_interest_income / 50.0],         # 1
        ])  # Total: 80

    @property
    def state_dim(self) -> int:
        return 80

    @property
    def total_repo(self) -> float:
        return float(np.sum(self.repo_positions))

    @property
    def total_reverse(self) -> float:
        return float(np.sum(self.reverse_positions))

    @property
    def balance_sheet_usage(self) -> float:
        return float(np.sum(self.repo_positions) + np.sum(self.reverse_positions))


class RepoAction(IntEnum):
    NOOP = 0
    REPO_LEND_COLL = 1     # Borrow cash (lend collateral)
    REVERSE_LEND_CASH = 2  # Lend cash (get collateral)
    ROLL_REPO = 3           # Roll maturing repo
    TERM_OUT = 4            # Extend from O/N to term
    UNWIND = 5              # Unwind position
    SEC_LEND = 6            # Lend securities
    SEC_BORROW = 7          # Borrow securities
    COLLATERAL_TRANSFORM = 8  # Upgrade: take junk, lend UST
    RRP_FACILITY = 9        # Park cash at Fed RRP
    MATCH_BOOK = 10         # Match repo vs reverse
    FLATTEN = 11
    CLOSE_DAY = 12


class RepoScenarioGenerator:
    REGIMES = ["normal", "quarter_end", "year_end", "collateral_squeeze",
               "rate_vol", "reserve_drain"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> RepoScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = RepoScenario(regime=regime)

        s.fed_funds = self.rng.uniform(0.0, 6.0)
        s.rrp_rate = s.fed_funds + self.rng.uniform(0.0, 0.10)

        # GC rates around fed funds
        gc_base = s.fed_funds + self.rng.uniform(-0.15, 0.10)
        for i in range(N_TERMS):
            term_premium = TERM_DAYS[i] / 365 * self.rng.uniform(0, 0.2)
            s.gc_rates[i] = gc_base + term_premium + self.rng.normal(0, 0.02)

        if regime in ("quarter_end", "year_end"):
            # Rates spike, balance sheet precious
            s.gc_rates += self.rng.uniform(0.1, 0.5)
            s.slr_utilization = self.rng.uniform(0.85, 0.98)
            s.days_to_quarter_end = self.rng.randint(1, 10)
        elif regime == "collateral_squeeze":
            s.collateral_scarcity[:3] = self.rng.uniform(2, 5, 3)
            s.specials_spread[:3] = self.rng.uniform(0.5, 2.0, 3)
            s.fails_volume_bn = self.rng.uniform(100, 300)
        else:
            s.days_to_quarter_end = self.rng.randint(10, 90)
            s.slr_utilization = self.rng.uniform(0.60, 0.85)

        # Specials
        for i in range(N_COLLATERAL):
            s.specials_spread[i] = self.rng.exponential(0.1) + \
                (0.5 if regime == "collateral_squeeze" and i < 3 else 0)

        # Sec lending fees
        s.sec_lending_fees = np.array([5, 8, 10, 15, 50, 30, 12], dtype=float) + \
            self.rng.normal(0, 3, N_COLLATERAL)
        s.sec_lending_fees = np.maximum(1, s.sec_lending_fees)

        # RRP usage
        if s.fed_funds < 1.0:
            s.rrp_usage_bn = self.rng.uniform(0, 200)
        else:
            s.rrp_usage_bn = self.rng.uniform(200, 2500)

        s.fails_volume_bn = max(10, s.fails_volume_bn + self.rng.normal(0, 20))
        s.lcr_ratio = self.rng.uniform(1.05, 1.50)

        # Collateral scarcity
        for i in range(N_COLLATERAL):
            s.collateral_scarcity[i] = self.rng.normal(0, 1)
            if regime == "collateral_squeeze":
                s.collateral_scarcity[i] += 2

        return s

    def step_scenario(self, s: RepoScenario) -> RepoScenario:
        new = RepoScenario(regime=s.regime)
        new.fed_funds = s.fed_funds
        new.rrp_rate = s.rrp_rate

        for i in range(N_TERMS):
            new.gc_rates[i] = s.gc_rates[i] + self.rng.normal(0, 0.02)

        new.specials_spread = np.maximum(0, s.specials_spread + self.rng.normal(0, 0.05, N_COLLATERAL))
        new.sec_lending_fees = np.maximum(1, s.sec_lending_fees + self.rng.normal(0, 2, N_COLLATERAL))
        new.fails_volume_bn = max(10, s.fails_volume_bn + self.rng.normal(0, 10))
        new.days_to_quarter_end = max(0, s.days_to_quarter_end - 1)
        new.slr_utilization = np.clip(s.slr_utilization + self.rng.normal(0, 0.01), 0.5, 0.99)
        new.lcr_ratio = max(1.0, s.lcr_ratio + self.rng.normal(0, 0.02))
        new.rrp_usage_bn = max(0, s.rrp_usage_bn + self.rng.normal(0, 30))
        new.collateral_scarcity = s.collateral_scarcity + self.rng.normal(0, 0.3, N_COLLATERAL)

        # Quarter-end squeeze
        if new.days_to_quarter_end <= 3 and new.days_to_quarter_end > 0:
            new.gc_rates += 0.1
            new.slr_utilization = min(0.99, new.slr_utilization + 0.05)

        return new


class RepoEnv(gym.Env):
    """
    Gymnasium environment for repo/financing desk.

    Action space: MultiDiscrete([13, 7, 5, 10])
      - action_type (13): NOOP through CLOSE_DAY
      - collateral_idx (7): collateral type
      - term_idx (5): tenor bucket
      - size_bucket (10): position size ($B)

    Observation: 113-dim
      - Market (33) + Book (80)

    Reward: Daily NII from spread capture + specials + sec lending
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 30,
        max_actions_per_day: int = 15,
        bs_limit: float = 100.0,  # $B balance sheet limit
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.bs_limit = bs_limit
        self.gen = RepoScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([13, 7, 5, 10])

        self._market_dim = 33
        self._book_dim = 80
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._size_buckets = np.linspace(0.5, 10, 10)  # $B

        self.scenario: Optional[RepoScenario] = None
        self.prev_scenario: Optional[RepoScenario] = None
        self.book: Optional[RepoBook] = None
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
            self.gen = RepoScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = RepoBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        act = RepoAction(action[0])
        coll_idx = min(action[1], N_COLLATERAL - 1)
        term_idx = min(action[2], N_TERMS - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == RepoAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == RepoAction.NOOP:
            pass
        elif act == RepoAction.REPO_LEND_COLL:
            reward = self._do_repo(coll_idx, term_idx, size)
        elif act == RepoAction.REVERSE_LEND_CASH:
            reward = self._do_reverse(coll_idx, term_idx, size)
        elif act == RepoAction.ROLL_REPO:
            reward = self._roll(coll_idx, term_idx)
        elif act == RepoAction.TERM_OUT:
            reward = self._term_out(coll_idx, term_idx)
        elif act == RepoAction.UNWIND:
            reward = self._unwind(coll_idx, term_idx)
        elif act == RepoAction.SEC_LEND:
            reward = self._sec_lend(coll_idx, size)
        elif act == RepoAction.SEC_BORROW:
            reward = self._sec_borrow(coll_idx, size)
        elif act == RepoAction.COLLATERAL_TRANSFORM:
            reward = self._collateral_transform(coll_idx, size)
        elif act == RepoAction.RRP_FACILITY:
            reward = self._rrp_park(size)
        elif act == RepoAction.MATCH_BOOK:
            reward = self._match_book(coll_idx, term_idx, size)
        elif act == RepoAction.FLATTEN:
            reward = self._flatten()

        # Balance sheet penalty
        if self.book.balance_sheet_usage > self.bs_limit:
            reward -= 5.0

        # SLR penalty
        if self.scenario.slr_utilization > 0.95:
            reward -= 3.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {
            "day": self.day,
            "total_repo": self.book.total_repo,
            "total_reverse": self.book.total_reverse,
            "bs_usage": self.book.balance_sheet_usage,
            "nii": self.book.net_interest_income,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _do_repo(self, coll: int, term: int, size: float) -> float:
        self.book.repo_positions[coll, term] += size
        self.book.net_cash += size
        return 0.0  # Cost comes through NII

    def _do_reverse(self, coll: int, term: int, size: float) -> float:
        self.book.reverse_positions[coll, term] += size
        self.book.net_cash -= size
        return 0.0

    def _roll(self, coll: int, term: int) -> float:
        # Roll O/N to next day (just keep position)
        if self.book.repo_positions[coll, 0] > 0:
            return 0.0  # Free to roll O/N
        return -0.1

    def _term_out(self, coll: int, term: int) -> float:
        # Move O/N to term
        amt = min(self.book.repo_positions[coll, 0], 2.0)
        if amt < 0.1:
            return -0.1
        self.book.repo_positions[coll, 0] -= amt
        self.book.repo_positions[coll, term] += amt
        return 0.0

    def _unwind(self, coll: int, term: int) -> float:
        repo = self.book.repo_positions[coll, term]
        rev = self.book.reverse_positions[coll, term]
        total = repo + rev
        if total < 0.1:
            return -0.1
        # Early termination cost
        cost = total * 0.01 / 100 * TERM_DAYS[term] * 10  # $M
        self.book.repo_positions[coll, term] *= 0.1
        self.book.reverse_positions[coll, term] *= 0.1
        self.book.net_cash -= repo * 0.9
        self.book.net_cash += rev * 0.9
        return -cost

    def _sec_lend(self, coll: int, size: float) -> float:
        self.book.sec_lending[coll] += size
        return 0.0  # Fee comes through daily NII

    def _sec_borrow(self, coll: int, size: float) -> float:
        self.book.sec_lending[coll] -= size
        return 0.0

    def _collateral_transform(self, coll: int, size: float) -> float:
        # Take lower quality collateral, lend UST
        if coll <= 1:  # Already high quality
            return -0.1
        # Reverse in low-quality, repo out UST
        self.book.reverse_positions[coll, 1] += size
        self.book.repo_positions[0, 1] += size  # UST
        return 0.0  # Earn spread through NII

    def _rrp_park(self, size: float) -> float:
        # Park cash at RRP (earns RRP rate, no balance sheet usage)
        self.book.net_cash -= size
        # Store as reverse with special handling
        self.book.reverse_positions[0, 0] += size
        return 0.0

    def _match_book(self, coll: int, term: int, size: float) -> float:
        # Matched book: borrow and lend simultaneously at spread
        self.book.repo_positions[coll, term] += size
        self.book.reverse_positions[coll, term] += size
        return 0.0

    def _flatten(self) -> float:
        cost = 0.0
        for i in range(N_COLLATERAL):
            for j in range(N_TERMS):
                cost += (abs(self.book.repo_positions[i, j]) +
                        abs(self.book.reverse_positions[i, j])) * 0.005
                self.book.repo_positions[i, j] *= 0.1
                self.book.reverse_positions[i, j] *= 0.1
        self.book.sec_lending *= 0.1
        self.book.net_cash *= 0.1
        return -cost

    def _end_of_day(self) -> float:
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        daily_pnl = 0.0

        # NII from matched book spread
        for i in range(N_COLLATERAL):
            for j in range(N_TERMS):
                # Repo: borrow cash at GC - specials
                repo_rate = self.prev_scenario.gc_rates[j] - \
                    self.prev_scenario.specials_spread[i]
                repo_cost = self.book.repo_positions[i, j] * repo_rate / 100 / 365 * 1000  # $M

                # Reverse: lend cash at GC
                rev_income = self.book.reverse_positions[i, j] * \
                    self.prev_scenario.gc_rates[j] / 100 / 365 * 1000

                daily_pnl += rev_income - repo_cost

        # Sec lending income
        for i in range(N_COLLATERAL):
            if abs(self.book.sec_lending[i]) > 0.01:
                fee = self.prev_scenario.sec_lending_fees[i]
                income = self.book.sec_lending[i] * fee / 10000 / 365 * 1000
                daily_pnl += income

        # Cost of carry on net cash (if negative, paying fed funds)
        if self.book.net_cash < 0:
            daily_pnl += self.book.net_cash * self.prev_scenario.fed_funds / 100 / 365 * 1000

        # Quarter-end bonus: if positioned well (low BS usage, term funded)
        if self.scenario.days_to_quarter_end <= 3:
            # Premium for having term repo (not O/N)
            term_funded = float(np.sum(self.book.repo_positions[:, 2:]))
            if term_funded > 0:
                daily_pnl += term_funded * 0.5  # Quarter-end premium

        self.book.net_interest_income += daily_pnl
        self.book.realized_pnl += daily_pnl

        return float(np.clip(daily_pnl, -20, 20))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        print(f"Fed Funds: {self.scenario.fed_funds:.2f}% | "
              f"GC O/N: {self.scenario.gc_rates[0]:.2f}%")
        print(f"Total Repo: ${self.book.total_repo:.1f}B | "
              f"Reverse: ${self.book.total_reverse:.1f}B")
        print(f"BS Usage: ${self.book.balance_sheet_usage:.1f}B | "
              f"SLR: {self.scenario.slr_utilization:.0%}")
        print(f"NII: ${self.book.net_interest_income:.1f}M | "
              f"Days to QE: {self.scenario.days_to_quarter_end}")


def make_repo_env(seed: Optional[int] = None, **kwargs) -> RepoEnv:
    return RepoEnv(seed=seed, **kwargs)
