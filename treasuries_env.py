"""
US Treasuries Trading Desk RL Environment.

Gymnasium-compatible environment for a US Treasuries market-making
and relative value trading desk. Covers:
  - Yield curve trading (2s5s10s butterflies, 2s10s steepeners)
  - Auction cycle (WI trading, auction tails, on/off-the-run spreads)
  - STRIPS (coupon stripping, reconstitution)
  - Futures basis (CTD, net basis, delivery option value)
  - Repo/financing (specials, GC rates, fails)
  - TIPS breakeven inflation trading

References:
  - Fleming (2003) "Measuring Treasury Market Liquidity"
  - Fontaine & Garcia (2012) "Bond Liquidity Premia" RFS
  - Hu, Pan & Wang (2013) "Noise as Information for Illiquidity" RFS
  - Krishnamurthy (2002) "The Bond/Old-Bond Spread" JFE
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Market State ─────────────────────────────────────────────────────────

TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]  # Standard Treasury tenors
TENOR_NAMES = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
N_TENORS = len(TENORS)

# Auction schedule: which tenors auction in which week of month
AUCTION_SCHEDULE = {
    # tenor_idx: (week_of_month, frequency_per_quarter)
    3: (1, 3),   # 2Y: week 1, monthly
    4: (1, 3),   # 3Y: week 1, monthly
    5: (2, 3),   # 5Y: week 2, monthly
    6: (2, 3),   # 7Y: week 2, monthly
    7: (2, 3),   # 10Y: week 2, monthly (refunding months: Feb/May/Aug/Nov)
    8: (3, 1),   # 20Y: week 3, quarterly
    9: (1, 1),   # 30Y: week 1, quarterly (refunding)
}


@dataclass
class TreasuryScenario:
    """Market state for the Treasuries desk."""
    # Yield curve (par yields, %)
    yields: np.ndarray = field(default_factory=lambda: np.zeros(N_TENORS))
    # On-the-run / off-the-run spread (bps)
    on_off_spreads: np.ndarray = field(default_factory=lambda: np.zeros(N_TENORS))
    # Repo rates (special rate for each tenor, %)
    repo_specials: np.ndarray = field(default_factory=lambda: np.zeros(N_TENORS))
    gc_rate: float = 5.3  # General collateral rate
    # TIPS breakeven rates (%)
    tips_breakevens: np.ndarray = field(default_factory=lambda: np.zeros(5))  # 2,5,7,10,30
    # Futures basis (32nds)
    futures_basis: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 2Y,5Y,10Y futs
    # Vol (bp/day realized)
    rate_vol: np.ndarray = field(default_factory=lambda: np.zeros(N_TENORS))
    # Auction state
    days_to_next_auction: int = 5
    next_auction_tenor: int = 7  # 10Y
    auction_size_bn: float = 42.0
    wi_spread: float = 0.0  # When-issued spread to OTR (bps)
    # Market regime
    regime: str = "normal"  # normal, risk_off, taper, hiking, cutting
    # Fed state
    fed_funds_rate: float = 5.25
    qe_pace_bn_month: float = 0.0  # positive = buying, negative = QT

    def to_observation(self) -> np.ndarray:
        """Flatten to observation vector."""
        return np.concatenate([
            self.yields / 10.0,                    # 10
            self.on_off_spreads / 20.0,            # 10
            self.repo_specials / 10.0,             # 10
            [self.gc_rate / 10.0],                 # 1
            self.tips_breakevens / 5.0,            # 5
            self.futures_basis / 50.0,             # 3
            self.rate_vol / 10.0,                  # 10
            [self.days_to_next_auction / 30.0],    # 1
            [self.next_auction_tenor / 10.0],      # 1
            [self.auction_size_bn / 100.0],        # 1
            [self.wi_spread / 10.0],               # 1
            [self.fed_funds_rate / 10.0],          # 1
            [self.qe_pace_bn_month / 100.0],       # 1
        ])  # Total: 56

    @property
    def obs_dim(self) -> int:
        return 56


# ─── Position State ───────────────────────────────────────────────────────

@dataclass
class TreasuryBook:
    """Current position book for the desk."""
    # DV01 by tenor ($K per bp)
    dv01_by_tenor: np.ndarray = field(default_factory=lambda: np.zeros(N_TENORS))
    # Net basis position (futures basis trades, $M notional)
    basis_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # TIPS breakeven position ($M DV01)
    tips_position: np.ndarray = field(default_factory=lambda: np.zeros(5))
    # WI position ($M face)
    wi_position: float = 0.0
    # Realized P&L this episode ($K)
    realized_pnl: float = 0.0
    # Unrealized P&L ($K)
    unrealized_pnl: float = 0.0
    # Carry earned ($K)
    carry_earned: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.dv01_by_tenor / 100.0,            # 10
            self.basis_position / 50.0,            # 3
            self.tips_position / 50.0,             # 5
            [self.wi_position / 100.0],            # 1
            [self.realized_pnl / 1000.0],          # 1
            [self.unrealized_pnl / 1000.0],        # 1
            [self.carry_earned / 1000.0],          # 1
        ])  # Total: 22

    @property
    def state_dim(self) -> int:
        return 22

    @property
    def total_dv01(self) -> float:
        return float(np.sum(np.abs(self.dv01_by_tenor)))

    @property
    def net_dv01(self) -> float:
        return float(np.sum(self.dv01_by_tenor))


# ─── Actions ──────────────────────────────────────────────────────────────

class TsyActionType(IntEnum):
    NOOP = 0
    BUY_TENOR = 1          # Buy a specific tenor (go long duration)
    SELL_TENOR = 2         # Sell a specific tenor (go short duration)
    STEEPENER = 3          # Buy long, sell short (2s10s etc.)
    FLATTENER = 4          # Sell long, buy short
    BUTTERFLY_BUY = 5      # Buy wings, sell belly
    BUTTERFLY_SELL = 6     # Sell wings, buy belly
    BUY_TIPS_BE = 7        # Buy breakeven (long TIPS, short nominal)
    SELL_TIPS_BE = 8       # Sell breakeven
    BUY_BASIS = 9          # Buy cash, sell futures
    SELL_BASIS = 10        # Sell cash, buy futures
    BID_AUCTION = 11       # Bid at upcoming auction
    FLATTEN_BOOK = 12      # Reduce all positions toward zero
    CLOSE_DAY = 13         # End trading day (terminal)


# ─── Scenario Generator ──────────────────────────────────────────────────

class TreasuryScenarioGenerator:
    """Generate realistic Treasury market scenarios."""

    REGIMES = ["normal", "risk_off", "hiking", "cutting", "qe", "qt"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> TreasuryScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        scenario = TreasuryScenario(regime=regime)

        # Base curve shape depends on regime
        if regime == "hiking":
            scenario.fed_funds_rate = self.rng.uniform(4.5, 6.0)
            short_rate = scenario.fed_funds_rate + self.rng.uniform(-0.1, 0.3)
            # Flat or inverted curve
            curve_slope = self.rng.uniform(-1.5, 0.5)  # 2s10s
        elif regime == "cutting":
            scenario.fed_funds_rate = self.rng.uniform(2.0, 4.5)
            short_rate = scenario.fed_funds_rate + self.rng.uniform(-0.3, 0.1)
            curve_slope = self.rng.uniform(0.5, 2.5)  # Steep curve
        elif regime == "risk_off":
            scenario.fed_funds_rate = self.rng.uniform(3.0, 5.5)
            short_rate = scenario.fed_funds_rate - self.rng.uniform(0.0, 0.5)
            curve_slope = self.rng.uniform(-0.5, 1.5)
        elif regime == "qe":
            scenario.fed_funds_rate = self.rng.uniform(0.0, 1.0)
            short_rate = scenario.fed_funds_rate + self.rng.uniform(0.0, 0.2)
            curve_slope = self.rng.uniform(1.0, 3.0)
            scenario.qe_pace_bn_month = self.rng.uniform(40, 120)
        elif regime == "qt":
            scenario.fed_funds_rate = self.rng.uniform(4.0, 5.5)
            short_rate = scenario.fed_funds_rate + self.rng.uniform(-0.1, 0.2)
            curve_slope = self.rng.uniform(-0.5, 1.0)
            scenario.qe_pace_bn_month = -self.rng.uniform(30, 95)
        else:  # normal
            scenario.fed_funds_rate = self.rng.uniform(2.0, 5.0)
            short_rate = scenario.fed_funds_rate + self.rng.uniform(-0.2, 0.3)
            curve_slope = self.rng.uniform(0.0, 2.0)

        # Build curve using Nelson-Siegel-like shape
        beta0 = short_rate + curve_slope  # long rate
        beta1 = -(curve_slope)  # slope factor
        beta2 = self.rng.uniform(-1.0, 1.5)  # curvature (butterfly)
        tau = self.rng.uniform(1.5, 4.0)

        for i, tenor in enumerate(TENORS):
            t = max(tenor, 0.1)
            decay = (1 - np.exp(-t / tau)) / (t / tau)
            hump = decay - np.exp(-t / tau)
            scenario.yields[i] = beta0 + beta1 * decay + beta2 * hump
            scenario.yields[i] += self.rng.normal(0, 0.03)  # noise
            scenario.yields[i] = max(0.01, scenario.yields[i])

        # On/off-the-run spreads (Krishnamurthy 2002)
        for i in range(N_TENORS):
            base_spread = 1.0 + TENORS[i] * 0.3  # longer maturity = wider
            if regime == "risk_off":
                base_spread *= 2.0  # flight to liquidity
            scenario.on_off_spreads[i] = base_spread + self.rng.normal(0, 0.5)

        # Repo specials
        scenario.gc_rate = scenario.fed_funds_rate - self.rng.uniform(0.0, 0.15)
        for i in range(N_TENORS):
            specialness = self.rng.exponential(3.0)  # bps below GC
            scenario.repo_specials[i] = scenario.gc_rate - specialness / 100.0

        # TIPS breakevens (2,5,7,10,30)
        base_be = self.rng.uniform(1.8, 3.0)
        for i in range(5):
            scenario.tips_breakevens[i] = base_be + self.rng.normal(0, 0.15)

        # Futures basis
        for i in range(3):
            scenario.futures_basis[i] = self.rng.uniform(-2, 8)  # 32nds

        # Realized vol
        base_vol = 5.0 if regime == "normal" else 8.0
        if regime == "risk_off":
            base_vol = 12.0
        for i in range(N_TENORS):
            duration_factor = min(TENORS[i], 10.0) / 10.0
            scenario.rate_vol[i] = base_vol * (0.5 + 0.5 * duration_factor) + \
                self.rng.normal(0, 1.0)

        # Auction
        auction_tenors = list(AUCTION_SCHEDULE.keys())
        scenario.next_auction_tenor = int(self.rng.choice(auction_tenors))
        scenario.days_to_next_auction = self.rng.randint(1, 15)
        sizes = {3: 69, 4: 56, 5: 70, 6: 44, 7: 42, 8: 16, 9: 25}
        scenario.auction_size_bn = float(sizes.get(scenario.next_auction_tenor, 40))
        scenario.wi_spread = self.rng.normal(0, 1.5)

        return scenario

    def step_scenario(self, scenario: TreasuryScenario) -> TreasuryScenario:
        """Evolve scenario by one trading day."""
        new = TreasuryScenario(regime=scenario.regime)

        # Mean-reverting yield changes
        vol_scale = 1.0 if scenario.regime == "normal" else 1.5
        for i in range(N_TENORS):
            daily_vol = scenario.rate_vol[i] / 100.0 / np.sqrt(252)
            shock = self.rng.normal(0, daily_vol) * vol_scale
            new.yields[i] = scenario.yields[i] + shock

        # Copy and evolve other fields
        new.on_off_spreads = scenario.on_off_spreads + self.rng.normal(0, 0.2, N_TENORS)
        new.repo_specials = scenario.repo_specials + self.rng.normal(0, 0.01, N_TENORS)
        new.gc_rate = scenario.gc_rate
        new.tips_breakevens = scenario.tips_breakevens + self.rng.normal(0, 0.02, 5)
        new.futures_basis = scenario.futures_basis + self.rng.normal(0, 0.5, 3)
        new.rate_vol = np.maximum(1.0, scenario.rate_vol + self.rng.normal(0, 0.3, N_TENORS))
        new.fed_funds_rate = scenario.fed_funds_rate
        new.qe_pace_bn_month = scenario.qe_pace_bn_month

        # Advance auction calendar
        new.days_to_next_auction = max(0, scenario.days_to_next_auction - 1)
        new.next_auction_tenor = scenario.next_auction_tenor
        new.auction_size_bn = scenario.auction_size_bn
        new.wi_spread = scenario.wi_spread + self.rng.normal(0, 0.3)

        if new.days_to_next_auction == 0:
            # Auction happened — generate tail and reset
            new.wi_spread = 0.0
            auction_tenors = list(AUCTION_SCHEDULE.keys())
            new.next_auction_tenor = int(self.rng.choice(auction_tenors))
            new.days_to_next_auction = self.rng.randint(5, 15)

        return new


# ─── Environment ──────────────────────────────────────────────────────────

class TreasuriesEnv(gym.Env):
    """
    Gymnasium environment for US Treasuries trading desk.

    The agent manages a Treasury trading book over a multi-day horizon,
    making curve trades, participating in auctions, and managing risk.

    Action space: MultiDiscrete([14, 10, 10])
      - action_type (14): NOOP through CLOSE_DAY
      - tenor_or_leg (10): which tenor or trade leg
      - size_bucket (10): position size

    Observation space: 78-dim continuous
      - Market state (56 dims)
      - Position book (22 dims)

    Reward: Daily P&L in $K from carry + mark-to-market + trading
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 20,
        max_actions_per_day: int = 10,
        dv01_limit: float = 500.0,  # $K total DV01 limit
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.dv01_limit = dv01_limit

        self.gen = TreasuryScenarioGenerator(seed=seed)

        # Action: [action_type, tenor/leg, size]
        self.action_space = spaces.MultiDiscrete([14, 10, 10])

        # Observation
        self._market_dim = 56
        self._book_dim = 22
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Size buckets: $5M to $50M DV01-equivalent notional
        self._size_buckets = np.linspace(5.0, 50.0, 10)  # $M

        # State
        self.scenario: Optional[TreasuryScenario] = None
        self.prev_scenario: Optional[TreasuryScenario] = None
        self.book: Optional[TreasuryBook] = None
        self.day = 0
        self.actions_today = 0

    def _get_obs(self) -> np.ndarray:
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        market = self.scenario.to_observation()
        book = self.book.to_vector()
        return np.concatenate([market, book]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.gen = TreasuryScenarioGenerator(seed=seed)

        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = TreasuryBook()
        self.day = 0
        self.actions_today = 0

        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        action_type = TsyActionType(action[0])
        tenor_idx = min(action[1], N_TENORS - 1)
        size = float(self._size_buckets[min(action[2], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if action_type == TsyActionType.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif action_type == TsyActionType.NOOP:
            reward = 0.0
        elif action_type == TsyActionType.BUY_TENOR:
            reward = self._trade_tenor(tenor_idx, size)
        elif action_type == TsyActionType.SELL_TENOR:
            reward = self._trade_tenor(tenor_idx, -size)
        elif action_type == TsyActionType.STEEPENER:
            # Buy 10Y, sell 2Y
            long_tenor = min(tenor_idx + 3, N_TENORS - 1)
            reward = self._curve_trade(tenor_idx, long_tenor, size)
        elif action_type == TsyActionType.FLATTENER:
            long_tenor = min(tenor_idx + 3, N_TENORS - 1)
            reward = self._curve_trade(tenor_idx, long_tenor, -size)
        elif action_type == TsyActionType.BUTTERFLY_BUY:
            reward = self._butterfly_trade(tenor_idx, size)
        elif action_type == TsyActionType.BUTTERFLY_SELL:
            reward = self._butterfly_trade(tenor_idx, -size)
        elif action_type == TsyActionType.BUY_TIPS_BE:
            tips_idx = min(tenor_idx, 4)
            reward = self._tips_trade(tips_idx, size)
        elif action_type == TsyActionType.SELL_TIPS_BE:
            tips_idx = min(tenor_idx, 4)
            reward = self._tips_trade(tips_idx, -size)
        elif action_type == TsyActionType.BUY_BASIS:
            basis_idx = min(tenor_idx, 2)
            reward = self._basis_trade(basis_idx, size)
        elif action_type == TsyActionType.SELL_BASIS:
            basis_idx = min(tenor_idx, 2)
            reward = self._basis_trade(basis_idx, -size)
        elif action_type == TsyActionType.BID_AUCTION:
            reward = self._bid_auction(size)
        elif action_type == TsyActionType.FLATTEN_BOOK:
            reward = self._flatten_book()

        # Risk limit check
        if self.book.total_dv01 > self.dv01_limit:
            reward -= 5.0  # Breach penalty

        # Too many actions in a day
        if self.actions_today >= self.max_actions_per_day:
            eod_reward = self._end_of_day()
            reward += eod_reward
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {
            "day": self.day,
            "total_dv01": self.book.total_dv01,
            "net_dv01": self.book.net_dv01,
            "realized_pnl": self.book.realized_pnl,
            "carry": self.book.carry_earned,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _trade_tenor(self, tenor_idx: int, size_m: float) -> float:
        """Outright buy/sell in a specific tenor."""
        # DV01 per $1M face, rough approximation
        dur = TENORS[tenor_idx] * 0.95  # modified duration ≈ tenor
        dv01_per_m = dur / 100.0 * 10.0  # $K DV01 per $M face

        notional = abs(size_m)
        dv01_change = np.sign(size_m) * notional * dv01_per_m

        self.book.dv01_by_tenor[tenor_idx] += dv01_change

        # Transaction cost: bid/ask spread (tighter for on-the-run)
        spread_32nds = 0.5 + 0.5 * (1.0 if tenor_idx >= 7 else 0.25)
        cost = notional * spread_32nds / 32.0 * 10.0  # $K
        return -cost  # Execution cost

    def _curve_trade(self, short_idx: int, long_idx: int, size_m: float) -> float:
        """Put on a curve trade (steepener if size > 0, flattener if < 0)."""
        # DV01-neutral: match DV01 of short and long legs
        dur_short = max(0.5, TENORS[short_idx] * 0.95)
        dur_long = max(0.5, TENORS[long_idx] * 0.95)

        notional_short = abs(size_m)
        notional_long = notional_short * dur_short / dur_long

        dv01_short = dur_short / 100.0 * 10.0
        dv01_long = dur_long / 100.0 * 10.0

        # Steepener: sell short, buy long
        sign = np.sign(size_m)
        self.book.dv01_by_tenor[short_idx] -= sign * notional_short * dv01_short
        self.book.dv01_by_tenor[long_idx] += sign * notional_long * dv01_long

        # Cost: two legs
        cost = (notional_short + notional_long) * 0.25 / 32.0 * 10.0
        return -cost

    def _butterfly_trade(self, center_idx: int, size_m: float) -> float:
        """2s5s10s style butterfly (buy wings, sell belly if size > 0)."""
        wing1_idx = max(0, center_idx - 2)
        wing2_idx = min(N_TENORS - 1, center_idx + 2)

        dur_w1 = max(0.5, TENORS[wing1_idx] * 0.95)
        dur_c = max(0.5, TENORS[center_idx] * 0.95)
        dur_w2 = max(0.5, TENORS[wing2_idx] * 0.95)

        # DV01-neutral butterfly
        notional = abs(size_m)
        sign = np.sign(size_m)

        self.book.dv01_by_tenor[wing1_idx] += sign * notional * dur_w1 / 100 * 10
        self.book.dv01_by_tenor[center_idx] -= sign * 2 * notional * dur_c / 100 * 10
        self.book.dv01_by_tenor[wing2_idx] += sign * notional * dur_w2 / 100 * 10

        cost = 3 * notional * 0.25 / 32.0 * 10.0
        return -cost

    def _tips_trade(self, tips_idx: int, size_m: float) -> float:
        """Trade TIPS breakeven inflation."""
        self.book.tips_position[tips_idx] += size_m
        cost = abs(size_m) * 0.5 / 32.0 * 10.0
        return -cost

    def _basis_trade(self, basis_idx: int, size_m: float) -> float:
        """Futures basis trade (cash vs futures)."""
        self.book.basis_position[basis_idx] += size_m
        cost = abs(size_m) * 0.25 / 32.0 * 10.0
        return -cost

    def _bid_auction(self, size_m: float) -> float:
        """Bid at the next Treasury auction."""
        if self.scenario is None or self.scenario.days_to_next_auction > 3:
            return -0.1  # Too early, wasted action

        # Simulate auction result
        tail = self.gen.rng.normal(0, 1.5)  # bps tail (negative = strong)
        tenor_idx = self.scenario.next_auction_tenor

        # Win allocation
        allocation = size_m * self.gen.rng.uniform(0.3, 1.0)

        # P&L from auction: tail * DV01
        dur = TENORS[tenor_idx] * 0.95
        dv01_per_m = dur / 100.0 * 10.0
        self.book.dv01_by_tenor[tenor_idx] += allocation * dv01_per_m

        # Immediate P&L: if you bid well and the auction tails, you profit
        pnl = -tail * allocation * dv01_per_m * 0.1  # $K
        self.book.wi_position += allocation
        return float(pnl)

    def _flatten_book(self) -> float:
        """Reduce all positions toward zero."""
        total_cost = 0.0
        for i in range(N_TENORS):
            if abs(self.book.dv01_by_tenor[i]) > 0.1:
                cost = abs(self.book.dv01_by_tenor[i]) * 0.5  # half a tick to flatten
                total_cost += cost
                self.book.dv01_by_tenor[i] *= 0.1  # Reduce to 10%

        for i in range(3):
            total_cost += abs(self.book.basis_position[i]) * 0.1
            self.book.basis_position[i] *= 0.1

        for i in range(5):
            total_cost += abs(self.book.tips_position[i]) * 0.1
            self.book.tips_position[i] *= 0.1

        self.book.wi_position *= 0.1
        return -total_cost

    def _end_of_day(self) -> float:
        """Process end-of-day: compute carry, MTM, advance scenario."""
        if self.scenario is None or self.book is None:
            return 0.0

        # Store previous scenario for MTM
        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)

        # ─── Carry ────────────────────────────────────────────────────
        # Long positions earn yield, pay repo (net carry)
        daily_carry = 0.0
        for i in range(N_TENORS):
            if abs(self.book.dv01_by_tenor[i]) > 0.01:
                # Carry = (yield - repo) * position * 1/252
                net_carry_rate = self.prev_scenario.yields[i] - self.prev_scenario.gc_rate
                # If repo is special, long position benefits more
                if self.book.dv01_by_tenor[i] > 0:
                    net_carry_rate = self.prev_scenario.yields[i] - \
                        self.prev_scenario.repo_specials[i]

                daily_carry += self.book.dv01_by_tenor[i] * net_carry_rate / 252.0

        self.book.carry_earned += daily_carry

        # ─── Mark-to-Market ───────────────────────────────────────────
        # P&L = -DV01 * yield_change (yields up = loss for long)
        daily_mtm = 0.0
        for i in range(N_TENORS):
            yield_change_bps = (self.scenario.yields[i] - self.prev_scenario.yields[i]) * 100
            daily_mtm -= self.book.dv01_by_tenor[i] * yield_change_bps

        # TIPS MTM
        for i in range(5):
            be_change = (self.scenario.tips_breakevens[i] - \
                        self.prev_scenario.tips_breakevens[i]) * 100
            daily_mtm += self.book.tips_position[i] * be_change * 0.1

        # Basis MTM
        for i in range(3):
            basis_change = self.scenario.futures_basis[i] - self.prev_scenario.futures_basis[i]
            daily_mtm += self.book.basis_position[i] * basis_change * 0.3

        self.book.unrealized_pnl += daily_mtm

        # Total daily P&L
        total_pnl = daily_carry + daily_mtm
        self.book.realized_pnl += daily_carry  # Only carry is "realized"

        # ─── Roll-down ────────────────────────────────────────────────
        # Bonds roll down the curve, picking up yield in steep curves
        rolldown_pnl = 0.0
        for i in range(1, N_TENORS):
            if abs(self.book.dv01_by_tenor[i]) > 0.01:
                # Approximate roll-down: yield pickup from aging 1 day
                if i > 0:
                    slope = (self.scenario.yields[i] - self.scenario.yields[i-1]) / \
                            max(0.1, TENORS[i] - TENORS[i-1])
                    daily_roll = slope / 365.0 * 100  # bps per day
                    rolldown_pnl += self.book.dv01_by_tenor[i] * daily_roll * 0.1

        total_pnl += rolldown_pnl

        return float(np.clip(total_pnl, -50.0, 50.0))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        print(f"Curve: 2Y={self.scenario.yields[3]:.2f}% 5Y={self.scenario.yields[5]:.2f}% "
              f"10Y={self.scenario.yields[7]:.2f}% 30Y={self.scenario.yields[9]:.2f}%")
        print(f"2s10s: {(self.scenario.yields[7]-self.scenario.yields[3])*100:.0f}bps")
        print(f"Total DV01: ${self.book.total_dv01:.0f}K | Net: ${self.book.net_dv01:.0f}K")
        print(f"P&L: ${self.book.realized_pnl + self.book.unrealized_pnl:.0f}K "
              f"(carry: ${self.book.carry_earned:.0f}K)")


def make_treasuries_env(seed: Optional[int] = None, **kwargs) -> TreasuriesEnv:
    return TreasuriesEnv(seed=seed, **kwargs)
