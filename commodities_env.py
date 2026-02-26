"""
Commodities Trading Desk RL Environment.

Gymnasium-compatible environment for commodities trading across energy,
metals, and agriculture.
Covers:
  - Crude oil (WTI, Brent) and refined products (gasoline, heating oil)
  - Natural gas (Henry Hub, basis)
  - Precious metals (gold, silver)
  - Base metals (copper, aluminum)
  - Calendar spreads (contango/backwardation trading)
  - Crack spreads (refining margins)
  - Spark spreads (power generation)
  - Storage and transportation

References:
  - Gorton & Rouwenhorst (2006) "Facts and Fantasies about Commodity Futures" FAJ
  - Tang & Xiong (2012) "Index Investment and Financialization of Commodities" FAJ
  - Hamilton (2009) "Causes and Consequences of the Oil Shock" Brookings
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Commodity Universe ──────────────────────────────────────────────────

N_COMMODITIES = 8
COMMODITY_NAMES = ["WTI", "Brent", "NatGas", "Gasoline",
                   "Gold", "Silver", "Copper", "Corn"]

N_MONTHS = 6  # Front-month through M6 on the curve

# Typical price levels
TYPICAL_PRICES = [75.0, 78.0, 3.50, 2.50, 2000.0, 25.0, 4.00, 5.50]
# Typical annualized vol (%)
TYPICAL_VOLS = [30, 28, 45, 35, 15, 22, 25, 28]
# Storage cost ($/unit/month approx)
STORAGE_COSTS = [0.50, 0.50, 0.02, 0.03, 0.10, 0.02, 0.01, 0.02]


@dataclass
class CommodityScenario:
    """Market state for commodities desk."""
    # Front-month prices
    prices: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Forward curve (spread to front month per month)
    fwd_curves: np.ndarray = field(
        default_factory=lambda: np.zeros((N_COMMODITIES, N_MONTHS)))
    # Implied vol (%)
    implied_vol: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Realized vol (%)
    realized_vol: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Inventory levels (days of supply, normalized)
    inventory_days: np.ndarray = field(default_factory=lambda: np.full(N_COMMODITIES, 30.0))
    # Convenience yield (%)
    convenience_yield: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Crack spread ($/bbl, WTI to gasoline + heating oil)
    crack_spread: float = 25.0
    # Brent-WTI spread
    brent_wti_spread: float = 3.0
    # Gold-silver ratio
    gold_silver_ratio: float = 80.0
    # OPEC spare capacity (mmbbl/d)
    opec_spare_capacity: float = 3.0
    # DXY (USD strength inversely related to commodities)
    dxy: float = 100.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.prices / np.array([150, 150, 10, 5, 3000, 50, 8, 10]),  # 8
            self.fwd_curves.flatten() / 10.0,          # 48
            self.implied_vol / 60.0,                   # 8
            self.realized_vol / 60.0,                  # 8
            self.inventory_days / 60.0,                # 8
            self.convenience_yield / 20.0,             # 8
            [self.crack_spread / 50.0],                # 1
            [self.brent_wti_spread / 15.0],            # 1
            [self.gold_silver_ratio / 100.0],          # 1
            [self.opec_spare_capacity / 10.0],         # 1
            [self.dxy / 120.0],                        # 1
        ])  # Total: 93

    @property
    def obs_dim(self) -> int:
        return 93


@dataclass
class CommodityBook:
    """Position book for commodities desk."""
    # Futures positions by commodity x month ($M notional)
    futures: np.ndarray = field(
        default_factory=lambda: np.zeros((N_COMMODITIES, N_MONTHS)))
    # Options delta by commodity ($M)
    option_delta: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Options vega by commodity ($K/vol pt)
    option_vega: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Calendar spread positions ($M)
    cal_spreads: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # Storage booked (units)
    storage_booked: np.ndarray = field(default_factory=lambda: np.zeros(N_COMMODITIES))
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.futures.flatten() / 50.0,             # 48
            self.option_delta / 50.0,                  # 8
            self.option_vega / 100.0,                  # 8
            self.cal_spreads / 20.0,                   # 8
            self.storage_booked / 100.0,               # 8
            [self.realized_pnl / 5000.0],              # 1
            [self.unrealized_pnl / 5000.0],            # 1
        ])  # Total: 82

    @property
    def state_dim(self) -> int:
        return 82

    @property
    def total_delta(self) -> float:
        return float(np.sum(np.abs(self.futures)) + np.sum(np.abs(self.option_delta)))


class CommodityAction(IntEnum):
    NOOP = 0
    BUY_FRONT = 1
    SELL_FRONT = 2
    BUY_BACK = 3          # Buy deferred month
    SELL_BACK = 4
    CAL_SPREAD_LONG = 5   # Buy front, sell back (backwardation play)
    CAL_SPREAD_SHORT = 6  # Sell front, buy back (contango play)
    CRACK_SPREAD = 7      # Buy crude, sell products
    REVERSE_CRACK = 8     # Buy products, sell crude
    BUY_CALL = 9
    BUY_PUT = 10
    BOOK_STORAGE = 11     # Book physical storage
    FLATTEN = 12
    CLOSE_DAY = 13


class CommodityScenarioGenerator:
    REGIMES = ["normal", "supply_shock", "demand_crash", "contango",
               "backwardation", "vol_spike"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> CommodityScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = CommodityScenario(regime=regime)

        # Base prices
        for i in range(N_COMMODITIES):
            mult = self.rng.uniform(0.7, 1.3)
            if regime == "supply_shock" and i < 4:  # Energy
                mult *= 1.3
            elif regime == "demand_crash":
                mult *= 0.7
            s.prices[i] = TYPICAL_PRICES[i] * mult

        # Forward curves
        for i in range(N_COMMODITIES):
            if regime == "contango" or (regime == "normal" and self.rng.random() < 0.5):
                # Contango: deferred months higher
                for m in range(N_MONTHS):
                    s.fwd_curves[i, m] = m * (STORAGE_COSTS[i] + self.rng.uniform(0, 0.3))
            elif regime == "backwardation" or regime == "supply_shock":
                # Backwardation: deferred months lower
                for m in range(N_MONTHS):
                    s.fwd_curves[i, m] = -m * self.rng.uniform(0.2, 2.0)
            else:
                for m in range(N_MONTHS):
                    s.fwd_curves[i, m] = m * self.rng.normal(0, 0.3)

        # Vols
        for i in range(N_COMMODITIES):
            vol_mult = 1.0 if regime != "vol_spike" else self.rng.uniform(1.3, 2.0)
            s.implied_vol[i] = TYPICAL_VOLS[i] * vol_mult + self.rng.normal(0, 3)
            s.realized_vol[i] = s.implied_vol[i] * self.rng.uniform(0.7, 1.3)

        # Inventories
        for i in range(N_COMMODITIES):
            base = 30.0
            if regime == "supply_shock":
                base = self.rng.uniform(15, 25)
            elif regime == "demand_crash":
                base = self.rng.uniform(35, 50)
            s.inventory_days[i] = base + self.rng.normal(0, 5)

        # Convenience yield (high when inventories low)
        for i in range(N_COMMODITIES):
            s.convenience_yield[i] = max(0, 20 - s.inventory_days[i] * 0.5 +
                                         self.rng.normal(0, 2))

        s.crack_spread = self.rng.uniform(10, 45)
        s.brent_wti_spread = self.rng.uniform(-2, 10)
        s.gold_silver_ratio = self.rng.uniform(60, 100)
        s.opec_spare_capacity = self.rng.uniform(1, 6)
        s.dxy = self.rng.uniform(88, 115)

        return s

    def step_scenario(self, s: CommodityScenario) -> CommodityScenario:
        new = CommodityScenario(regime=s.regime)

        # Price evolution
        for i in range(N_COMMODITIES):
            daily_vol = s.realized_vol[i] / 100 / np.sqrt(252)
            shock = self.rng.normal(0, daily_vol)
            # Mean reversion toward typical
            mr = 0.01 * (TYPICAL_PRICES[i] - s.prices[i]) / TYPICAL_PRICES[i]
            new.prices[i] = s.prices[i] * (1 + shock + mr)
            new.prices[i] = max(0.01, new.prices[i])

        # Forward curves evolve
        for i in range(N_COMMODITIES):
            for m in range(N_MONTHS):
                new.fwd_curves[i, m] = s.fwd_curves[i, m] + self.rng.normal(0, 0.1)

        new.implied_vol = np.maximum(5, s.implied_vol + self.rng.normal(0, 0.5, N_COMMODITIES))
        new.realized_vol = np.maximum(5, s.realized_vol + self.rng.normal(0, 0.7, N_COMMODITIES))
        new.inventory_days = np.maximum(5, s.inventory_days + self.rng.normal(0, 0.5, N_COMMODITIES))
        for i in range(N_COMMODITIES):
            new.convenience_yield[i] = max(0, 20 - new.inventory_days[i] * 0.5 +
                                           self.rng.normal(0, 1))

        new.crack_spread = max(5, s.crack_spread + self.rng.normal(0, 1))
        new.brent_wti_spread = s.brent_wti_spread + self.rng.normal(0, 0.3)
        new.gold_silver_ratio = max(40, s.gold_silver_ratio + self.rng.normal(0, 0.5))
        new.opec_spare_capacity = max(0, s.opec_spare_capacity + self.rng.normal(0, 0.1))
        new.dxy = s.dxy + self.rng.normal(0, 0.2)

        return new


class CommoditiesEnv(gym.Env):
    """
    Gymnasium environment for commodities trading desk.

    Action space: MultiDiscrete([14, 8, 6, 10])
      - action_type (14): NOOP through CLOSE_DAY
      - commodity_idx (8): which commodity
      - month_idx (6): which contract month
      - size_bucket (10): position size

    Observation: 175-dim
      - Market (93) + Book (82)

    Reward: Daily P&L from price moves + roll + storage + spreads
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 20,
        max_actions_per_day: int = 12,
        delta_limit: float = 200.0,  # $M total delta
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.delta_limit = delta_limit
        self.gen = CommodityScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([14, 8, 6, 10])

        self._market_dim = 93
        self._book_dim = 82
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._size_buckets = np.linspace(2, 50, 10)  # $M notional

        self.scenario: Optional[CommodityScenario] = None
        self.prev_scenario: Optional[CommodityScenario] = None
        self.book: Optional[CommodityBook] = None
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
            self.gen = CommodityScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = CommodityBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        act = CommodityAction(action[0])
        comm = min(action[1], N_COMMODITIES - 1)
        month = min(action[2], N_MONTHS - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == CommodityAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == CommodityAction.NOOP:
            pass
        elif act == CommodityAction.BUY_FRONT:
            reward = self._trade_futures(comm, 0, size)
        elif act == CommodityAction.SELL_FRONT:
            reward = self._trade_futures(comm, 0, -size)
        elif act == CommodityAction.BUY_BACK:
            reward = self._trade_futures(comm, month, size)
        elif act == CommodityAction.SELL_BACK:
            reward = self._trade_futures(comm, month, -size)
        elif act == CommodityAction.CAL_SPREAD_LONG:
            reward = self._cal_spread(comm, month, size)
        elif act == CommodityAction.CAL_SPREAD_SHORT:
            reward = self._cal_spread(comm, month, -size)
        elif act == CommodityAction.CRACK_SPREAD:
            reward = self._crack_spread(size)
        elif act == CommodityAction.REVERSE_CRACK:
            reward = self._crack_spread(-size)
        elif act == CommodityAction.BUY_CALL:
            reward = self._trade_option(comm, size, is_call=True)
        elif act == CommodityAction.BUY_PUT:
            reward = self._trade_option(comm, size, is_call=False)
        elif act == CommodityAction.BOOK_STORAGE:
            reward = self._book_storage(comm, size)
        elif act == CommodityAction.FLATTEN:
            reward = self._flatten()

        if self.book.total_delta > self.delta_limit:
            reward -= 5.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {
            "day": self.day,
            "total_delta": self.book.total_delta,
            "realized_pnl": self.book.realized_pnl,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _trade_futures(self, comm: int, month: int, size: float) -> float:
        self.book.futures[comm, month] += size
        # Execution cost: ~1-3 ticks depending on liquidity
        price = self.scenario.prices[comm]
        cost = abs(size) * 0.001 * price  # ~10bps
        return -cost

    def _cal_spread(self, comm: int, month: int, size: float) -> float:
        # Long front, short back (or vice versa)
        back = min(month + 2, N_MONTHS - 1)
        self.book.futures[comm, 0] += size
        self.book.futures[comm, back] -= size
        self.book.cal_spreads[comm] += size
        cost = abs(size) * 0.0005 * self.scenario.prices[comm]
        return -cost

    def _crack_spread(self, size: float) -> float:
        # 3-2-1 crack: 3 bbl crude -> 2 bbl gasoline + 1 bbl heating oil
        self.book.futures[0, 0] -= size * 3  # Sell crude
        self.book.futures[3, 0] += size * 2   # Buy gasoline
        # Heating oil not in universe, use NatGas as proxy
        self.book.futures[2, 0] += size * 0.5
        cost = abs(size) * 0.05
        return -cost

    def _trade_option(self, comm: int, size: float, is_call: bool) -> float:
        vol = self.scenario.implied_vol[comm]
        price = self.scenario.prices[comm]
        premium = vol / 100 * np.sqrt(30 / 365) * 0.4 * abs(size) * price * 0.01
        delta_sign = 0.5 if is_call else -0.5
        self.book.option_delta[comm] += delta_sign * abs(size)
        self.book.option_vega[comm] += abs(size) * 0.3
        return -premium * 0.1

    def _book_storage(self, comm: int, size: float) -> float:
        self.book.storage_booked[comm] += abs(size)
        # Storage cost
        monthly_cost = abs(size) * STORAGE_COSTS[comm] * 10  # $K/month
        return -monthly_cost / 30  # Daily amortized

    def _flatten(self) -> float:
        cost = 0.0
        for i in range(N_COMMODITIES):
            for m in range(N_MONTHS):
                cost += abs(self.book.futures[i, m]) * 0.001 * self.scenario.prices[i]
                self.book.futures[i, m] *= 0.1
        self.book.option_delta *= 0.1
        self.book.option_vega *= 0.1
        self.book.cal_spreads *= 0.1
        return -cost

    def _end_of_day(self) -> float:
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        daily_pnl = 0.0

        # Futures MTM
        for i in range(N_COMMODITIES):
            pct_move = (self.scenario.prices[i] - self.prev_scenario.prices[i]) / \
                max(0.01, self.prev_scenario.prices[i])
            for m in range(N_MONTHS):
                # Deferred months move with front but also curve change
                curve_change = self.scenario.fwd_curves[i, m] - self.prev_scenario.fwd_curves[i, m]
                total_move = pct_move + curve_change / max(0.01, self.prev_scenario.prices[i])
                daily_pnl += self.book.futures[i, m] * total_move * 1000

        # Option P&L (delta + vega)
        for i in range(N_COMMODITIES):
            pct_move = (self.scenario.prices[i] - self.prev_scenario.prices[i]) / \
                max(0.01, self.prev_scenario.prices[i])
            daily_pnl += self.book.option_delta[i] * pct_move * 1000
            vol_change = self.scenario.implied_vol[i] - self.prev_scenario.implied_vol[i]
            daily_pnl += self.book.option_vega[i] * vol_change

        # Roll yield (contango = cost, backwardation = income)
        for i in range(N_COMMODITIES):
            total_front = self.book.futures[i, 0]
            if abs(total_front) > 0.01:
                # Daily roll = fwd_curve[1] / 30 (monthly spread / days)
                daily_roll = self.prev_scenario.fwd_curves[i, 1] / 30.0
                roll_pnl = -total_front * daily_roll / max(0.01, self.prev_scenario.prices[i]) * 1000
                daily_pnl += roll_pnl

        # Storage income (if in backwardation with storage)
        for i in range(N_COMMODITIES):
            if self.book.storage_booked[i] > 0:
                convenience = self.prev_scenario.convenience_yield[i]
                income = self.book.storage_booked[i] * convenience / 100 / 365 * 1000
                cost = self.book.storage_booked[i] * STORAGE_COSTS[i] / 30 * 10
                daily_pnl += income - cost

        self.book.realized_pnl += daily_pnl
        return float(np.clip(daily_pnl, -100, 100))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        print(f"WTI: ${self.scenario.prices[0]:.2f} | Brent: ${self.scenario.prices[1]:.2f} | "
              f"Gold: ${self.scenario.prices[4]:.0f}")
        print(f"Crack: ${self.scenario.crack_spread:.1f} | "
              f"B-W: ${self.scenario.brent_wti_spread:.2f}")
        print(f"Total Delta: ${self.book.total_delta:.0f}M | "
              f"P&L: ${self.book.realized_pnl:.0f}K")


def make_commodities_env(seed: Optional[int] = None, **kwargs) -> CommoditiesEnv:
    return CommoditiesEnv(seed=seed, **kwargs)
