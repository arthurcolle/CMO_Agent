"""
Equities Cash Trading Desk RL Environment.

Gymnasium-compatible environment for equity market-making and prop trading.
Covers:
  - Market-making (bid/ask management)
  - Sector rotation
  - Pairs / stat arb trading
  - ETF vs basket arb
  - Block trading (facilitation)
  - Index rebalance front-running
  - Earnings event trading
  - VWAP/TWAP execution

References:
  - Avellaneda & Stoikov (2008) "High-Frequency Trading in a Limit Order Book"
  - Cont, Kukanov & Stoikov (2014) "The Price Impact of Order Book Events"
  - Grinold & Kahn (2000) "Active Portfolio Management"
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


N_SECTORS = 11  # GICS sectors
SECTOR_NAMES = ["Tech", "Healthcare", "Financials", "ConsDisc", "Industrials",
                "ConsStaples", "Energy", "Utilities", "Materials", "RealEstate", "Telecom"]
N_STOCKS = 20  # Representative stocks in universe
N_ETFS = 3     # SPY, QQQ, IWM


@dataclass
class EquityScenario:
    """Market state for equities desk."""
    # Stock prices
    prices: np.ndarray = field(default_factory=lambda: np.zeros(N_STOCKS))
    # Sector returns (today, %)
    sector_returns: np.ndarray = field(default_factory=lambda: np.zeros(N_SECTORS))
    # Implied vol (30d, %)
    stock_iv: np.ndarray = field(default_factory=lambda: np.full(N_STOCKS, 25.0))
    # Realized vol (30d, %)
    stock_rv: np.ndarray = field(default_factory=lambda: np.full(N_STOCKS, 22.0))
    # Short interest (% of float)
    short_interest: np.ndarray = field(default_factory=lambda: np.full(N_STOCKS, 3.0))
    # Bid-ask spread (bps)
    bid_ask: np.ndarray = field(default_factory=lambda: np.full(N_STOCKS, 5.0))
    # Volume relative to average (1.0 = normal)
    rel_volume: np.ndarray = field(default_factory=lambda: np.ones(N_STOCKS))
    # ETF prices
    etf_prices: np.ndarray = field(default_factory=lambda: np.array([450.0, 380.0, 200.0]))
    # ETF premium/discount to NAV (bps)
    etf_premium: np.ndarray = field(default_factory=lambda: np.zeros(N_ETFS))
    # Market-wide
    spx_level: float = 4500.0
    vix: float = 18.0
    market_breadth: float = 0.5  # % stocks above 200d MA
    put_call_ratio: float = 0.9
    days_to_earnings: np.ndarray = field(default_factory=lambda: np.full(N_STOCKS, 30.0))
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.prices / 500.0,               # 20
            self.sector_returns / 5.0,         # 11
            self.stock_iv / 80.0,              # 20
            self.stock_rv / 80.0,              # 20
            self.short_interest / 30.0,        # 20
            self.bid_ask / 50.0,               # 20
            self.rel_volume / 5.0,             # 20
            self.etf_prices / 500.0,           # 3
            self.etf_premium / 50.0,           # 3
            [self.spx_level / 6000.0],         # 1
            [self.vix / 60.0],                 # 1
            [self.market_breadth],             # 1
            [self.put_call_ratio / 2.0],       # 1
            self.days_to_earnings / 90.0,      # 20
        ])  # Total: 161

    @property
    def obs_dim(self) -> int:
        return 161


@dataclass
class EquityBook:
    """Position book for equities desk."""
    stock_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_STOCKS))  # $M
    etf_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_ETFS))
    # Pairs positions (long - short)
    pairs_long: np.ndarray = field(default_factory=lambda: np.zeros(N_STOCKS))
    pairs_short: np.ndarray = field(default_factory=lambda: np.zeros(N_STOCKS))
    # Block trades pending
    block_pending: float = 0.0
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission_earned: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.stock_positions / 50.0,       # 20
            self.etf_positions / 100.0,        # 3
            self.pairs_long / 20.0,            # 20
            self.pairs_short / 20.0,           # 20
            [self.block_pending / 50.0],       # 1
            [self.realized_pnl / 5000.0],      # 1
            [self.unrealized_pnl / 5000.0],    # 1
            [self.commission_earned / 500.0],  # 1
        ])  # Total: 67

    @property
    def state_dim(self) -> int:
        return 67

    @property
    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.stock_positions)) +
                     np.sum(np.abs(self.etf_positions)) +
                     np.sum(np.abs(self.pairs_long)) +
                     np.sum(np.abs(self.pairs_short)))

    @property
    def net_exposure(self) -> float:
        return float(np.sum(self.stock_positions) + np.sum(self.etf_positions) +
                     np.sum(self.pairs_long) - np.sum(self.pairs_short))


class EqAction(IntEnum):
    NOOP = 0
    BUY_STOCK = 1
    SELL_STOCK = 2
    BUY_ETF = 3
    SELL_ETF = 4
    PAIRS_LONG = 5       # Long stock A, short stock B
    PAIRS_SHORT = 6
    ETF_ARB_BUY = 7      # Buy ETF, sell basket
    ETF_ARB_SELL = 8
    BLOCK_BID = 9        # Bid on block trade
    SECTOR_ROTATE = 10   # Overweight one sector, underweight another
    EARNINGS_TRADE = 11  # Position for earnings event
    FLATTEN = 12
    CLOSE_DAY = 13


class EquityScenarioGenerator:
    REGIMES = ["normal", "bull", "bear", "vol_spike", "rotation", "squeeze"]

    TYPICAL_PRICES = np.concatenate([
        np.random.RandomState(42).uniform(50, 400, N_STOCKS)
    ])

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> EquityScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = EquityScenario(regime=regime)

        # Prices
        for i in range(N_STOCKS):
            s.prices[i] = self.TYPICAL_PRICES[i] * self.rng.uniform(0.8, 1.2)

        # Market levels by regime
        if regime == "bull":
            s.spx_level = self.rng.uniform(4500, 5500)
            s.vix = self.rng.uniform(10, 18)
            s.market_breadth = self.rng.uniform(0.55, 0.80)
        elif regime == "bear":
            s.spx_level = self.rng.uniform(3500, 4200)
            s.vix = self.rng.uniform(22, 40)
            s.market_breadth = self.rng.uniform(0.20, 0.45)
        elif regime == "vol_spike":
            s.spx_level = self.rng.uniform(3800, 4800)
            s.vix = self.rng.uniform(30, 60)
        elif regime == "squeeze":
            s.short_interest = self.rng.uniform(10, 40, N_STOCKS)
            s.vix = self.rng.uniform(20, 35)
        else:
            s.spx_level = self.rng.uniform(4000, 5000)
            s.vix = self.rng.uniform(14, 25)
            s.market_breadth = self.rng.uniform(0.40, 0.65)

        # Sector returns
        for i in range(N_SECTORS):
            s.sector_returns[i] = self.rng.normal(0, 1.0)
            if regime == "rotation":
                s.sector_returns[i] *= 2  # Larger sector moves

        # Stock vols
        for i in range(N_STOCKS):
            s.stock_iv[i] = s.vix * self.rng.uniform(0.7, 1.5) + self.rng.normal(0, 3)
            s.stock_rv[i] = s.stock_iv[i] * self.rng.uniform(0.7, 1.3)

        s.short_interest = np.maximum(0.5, s.short_interest + self.rng.normal(0, 2, N_STOCKS))
        s.bid_ask = np.maximum(1, self.rng.uniform(2, 15, N_STOCKS))
        s.rel_volume = np.maximum(0.2, self.rng.lognormal(0, 0.5, N_STOCKS))
        s.etf_prices = np.array([s.spx_level / 10, s.spx_level / 12, s.spx_level / 22])
        s.etf_premium = self.rng.normal(0, 5, N_ETFS)
        s.put_call_ratio = self.rng.uniform(0.6, 1.4)
        s.days_to_earnings = self.rng.uniform(1, 90, N_STOCKS)

        return s

    def step_scenario(self, s: EquityScenario) -> EquityScenario:
        new = EquityScenario(regime=s.regime)

        for i in range(N_STOCKS):
            daily_vol = s.stock_rv[i] / 100 / np.sqrt(252)
            sector_idx = i % N_SECTORS
            systematic = self.rng.normal(0, 0.005)  # Market factor
            sector = self.rng.normal(0, 0.003)  # Sector factor
            idio = self.rng.normal(0, daily_vol)
            ret = systematic + sector + idio
            new.prices[i] = s.prices[i] * (1 + ret)

        new.sector_returns = self.rng.normal(0, 1.0, N_SECTORS)
        new.stock_iv = np.maximum(8, s.stock_iv + self.rng.normal(0, 0.5, N_STOCKS))
        new.stock_rv = np.maximum(8, s.stock_rv + self.rng.normal(0, 0.7, N_STOCKS))
        new.short_interest = np.maximum(0.5, s.short_interest + self.rng.normal(0, 0.3, N_STOCKS))
        new.bid_ask = np.maximum(1, s.bid_ask + self.rng.normal(0, 0.5, N_STOCKS))
        new.rel_volume = np.maximum(0.2, s.rel_volume * np.exp(self.rng.normal(0, 0.1, N_STOCKS)))
        new.etf_prices = s.etf_prices * (1 + self.rng.normal(0, 0.005, N_ETFS))
        new.etf_premium = s.etf_premium + self.rng.normal(0, 1, N_ETFS)
        new.spx_level = s.spx_level * (1 + self.rng.normal(0, 0.008))
        new.vix = max(8, s.vix + self.rng.normal(0, 1))
        new.market_breadth = np.clip(s.market_breadth + self.rng.normal(0, 0.02), 0.1, 0.9)
        new.put_call_ratio = max(0.3, s.put_call_ratio + self.rng.normal(0, 0.05))
        new.days_to_earnings = np.maximum(0, s.days_to_earnings - 1)

        return new


class EquitiesEnv(gym.Env):
    """Equities cash trading desk. Action: MultiDiscrete([14, 20, 3, 10])"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_days=20, max_actions_per_day=15,
                 gross_limit=500.0, seed=None):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.gross_limit = gross_limit
        self.gen = EquityScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([14, 20, 3, 10])
        self._market_dim = 161
        self._book_dim = 67
        self.observation_space = spaces.Box(-10, 10, (self._market_dim + self._book_dim,), np.float32)
        self._size_buckets = np.linspace(1, 30, 10)

        self.scenario: Optional[EquityScenario] = None
        self.prev_scenario: Optional[EquityScenario] = None
        self.book: Optional[EquityBook] = None
        self.day = 0
        self.actions_today = 0

    def _get_obs(self):
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([self.scenario.to_observation(),
                               self.book.to_vector()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.gen = EquityScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = EquityBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None
        act = EqAction(action[0])
        stock = min(action[1], N_STOCKS - 1)
        etf = min(action[2], N_ETFS - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == EqAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1; self.actions_today = 0
            if self.day >= self.max_days: terminated = True
        elif act == EqAction.NOOP: pass
        elif act == EqAction.BUY_STOCK:
            self.book.stock_positions[stock] += size
            reward = -abs(size) * self.scenario.bid_ask[stock] / 10000 * 1000
        elif act == EqAction.SELL_STOCK:
            self.book.stock_positions[stock] -= size
            reward = -abs(size) * self.scenario.bid_ask[stock] / 10000 * 1000
        elif act == EqAction.BUY_ETF:
            self.book.etf_positions[etf] += size
            reward = -abs(size) * 0.5 / 10000 * 1000
        elif act == EqAction.SELL_ETF:
            self.book.etf_positions[etf] -= size
            reward = -abs(size) * 0.5 / 10000 * 1000
        elif act == EqAction.PAIRS_LONG:
            # Pair within same sector (stat arb: long stock, short sector peer)
            sector = stock % N_SECTORS
            pair_b = ((stock // N_SECTORS + 1) * N_SECTORS + sector) % N_STOCKS
            self.book.pairs_long[stock] += size
            self.book.pairs_short[pair_b] += size
            reward = -abs(size) * 2 * 5 / 10000 * 1000
        elif act == EqAction.PAIRS_SHORT:
            sector = stock % N_SECTORS
            pair_b = ((stock // N_SECTORS + 1) * N_SECTORS + sector) % N_STOCKS
            self.book.pairs_long[pair_b] += size
            self.book.pairs_short[stock] += size
            reward = -abs(size) * 2 * 5 / 10000 * 1000
        elif act == EqAction.ETF_ARB_BUY:
            self.book.etf_positions[etf] += size
            reward = abs(self.scenario.etf_premium[etf]) * size * 0.01
        elif act == EqAction.ETF_ARB_SELL:
            self.book.etf_positions[etf] -= size
            reward = abs(self.scenario.etf_premium[etf]) * size * 0.01
        elif act == EqAction.BLOCK_BID:
            alloc = size * self.gen.rng.uniform(0.3, 1.0)
            self.book.stock_positions[stock] += alloc
            discount = self.gen.rng.uniform(1, 5)  # % discount
            reward = alloc * discount / 100 * 1000 * 0.3
            self.book.commission_earned += abs(alloc) * 0.001 * 1000
        elif act == EqAction.SECTOR_ROTATE:
            sector_a = stock % N_SECTORS
            sector_b = (stock + 3) % N_SECTORS
            for i in range(N_STOCKS):
                if i % N_SECTORS == sector_a:
                    self.book.stock_positions[i] += size / 3
                elif i % N_SECTORS == sector_b:
                    self.book.stock_positions[i] -= size / 3
            reward = -abs(size) * 5 / 10000 * 1000
        elif act == EqAction.EARNINGS_TRADE:
            if self.scenario.days_to_earnings[stock] <= 3:
                self.book.stock_positions[stock] += size
                reward = 0
            else:
                reward = -0.1
        elif act == EqAction.FLATTEN:
            cost = float(np.sum(np.abs(self.book.stock_positions) *
                               self.scenario.bid_ask / 10000 * 1000))
            self.book.stock_positions *= 0.1
            self.book.etf_positions *= 0.1
            self.book.pairs_long *= 0.1
            self.book.pairs_short *= 0.1
            reward = -cost * 0.5

        if self.book.gross_exposure > self.gross_limit: reward -= 5.0
        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1; self.actions_today = 0
            if self.day >= self.max_days: terminated = True

        return self._get_obs(), reward, terminated, truncated, {
            "day": self.day, "gross": self.book.gross_exposure,
            "realized_pnl": self.book.realized_pnl}

    def _end_of_day(self):
        if self.scenario is None or self.book is None: return 0.0
        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        pnl = 0.0
        for i in range(N_STOCKS):
            ret = (self.scenario.prices[i] - self.prev_scenario.prices[i]) / \
                max(0.01, self.prev_scenario.prices[i])
            pnl += self.book.stock_positions[i] * ret * 1000
            pnl += self.book.pairs_long[i] * ret * 1000
            pnl -= self.book.pairs_short[i] * ret * 1000
        for i in range(N_ETFS):
            ret = (self.scenario.etf_prices[i] - self.prev_scenario.etf_prices[i]) / \
                max(0.01, self.prev_scenario.etf_prices[i])
            pnl += self.book.etf_positions[i] * ret * 1000
        # Earnings events
        for i in range(N_STOCKS):
            if self.prev_scenario.days_to_earnings[i] <= 1:
                surprise = self.gen.rng.normal(0, 5)  # % move
                pnl += self.book.stock_positions[i] * surprise / 100 * 1000
        self.book.realized_pnl += pnl
        return float(np.clip(pnl, -100, 100))


def make_equities_env(seed=None, **kwargs): return EquitiesEnv(seed=seed, **kwargs)
