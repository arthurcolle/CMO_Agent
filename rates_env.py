"""
Rates Swaps & Options Trading Desk RL Environment.

Gymnasium-compatible environment for interest rate derivatives trading.
Covers:
  - Vanilla IRS (pay/receive fixed)
  - Swaptions (payer/receiver)
  - Caps/Floors
  - Basis swaps (SOFR vs FF, cross-currency)
  - Curve steepener/flattener swaps
  - Gamma/vega/theta management

References:
  - Hull (2022) "Options, Futures, and Other Derivatives"
  - Andersen & Piterbarg (2010) "Interest Rate Modeling" (3 vols)
  - Rebonato (2002) "Modern Pricing of Interest-Rate Derivatives"
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Market State ─────────────────────────────────────────────────────────

SWAP_TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]  # Standard swap tenors (years)
N_SWAP_TENORS = len(SWAP_TENORS)

SWAPTION_EXPIRIES = [1, 2, 3, 5, 7, 10]  # Option expiries (months/years mapped to indices)
N_EXPIRIES = len(SWAPTION_EXPIRIES)

# Normal vol grid: expiry x tail (subset)
NVOL_GRID_SHAPE = (6, 9)  # 6 expiries x 9 tenors


@dataclass
class RatesScenario:
    """Market state for rates derivatives desk."""
    # Swap curve (par swap rates, %)
    swap_rates: np.ndarray = field(default_factory=lambda: np.zeros(N_SWAP_TENORS))
    # SOFR vs Fed Funds basis (bps)
    sofr_ff_basis: np.ndarray = field(default_factory=lambda: np.zeros(N_SWAP_TENORS))
    # Normal implied vol surface (bps/day annualized)
    nvol_surface: np.ndarray = field(default_factory=lambda: np.zeros(NVOL_GRID_SHAPE))
    # Realized vol (bps/day)
    realized_vol: np.ndarray = field(default_factory=lambda: np.zeros(N_SWAP_TENORS))
    # Vol of vol (skew indicator)
    vol_of_vol: float = 1.0
    # Correlation (2s10s)
    swap_corr_2s10s: float = 0.85
    # Fed state
    fed_funds_rate: float = 5.25
    next_fomc_days: int = 20
    rate_cut_prob: float = 0.3  # Implied from FF futures
    # Macro
    cpi_yoy: float = 3.0
    employment_change_k: float = 200.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.swap_rates / 10.0,                    # 9
            self.sofr_ff_basis / 50.0,                 # 9
            self.nvol_surface.flatten() / 200.0,       # 54
            self.realized_vol / 200.0,                 # 9
            [self.vol_of_vol / 3.0],                   # 1
            [self.swap_corr_2s10s],                    # 1
            [self.fed_funds_rate / 10.0],              # 1
            [self.next_fomc_days / 45.0],              # 1
            [self.rate_cut_prob],                      # 1
            [self.cpi_yoy / 10.0],                     # 1
            [self.employment_change_k / 500.0],        # 1
        ])  # Total: 88

    @property
    def obs_dim(self) -> int:
        return 88


@dataclass
class RatesBook:
    """Position book for rates desk."""
    # DV01 by swap tenor ($K/bp)
    swap_dv01: np.ndarray = field(default_factory=lambda: np.zeros(N_SWAP_TENORS))
    # Gamma by expiry-tenor ($K/bp^2)
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(NVOL_GRID_SHAPE))
    # Vega by expiry-tenor ($K/nvol bp)
    vega: np.ndarray = field(default_factory=lambda: np.zeros(NVOL_GRID_SHAPE))
    # Basis swap DV01 ($K/bp)
    basis_dv01: np.ndarray = field(default_factory=lambda: np.zeros(N_SWAP_TENORS))
    # Net theta ($K/day)
    daily_theta: float = 0.0
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.swap_dv01 / 200.0,           # 9
            self.gamma.flatten() / 50.0,       # 54
            self.vega.flatten() / 100.0,       # 54
            self.basis_dv01 / 100.0,           # 9
            [self.daily_theta / 50.0],         # 1
            [self.realized_pnl / 5000.0],      # 1
            [self.unrealized_pnl / 5000.0],    # 1
        ])  # Total: 129

    @property
    def state_dim(self) -> int:
        return 129

    @property
    def total_dv01(self) -> float:
        return float(np.sum(np.abs(self.swap_dv01)))

    @property
    def total_vega(self) -> float:
        return float(np.sum(np.abs(self.vega)))


class RatesAction(IntEnum):
    NOOP = 0
    PAY_FIXED = 1          # Enter payer swap (short duration)
    RECEIVE_FIXED = 2      # Enter receiver swap (long duration)
    BUY_PAYER_SWPTN = 3    # Buy payer swaption (gamma long, short delta)
    BUY_RECEIVER_SWPTN = 4 # Buy receiver swaption (gamma long, long delta)
    SELL_PAYER_SWPTN = 5   # Sell payer swaption (gamma short)
    SELL_RECEIVER_SWPTN = 6
    BUY_STRADDLE = 7       # Buy straddle (pure vol)
    SELL_STRADDLE = 8      # Sell straddle
    PAY_BASIS = 9          # Pay fixed in basis swap
    RECEIVE_BASIS = 10     # Receive fixed in basis swap
    STEEPENER_SWAP = 11    # DV01-neutral steepener via swaps
    FLATTENER_SWAP = 12    # DV01-neutral flattener
    FLATTEN_BOOK = 13      # Reduce risk
    CLOSE_DAY = 14         # End trading day


class RatesScenarioGenerator:
    REGIMES = ["normal", "hiking", "cutting", "vol_spike", "curve_inversion", "qe"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> RatesScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = RatesScenario(regime=regime)

        # Base swap curve
        if regime == "hiking":
            s.fed_funds_rate = self.rng.uniform(4.5, 6.5)
            front = s.fed_funds_rate + self.rng.uniform(0, 0.5)
            slope = self.rng.uniform(-1.5, 0.3)
        elif regime == "cutting":
            s.fed_funds_rate = self.rng.uniform(1.5, 4.0)
            front = s.fed_funds_rate + self.rng.uniform(-0.2, 0.3)
            slope = self.rng.uniform(0.5, 2.5)
        elif regime == "curve_inversion":
            s.fed_funds_rate = self.rng.uniform(4.0, 6.0)
            front = s.fed_funds_rate + self.rng.uniform(0.2, 0.8)
            slope = self.rng.uniform(-2.0, -0.5)
        elif regime == "qe":
            s.fed_funds_rate = self.rng.uniform(0.0, 0.5)
            front = s.fed_funds_rate + self.rng.uniform(0.0, 0.3)
            slope = self.rng.uniform(1.5, 3.5)
        else:
            s.fed_funds_rate = self.rng.uniform(2.0, 5.0)
            front = s.fed_funds_rate + self.rng.uniform(-0.1, 0.3)
            slope = self.rng.uniform(-0.5, 2.0)

        for i, t in enumerate(SWAP_TENORS):
            s.swap_rates[i] = front + slope * (1 - np.exp(-t / 5.0)) + \
                self.rng.normal(0, 0.05)
            s.swap_rates[i] = max(0.01, s.swap_rates[i])

        # SOFR-FF basis
        for i in range(N_SWAP_TENORS):
            s.sofr_ff_basis[i] = self.rng.uniform(-5, 15) + i * 0.5

        # Normal vol surface
        base_vol = 80 if regime in ("vol_spike",) else self.rng.uniform(40, 90)
        if regime == "vol_spike":
            base_vol = self.rng.uniform(100, 180)
        for ei in range(N_EXPIRIES):
            for ti in range(N_SWAP_TENORS):
                # Vol term structure: short expiry higher, hump
                exp_factor = 1.0 + 0.3 * np.exp(-SWAPTION_EXPIRIES[ei] / 3.0)
                tail_factor = 0.7 + 0.3 * min(SWAP_TENORS[ti], 10) / 10.0
                s.nvol_surface[ei, ti] = base_vol * exp_factor * tail_factor + \
                    self.rng.normal(0, 3)

        # Realized vol
        for i in range(N_SWAP_TENORS):
            s.realized_vol[i] = np.mean(s.nvol_surface[:, i]) * \
                self.rng.uniform(0.7, 1.3)

        s.vol_of_vol = self.rng.uniform(0.5, 2.5)
        s.swap_corr_2s10s = self.rng.uniform(0.6, 0.95)
        s.next_fomc_days = self.rng.randint(1, 42)
        s.rate_cut_prob = self.rng.uniform(0, 1)
        s.cpi_yoy = self.rng.uniform(1.0, 6.0)
        s.employment_change_k = self.rng.uniform(-200, 400)

        return s

    def step_scenario(self, s: RatesScenario) -> RatesScenario:
        new = RatesScenario(regime=s.regime)

        # Evolve swap curve
        vol_scale = 1.5 if s.regime == "vol_spike" else 1.0
        for i in range(N_SWAP_TENORS):
            daily_vol = s.realized_vol[i] / 100.0 / np.sqrt(252)
            shock = self.rng.normal(0, daily_vol) * vol_scale
            new.swap_rates[i] = max(0.01, s.swap_rates[i] + shock)

        new.sofr_ff_basis = s.sofr_ff_basis + self.rng.normal(0, 0.5, N_SWAP_TENORS)
        new.nvol_surface = np.maximum(20, s.nvol_surface + self.rng.normal(0, 1.5, NVOL_GRID_SHAPE))
        new.realized_vol = np.maximum(10, s.realized_vol + self.rng.normal(0, 2, N_SWAP_TENORS))
        new.vol_of_vol = max(0.3, s.vol_of_vol + self.rng.normal(0, 0.1))
        new.swap_corr_2s10s = np.clip(s.swap_corr_2s10s + self.rng.normal(0, 0.02), 0.3, 0.99)
        new.fed_funds_rate = s.fed_funds_rate
        new.next_fomc_days = max(0, s.next_fomc_days - 1)
        new.rate_cut_prob = np.clip(s.rate_cut_prob + self.rng.normal(0, 0.05), 0, 1)
        new.cpi_yoy = s.cpi_yoy
        new.employment_change_k = s.employment_change_k

        # FOMC event
        if new.next_fomc_days == 0:
            if self.rng.random() < s.rate_cut_prob:
                new.fed_funds_rate = max(0, s.fed_funds_rate - 0.25)
            elif self.rng.random() < 0.2:
                new.fed_funds_rate = s.fed_funds_rate + 0.25
            new.next_fomc_days = self.rng.randint(28, 42)
            new.rate_cut_prob = self.rng.uniform(0.1, 0.5)

        return new


class RatesEnv(gym.Env):
    """
    Gymnasium environment for rates swaps & options trading desk.

    Action space: MultiDiscrete([15, 9, 6, 10])
      - action_type (15): NOOP through CLOSE_DAY
      - tenor_idx (9): swap tenor
      - expiry_idx (6): option expiry
      - size_bucket (10): notional size

    Observation: 217-dim
      - Market (88) + Book (129)

    Reward: Daily P&L from carry/theta + delta/gamma + vega
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 20,
        max_actions_per_day: int = 12,
        dv01_limit: float = 800.0,
        vega_limit: float = 2000.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.dv01_limit = dv01_limit
        self.vega_limit = vega_limit
        self.gen = RatesScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([15, 9, 6, 10])

        self._market_dim = 88
        self._book_dim = 129
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32,
        )

        self._size_buckets = np.linspace(10, 200, 10)  # $M notional

        self.scenario: Optional[RatesScenario] = None
        self.prev_scenario: Optional[RatesScenario] = None
        self.book: Optional[RatesBook] = None
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
            self.gen = RatesScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = RatesBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        act = RatesAction(action[0])
        tenor_idx = min(action[1], N_SWAP_TENORS - 1)
        expiry_idx = min(action[2], N_EXPIRIES - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == RatesAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == RatesAction.NOOP:
            pass
        elif act == RatesAction.PAY_FIXED:
            reward = self._enter_swap(tenor_idx, -size)
        elif act == RatesAction.RECEIVE_FIXED:
            reward = self._enter_swap(tenor_idx, size)
        elif act == RatesAction.BUY_PAYER_SWPTN:
            reward = self._trade_swaption(expiry_idx, tenor_idx, size, is_payer=True, is_buy=True)
        elif act == RatesAction.BUY_RECEIVER_SWPTN:
            reward = self._trade_swaption(expiry_idx, tenor_idx, size, is_payer=False, is_buy=True)
        elif act == RatesAction.SELL_PAYER_SWPTN:
            reward = self._trade_swaption(expiry_idx, tenor_idx, size, is_payer=True, is_buy=False)
        elif act == RatesAction.SELL_RECEIVER_SWPTN:
            reward = self._trade_swaption(expiry_idx, tenor_idx, size, is_payer=False, is_buy=False)
        elif act == RatesAction.BUY_STRADDLE:
            reward = self._trade_straddle(expiry_idx, tenor_idx, size, is_buy=True)
        elif act == RatesAction.SELL_STRADDLE:
            reward = self._trade_straddle(expiry_idx, tenor_idx, size, is_buy=False)
        elif act == RatesAction.PAY_BASIS:
            reward = self._basis_swap(tenor_idx, -size)
        elif act == RatesAction.RECEIVE_BASIS:
            reward = self._basis_swap(tenor_idx, size)
        elif act == RatesAction.STEEPENER_SWAP:
            reward = self._curve_swap(tenor_idx, size)
        elif act == RatesAction.FLATTENER_SWAP:
            reward = self._curve_swap(tenor_idx, -size)
        elif act == RatesAction.FLATTEN_BOOK:
            reward = self._flatten()

        # Risk limits
        if self.book.total_dv01 > self.dv01_limit:
            reward -= 5.0
        if self.book.total_vega > self.vega_limit:
            reward -= 3.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {
            "day": self.day,
            "total_dv01": self.book.total_dv01,
            "total_vega": self.book.total_vega,
            "realized_pnl": self.book.realized_pnl,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _enter_swap(self, tenor_idx: int, size: float) -> float:
        """Enter vanilla IRS."""
        tenor = SWAP_TENORS[tenor_idx]
        dv01_per_m = tenor * 0.01 * 10  # $K DV01 per $M notional
        self.book.swap_dv01[tenor_idx] += np.sign(size) * abs(size) * dv01_per_m / 1000
        # Bid/ask in swaps ~0.25bp for liquid tenors
        cost = abs(size) * 0.25 / 10000 * tenor * 10  # $K
        return -cost

    def _trade_swaption(self, expiry_idx: int, tenor_idx: int, size: float,
                        is_payer: bool, is_buy: bool) -> float:
        """Trade a swaption."""
        nvol = self.scenario.nvol_surface[expiry_idx, tenor_idx]
        tenor = SWAP_TENORS[tenor_idx]
        expiry = SWAPTION_EXPIRIES[expiry_idx]

        # Approximate Black premium in bps running
        premium_bps = nvol * np.sqrt(expiry) * 0.4  # simplified
        premium_per_m = premium_bps / 10000 * tenor * 10 * 1000  # $K per $M

        sign = 1 if is_buy else -1
        delta_sign = -1 if is_payer else 1  # Payer swaption has negative delta
        notional = abs(size)

        # Greeks per $M notional (simplified)
        gamma_per_m = notional * 0.001 * tenor  # $K/bp^2
        vega_per_m = notional * tenor * 0.01    # $K/nvol bp
        dv01_per_m = notional * tenor * 0.005 * delta_sign  # Delta hedge

        self.book.gamma[expiry_idx, tenor_idx] += sign * gamma_per_m
        self.book.vega[expiry_idx, tenor_idx] += sign * vega_per_m
        self.book.swap_dv01[tenor_idx] += sign * dv01_per_m

        if is_buy:
            self.book.daily_theta -= notional * premium_bps / 10000 * tenor / \
                max(1, expiry * 252) * 1000  # daily theta bleed
        else:
            self.book.daily_theta += notional * premium_bps / 10000 * tenor / \
                max(1, expiry * 252) * 1000

        cost = notional * premium_per_m / 1000  # bid/ask ~10% of premium
        return -abs(cost) * 0.1

    def _trade_straddle(self, expiry_idx: int, tenor_idx: int, size: float,
                        is_buy: bool) -> float:
        """Trade a straddle (payer + receiver)."""
        cost_p = self._trade_swaption(expiry_idx, tenor_idx, size, True, is_buy)
        cost_r = self._trade_swaption(expiry_idx, tenor_idx, size, False, is_buy)
        return cost_p + cost_r

    def _basis_swap(self, tenor_idx: int, size: float) -> float:
        """Enter basis swap (SOFR vs FF)."""
        self.book.basis_dv01[tenor_idx] += np.sign(size) * abs(size) * 0.01
        cost = abs(size) * 0.1 / 10000 * SWAP_TENORS[tenor_idx] * 10
        return -cost

    def _curve_swap(self, tenor_idx: int, size: float) -> float:
        """DV01-neutral curve trade via swaps."""
        short_idx = max(0, tenor_idx - 2)
        long_idx = min(N_SWAP_TENORS - 1, tenor_idx + 2)
        t_short = SWAP_TENORS[short_idx]
        t_long = SWAP_TENORS[long_idx]
        dv01_s = t_short * 0.01 * 10
        dv01_l = t_long * 0.01 * 10
        notional = abs(size)
        sign = np.sign(size)
        self.book.swap_dv01[short_idx] -= sign * notional * dv01_s / 1000
        self.book.swap_dv01[long_idx] += sign * notional * dv01_l / 1000
        cost = 2 * notional * 0.25 / 10000 * (t_short + t_long) / 2 * 10
        return -cost

    def _flatten(self) -> float:
        cost = 0.0
        for i in range(N_SWAP_TENORS):
            cost += abs(self.book.swap_dv01[i]) * 0.5
            self.book.swap_dv01[i] *= 0.1
            cost += abs(self.book.basis_dv01[i]) * 0.3
            self.book.basis_dv01[i] *= 0.1
        self.book.gamma *= 0.1
        self.book.vega *= 0.1
        self.book.daily_theta *= 0.1
        return -cost

    def _end_of_day(self) -> float:
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        daily_pnl = 0.0

        # Delta P&L
        for i in range(N_SWAP_TENORS):
            rate_change_bps = (self.scenario.swap_rates[i] - self.prev_scenario.swap_rates[i]) * 100
            daily_pnl -= self.book.swap_dv01[i] * rate_change_bps

        # Gamma P&L
        for ei in range(N_EXPIRIES):
            for ti in range(N_SWAP_TENORS):
                rate_change = (self.scenario.swap_rates[ti] - self.prev_scenario.swap_rates[ti]) * 100
                daily_pnl += 0.5 * self.book.gamma[ei, ti] * rate_change ** 2

        # Vega P&L
        for ei in range(N_EXPIRIES):
            for ti in range(N_SWAP_TENORS):
                vol_change = self.scenario.nvol_surface[ei, ti] - self.prev_scenario.nvol_surface[ei, ti]
                daily_pnl += self.book.vega[ei, ti] * vol_change * 0.1

        # Basis P&L
        for i in range(N_SWAP_TENORS):
            basis_change = self.scenario.sofr_ff_basis[i] - self.prev_scenario.sofr_ff_basis[i]
            daily_pnl += self.book.basis_dv01[i] * basis_change * 10

        # Theta
        daily_pnl += self.book.daily_theta

        # Carry on swaps (receive fixed earns swap rate - SOFR)
        for i in range(N_SWAP_TENORS):
            if abs(self.book.swap_dv01[i]) > 0.01:
                net_rate = self.prev_scenario.swap_rates[i] - self.prev_scenario.fed_funds_rate
                carry = self.book.swap_dv01[i] * net_rate / 252 * 10
                daily_pnl += carry

        self.book.realized_pnl += daily_pnl
        return float(np.clip(daily_pnl, -100, 100))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        print(f"2Y={self.scenario.swap_rates[1]:.2f}% 5Y={self.scenario.swap_rates[3]:.2f}% "
              f"10Y={self.scenario.swap_rates[5]:.2f}% 30Y={self.scenario.swap_rates[8]:.2f}%")
        print(f"Total DV01: ${self.book.total_dv01:.0f}K | Total Vega: ${self.book.total_vega:.0f}K")
        print(f"Theta: ${self.book.daily_theta:.1f}K/day | P&L: ${self.book.realized_pnl:.0f}K")


def make_rates_env(seed: Optional[int] = None, **kwargs) -> RatesEnv:
    return RatesEnv(seed=seed, **kwargs)
