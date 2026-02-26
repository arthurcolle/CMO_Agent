"""
Equity Derivatives Trading Desk RL Environment.

Gymnasium-compatible environment for equity options and vol trading.
Covers:
  - Vanilla options market-making (calls/puts)
  - Vol surface trading (skew, term structure)
  - Dispersion trading (index vol vs single-stock)
  - Variance/vol swaps
  - Convertible bond arbitrage
  - Greeks management (delta, gamma, vega, theta, vanna, volga)
  - Exotic risk (barriers, autocallables)

References:
  - Gatheral (2006) "The Volatility Surface"
  - Carr & Wu (2009) "Variance Risk Premiums" RFS
  - Bakshi & Kapadia (2003) "Delta-Hedged Gains" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


N_UNDERLYINGS = 5   # SPX, NDX, single stock 1-3
N_EXPIRIES = 6      # 1W, 1M, 2M, 3M, 6M, 1Y
N_STRIKES = 5       # 80%, 90%, ATM, 110%, 120%


@dataclass
class EqDerivScenario:
    """Market state for equity derivatives desk."""
    # Spot levels
    spots: np.ndarray = field(default_factory=lambda: np.array([4500, 15000, 200, 150, 300.0]))
    # Implied vol surface: underlying x expiry x strike (%)
    vol_surface: np.ndarray = field(
        default_factory=lambda: np.full((N_UNDERLYINGS, N_EXPIRIES, N_STRIKES), 20.0))
    # Realized vol by underlying (30d, %)
    realized_vol: np.ndarray = field(default_factory=lambda: np.full(N_UNDERLYINGS, 18.0))
    # Vol of vol
    vvix: float = 100.0
    # Correlation (index vs singles)
    impl_correlation: float = 0.65
    real_correlation: float = 0.60
    # Variance swap levels (variance points) by underlying
    var_swap_levels: np.ndarray = field(default_factory=lambda: np.full(N_UNDERLYINGS, 400.0))
    # Skew (25d put - 25d call vol, %)
    skew: np.ndarray = field(default_factory=lambda: np.full(N_UNDERLYINGS, 5.0))
    # Term structure slope (6M vol - 1M vol, %)
    term_slope: np.ndarray = field(default_factory=lambda: np.full(N_UNDERLYINGS, 2.0))
    # Dividends (annualized yield %)
    div_yield: np.ndarray = field(default_factory=lambda: np.full(N_UNDERLYINGS, 1.5))
    # Risk-free rate
    risk_free: float = 5.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.spots / np.array([6000, 20000, 400, 300, 500]),  # 5
            self.vol_surface.flatten() / 80.0,     # 150
            self.realized_vol / 60.0,               # 5
            [self.vvix / 200.0],                    # 1
            [self.impl_correlation],                # 1
            [self.real_correlation],                # 1
            self.var_swap_levels / 2000.0,          # 5
            self.skew / 15.0,                       # 5
            self.term_slope / 10.0,                 # 5
            self.div_yield / 5.0,                   # 5
            [self.risk_free / 10.0],                # 1
        ])  # Total: 184

    @property
    def obs_dim(self) -> int:
        return 184


@dataclass
class EqDerivBook:
    """Position book for equity derivatives desk."""
    # Delta by underlying ($M per 1% move)
    delta: np.ndarray = field(default_factory=lambda: np.zeros(N_UNDERLYINGS))
    # Gamma by underlying ($K per 1% squared)
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(N_UNDERLYINGS))
    # Vega by underlying ($K per vol point)
    vega: np.ndarray = field(default_factory=lambda: np.zeros(N_UNDERLYINGS))
    # Theta ($K per day)
    theta: float = 0.0
    # Vanna by underlying ($K per 1% spot per vol pt)
    vanna: np.ndarray = field(default_factory=lambda: np.zeros(N_UNDERLYINGS))
    # Variance swap notional ($K per variance pt)
    var_swap: np.ndarray = field(default_factory=lambda: np.zeros(N_UNDERLYINGS))
    # Dispersion position ($K)
    dispersion: float = 0.0
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.delta / 50.0,                  # 5
            self.gamma / 100.0,                 # 5
            self.vega / 200.0,                  # 5
            [self.theta / 100.0],               # 1
            self.vanna / 50.0,                  # 5
            self.var_swap / 50.0,               # 5
            [self.dispersion / 100.0],          # 1
            [self.realized_pnl / 5000.0],       # 1
            [self.unrealized_pnl / 5000.0],     # 1
        ])  # Total: 29

    @property
    def state_dim(self) -> int:
        return 29

    @property
    def total_vega(self) -> float:
        return float(np.sum(np.abs(self.vega)))

    @property
    def total_gamma(self) -> float:
        return float(np.sum(np.abs(self.gamma)))


class EqDerivAction(IntEnum):
    NOOP = 0
    BUY_CALL = 1
    SELL_CALL = 2
    BUY_PUT = 3
    SELL_PUT = 4
    BUY_STRADDLE = 5
    SELL_STRADDLE = 6
    BUY_RISK_REV = 7      # Buy call, sell put (bullish skew)
    SELL_RISK_REV = 8
    BUY_CALENDAR = 9      # Buy far, sell near (vol term)
    SELL_CALENDAR = 10
    LONG_VAR_SWAP = 11    # Buy variance
    SHORT_VAR_SWAP = 12   # Sell variance
    DISPERSION_LONG = 13  # Long single-name vol, short index vol
    DISPERSION_SHORT = 14
    DELTA_HEDGE = 15
    FLATTEN = 16
    CLOSE_DAY = 17


class EqDerivScenarioGenerator:
    REGIMES = ["normal", "low_vol", "high_vol", "skew_rich", "term_steep", "dispersion"]

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime=None):
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = EqDerivScenario(regime=regime)
        s.spots = np.array([4500, 15000, 200, 150, 300.0]) * \
            self.rng.uniform(0.85, 1.15, N_UNDERLYINGS)

        base_vol = {"low_vol": 12, "high_vol": 35, "normal": 20}.get(regime, 20)

        for u in range(N_UNDERLYINGS):
            for e in range(N_EXPIRIES):
                for k in range(N_STRIKES):
                    moneyness = [0.80, 0.90, 1.00, 1.10, 1.20][k]
                    # Vol smile: higher OTM puts
                    skew_adj = max(0, (1.0 - moneyness)) * 15
                    # Term structure
                    term_adj = e * 0.5
                    s.vol_surface[u, e, k] = base_vol + skew_adj + term_adj + \
                        self.rng.normal(0, 1.5)

        s.realized_vol = np.mean(s.vol_surface[:, 1, 2], axis=-1) * \
            self.rng.uniform(0.7, 1.3, N_UNDERLYINGS)  # Scalar per underlying
        s.vvix = self.rng.uniform(60, 150)
        s.impl_correlation = self.rng.uniform(0.4, 0.85)
        s.real_correlation = s.impl_correlation * self.rng.uniform(0.8, 1.2)
        s.var_swap_levels = np.mean(s.vol_surface[:, 2, 2]) ** 2 / 100 * \
            self.rng.uniform(0.9, 1.1, N_UNDERLYINGS)
        s.skew = np.array([s.vol_surface[u, 1, 0] - s.vol_surface[u, 1, 4]
                           for u in range(N_UNDERLYINGS)])
        s.term_slope = np.array([s.vol_surface[u, 4, 2] - s.vol_surface[u, 1, 2]
                                 for u in range(N_UNDERLYINGS)])
        s.risk_free = self.rng.uniform(2, 6)

        return s

    def step_scenario(self, s):
        new = EqDerivScenario(regime=s.regime)
        for u in range(N_UNDERLYINGS):
            daily_vol = s.realized_vol[u] / 100 / np.sqrt(252)
            new.spots[u] = s.spots[u] * (1 + self.rng.normal(0, daily_vol))

        new.vol_surface = np.maximum(5, s.vol_surface + self.rng.normal(0, 0.3, s.vol_surface.shape))
        new.realized_vol = np.maximum(5, s.realized_vol + self.rng.normal(0, 0.5, N_UNDERLYINGS))
        new.vvix = max(40, s.vvix + self.rng.normal(0, 3))
        new.impl_correlation = np.clip(s.impl_correlation + self.rng.normal(0, 0.01), 0.2, 0.95)
        new.real_correlation = np.clip(s.real_correlation + self.rng.normal(0, 0.02), 0.2, 0.95)
        new.var_swap_levels = np.maximum(50, s.var_swap_levels + self.rng.normal(0, 10, N_UNDERLYINGS))
        new.skew = s.skew + self.rng.normal(0, 0.2, N_UNDERLYINGS)
        new.term_slope = s.term_slope + self.rng.normal(0, 0.1, N_UNDERLYINGS)
        new.risk_free = s.risk_free
        new.div_yield = s.div_yield

        return new


class EqDerivativesEnv(gym.Env):
    """Equity derivatives desk. Action: MultiDiscrete([18, 5, 6, 5, 10])"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_days=20, max_actions_per_day=15,
                 vega_limit=1000.0, seed=None):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.vega_limit = vega_limit
        self.gen = EqDerivScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([18, 5, 6, 5, 10])
        self._market_dim = 184
        self._book_dim = 29
        self.observation_space = spaces.Box(-10, 10,
            (self._market_dim + self._book_dim,), np.float32)
        self._size_buckets = np.linspace(5, 100, 10)

        self.scenario: Optional[EqDerivScenario] = None
        self.prev_scenario: Optional[EqDerivScenario] = None
        self.book: Optional[EqDerivBook] = None
        self.day = 0
        self.actions_today = 0

    def _get_obs(self):
        if self.scenario is None or self.book is None:
            return np.zeros(self._market_dim + self._book_dim, dtype=np.float32)
        return np.concatenate([self.scenario.to_observation(),
                               self.book.to_vector()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.gen = EqDerivScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = EqDerivBook()
        self.day = 0; self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None
        act = EqDerivAction(action[0])
        und = min(action[1], N_UNDERLYINGS - 1)
        exp = min(action[2], N_EXPIRIES - 1)
        strike = min(action[3], N_STRIKES - 1)
        size = float(self._size_buckets[min(action[4], 9)])

        reward = 0.0
        terminated = False
        self.actions_today += 1

        vol = self.scenario.vol_surface[und, exp, strike]

        # Variance risk premium: IV typically trades above RV by 2-4 vol pts
        iv_rv_spread = (vol - self.scenario.realized_vol[und]) / 100.0
        # Correlation risk premium: impl > real
        corr_premium = self.scenario.impl_correlation - self.scenario.real_correlation

        if act == EqDerivAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1; self.actions_today = 0
            if self.day >= self.max_days: terminated = True
        elif act == EqDerivAction.NOOP: pass
        elif act in (EqDerivAction.BUY_CALL, EqDerivAction.BUY_PUT):
            delta_sign = 0.4 if act == EqDerivAction.BUY_CALL else -0.4
            self.book.delta[und] += delta_sign * size * 0.01
            self.book.gamma[und] += size * 0.002
            self.book.vega[und] += size * 0.1
            self.book.theta -= size * vol / 100 / np.sqrt(252) * 0.5
            # Bid/ask cost (reduced) + long vol edge when RV > IV
            bid_ask = size * vol / 100 * 0.02
            long_vol_edge = size * max(0, -iv_rv_spread) * 2.0
            reward = -bid_ask + long_vol_edge
        elif act in (EqDerivAction.SELL_CALL, EqDerivAction.SELL_PUT):
            delta_sign = -0.4 if act == EqDerivAction.SELL_CALL else 0.4
            self.book.delta[und] += delta_sign * size * 0.01
            self.book.gamma[und] -= size * 0.002
            self.book.vega[und] -= size * 0.1
            self.book.theta += size * vol / 100 / np.sqrt(252) * 0.5
            # Bid/ask cost + variance risk premium edge (the core business)
            bid_ask = size * vol / 100 * 0.02
            short_vol_edge = size * max(0, iv_rv_spread) * 3.0
            reward = -bid_ask + short_vol_edge
        elif act in (EqDerivAction.BUY_STRADDLE, EqDerivAction.SELL_STRADDLE):
            sign = 1 if act == EqDerivAction.BUY_STRADDLE else -1
            self.book.gamma[und] += sign * size * 0.004
            self.book.vega[und] += sign * size * 0.2
            self.book.theta -= sign * size * vol / 100 / np.sqrt(252)
            # Selling straddles captures variance risk premium
            bid_ask = size * vol / 100 * 0.04
            vrp_edge = -sign * size * iv_rv_spread * 4.0
            reward = -bid_ask + vrp_edge
        elif act in (EqDerivAction.BUY_RISK_REV, EqDerivAction.SELL_RISK_REV):
            sign = 1 if act == EqDerivAction.BUY_RISK_REV else -1
            self.book.delta[und] += sign * 0.5 * size * 0.01
            self.book.vanna[und] += sign * size * 0.05
            # Skew trade: earn carry from skew overpricing
            skew_edge = abs(self.scenario.skew[und]) * size * 0.001 * 0.3
            reward = -size * 0.1 + skew_edge
        elif act in (EqDerivAction.BUY_CALENDAR, EqDerivAction.SELL_CALENDAR):
            sign = 1 if act == EqDerivAction.BUY_CALENDAR else -1
            self.book.vega[und] += sign * size * 0.05
            self.book.theta -= sign * size * 0.1
            # Term structure carry: long front/short back earns roll-down
            term_carry = sign * self.scenario.term_slope[und] * size * 0.05
            reward = -size * 0.3 + term_carry
        elif act in (EqDerivAction.LONG_VAR_SWAP, EqDerivAction.SHORT_VAR_SWAP):
            sign = 1 if act == EqDerivAction.LONG_VAR_SWAP else -1
            self.book.var_swap[und] += sign * size * 0.01
            self.book.vega[und] += sign * size * 0.15
            # Short var swap is the classic VRP trade
            vrp_carry = -sign * size * iv_rv_spread * 5.0
            reward = -size * 0.2 + vrp_carry
        elif act in (EqDerivAction.DISPERSION_LONG, EqDerivAction.DISPERSION_SHORT):
            sign = 1 if act == EqDerivAction.DISPERSION_LONG else -1
            self.book.dispersion += sign * size
            for u in range(1, N_UNDERLYINGS):
                self.book.vega[u] += sign * size * 0.03
            self.book.vega[0] -= sign * size * 0.1
            # Short correlation (long dispersion) earns correlation risk premium
            disp_edge = sign * corr_premium * size * 10.0
            reward = -size * 0.5 + disp_edge
        elif act == EqDerivAction.DELTA_HEDGE:
            cost = abs(self.book.delta[und]) * 0.005
            self.book.delta[und] = 0
            reward = -cost
        elif act == EqDerivAction.FLATTEN:
            cost = float(np.sum(np.abs(self.book.vega))) * 0.02
            self.book.delta *= 0.1; self.book.gamma *= 0.1
            self.book.vega *= 0.1; self.book.theta *= 0.1
            self.book.vanna *= 0.1; self.book.var_swap *= 0.1
            self.book.dispersion *= 0.1
            reward = -cost

        if self.book.total_vega > self.vega_limit: reward -= 2.0
        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1; self.actions_today = 0
            if self.day >= self.max_days: terminated = True

        return self._get_obs(), reward, terminated, False, {
            "day": self.day, "vega": self.book.total_vega,
            "gamma": self.book.total_gamma, "pnl": self.book.realized_pnl}

    def _end_of_day(self):
        if self.scenario is None or self.book is None: return 0.0
        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        pnl = 0.0

        for u in range(N_UNDERLYINGS):
            pct = (self.scenario.spots[u] - self.prev_scenario.spots[u]) / \
                max(0.01, self.prev_scenario.spots[u])
            pnl += self.book.delta[u] * pct * 1000
            pnl += 0.5 * self.book.gamma[u] * (pct * 100) ** 2
            vol_chg = np.mean(self.scenario.vol_surface[u]) - np.mean(self.prev_scenario.vol_surface[u])
            pnl += self.book.vega[u] * vol_chg * 5.0
            pnl += self.book.vanna[u] * pct * 100 * vol_chg * 2.0
            var_chg = self.scenario.var_swap_levels[u] - self.prev_scenario.var_swap_levels[u]
            pnl += self.book.var_swap[u] * var_chg * 50

        # Theta (daily decay — this is the premium the desk collects)
        pnl += self.book.theta * 2.0

        # Dispersion P&L from correlation moves
        corr_move = self.scenario.impl_correlation - self.prev_scenario.impl_correlation
        pnl -= self.book.dispersion * corr_move * 500

        self.book.realized_pnl += pnl
        return float(np.clip(pnl, -500, 500))


def make_eq_derivatives_env(seed=None, **kwargs): return EqDerivativesEnv(seed=seed, **kwargs)
