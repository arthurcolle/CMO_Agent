"""
FX Spot & Forwards Trading Desk RL Environment.

Gymnasium-compatible environment for G10 FX trading.
Covers:
  - Spot FX trading
  - FX forwards and swaps (carry trades)
  - Cross-currency basis
  - Options (vanilla calls/puts, risk reversals, strangles)
  - Emerging market FX (high carry, crash risk)
  - Central bank intervention risk
  - Economic data event trading

References:
  - Brunnermeier, Nagel & Pedersen (2008) "Carry Trades and Currency Crashes" NBER
  - Lustig, Roussanov & Verdelhan (2011) "Common Risk Factors in Currency Markets" RFS
  - Menkhoff et al. (2012) "Carry Trades and Global FX Volatility" JF
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


# ─── Currency Universe ────────────────────────────────────────────────────

N_PAIRS = 8  # G10 majors + 2 EM
PAIR_NAMES = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF",
              "AUDUSD", "USDCAD", "NZDUSD", "USDMXN"]
# Which pairs have USD as base (1 = XXXUSD, -1 = USDXXX)
PAIR_CONV = [1, -1, 1, -1, 1, -1, 1, -1]

N_TENORS = 5  # O/N, 1W, 1M, 3M, 1Y forward points


@dataclass
class FXScenario:
    """Market state for FX desk."""
    # Spot rates
    spots: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Forward points (pips) by pair x tenor
    fwd_points: np.ndarray = field(default_factory=lambda: np.zeros((N_PAIRS, N_TENORS)))
    # Implied vol (%) by pair
    implied_vol: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Realized vol (%)
    realized_vol: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Risk reversal (25d) by pair
    risk_reversals: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Cross-currency basis (bps) by pair
    xccy_basis: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Central bank rates for each currency (%)
    cb_rates: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # USD DXY index level
    dxy: float = 100.0
    # VIX
    vix: float = 15.0
    # EM stress indicator (0-10)
    em_stress: float = 3.0
    # Regime
    regime: str = "normal"

    def to_observation(self) -> np.ndarray:
        return np.concatenate([
            self.spots / 200.0,                        # 8
            self.fwd_points.flatten() / 500.0,         # 40
            self.implied_vol / 30.0,                   # 8
            self.realized_vol / 30.0,                  # 8
            self.risk_reversals / 5.0,                 # 8
            self.xccy_basis / 100.0,                   # 8
            self.cb_rates / 15.0,                      # 8
            [self.dxy / 120.0],                        # 1
            [self.vix / 50.0],                         # 1
            [self.em_stress / 10.0],                   # 1
        ])  # Total: 91

    @property
    def obs_dim(self) -> int:
        return 91


@dataclass
class FXBook:
    """Position book for FX desk."""
    # Spot positions ($M base ccy equiv)
    spot_positions: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Forward positions by pair x tenor ($M)
    fwd_positions: np.ndarray = field(default_factory=lambda: np.zeros((N_PAIRS, N_TENORS)))
    # Option delta by pair ($M)
    option_delta: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Option vega by pair ($K per vol pt)
    option_vega: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # Option gamma by pair ($K per % move)
    option_gamma: np.ndarray = field(default_factory=lambda: np.zeros(N_PAIRS))
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.spot_positions / 100.0,               # 8
            self.fwd_positions.flatten() / 50.0,       # 40
            self.option_delta / 50.0,                  # 8
            self.option_vega / 100.0,                  # 8
            self.option_gamma / 50.0,                  # 8
            [self.realized_pnl / 5000.0],              # 1
            [self.unrealized_pnl / 5000.0],            # 1
        ])  # Total: 74

    @property
    def state_dim(self) -> int:
        return 74

    @property
    def total_delta(self) -> float:
        return float(np.sum(np.abs(self.spot_positions)) +
                     np.sum(np.abs(self.fwd_positions)) +
                     np.sum(np.abs(self.option_delta)))

    @property
    def total_vega(self) -> float:
        return float(np.sum(np.abs(self.option_vega)))


class FXAction(IntEnum):
    NOOP = 0
    BUY_SPOT = 1
    SELL_SPOT = 2
    BUY_FORWARD = 3
    SELL_FORWARD = 4
    CARRY_TRADE = 5      # Buy high-yield, sell low-yield
    REVERSE_CARRY = 6
    BUY_CALL = 7
    BUY_PUT = 8
    SELL_STRADDLE = 9    # Vol selling
    BUY_RISK_REV = 10    # Buy RR (long call, short put)
    HEDGE_DELTA = 11     # Delta hedge options
    FLATTEN = 12
    CLOSE_DAY = 13


class FXScenarioGenerator:
    REGIMES = ["normal", "risk_off", "usd_rally", "usd_sell", "em_crisis", "vol_spike"]

    # Typical spot levels
    TYPICAL_SPOTS = [1.08, 150.0, 1.27, 0.88, 0.66, 1.36, 0.62, 17.5]
    # Typical CB rates (ECB, BOJ, BOE, SNB, RBA, BOC, RBNZ, Banxico)
    TYPICAL_CB_RATES = [3.75, 0.25, 5.0, 1.5, 4.35, 4.5, 5.25, 11.0]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(self, regime: Optional[str] = None) -> FXScenario:
        if regime is None:
            regime = self.rng.choice(self.REGIMES)

        s = FXScenario(regime=regime)

        # Spots with regime-dependent variation
        for i in range(N_PAIRS):
            base = self.TYPICAL_SPOTS[i]
            pct_move = self.rng.normal(0, 0.05)
            if regime == "usd_rally":
                pct_move -= 0.03 * PAIR_CONV[i]  # USD strengthens
            elif regime == "usd_sell":
                pct_move += 0.03 * PAIR_CONV[i]
            elif regime == "em_crisis" and i == 7:  # MXN
                pct_move += 0.10  # MXN weakens
            s.spots[i] = base * (1 + pct_move)

        # Central bank rates
        for i in range(N_PAIRS):
            s.cb_rates[i] = max(0, self.TYPICAL_CB_RATES[i] + self.rng.normal(0, 0.5))

        us_rate = self.rng.uniform(3.0, 6.0)

        # Forward points driven by rate differentials
        for i in range(N_PAIRS):
            rate_diff = s.cb_rates[i] - us_rate  # Foreign - USD
            for j, days in enumerate([1, 7, 30, 90, 365]):
                # Forward points = spot * rate_diff * days/365
                base_pts = s.spots[i] * rate_diff / 100 * days / 365 * PAIR_CONV[i]
                s.fwd_points[i, j] = base_pts * 10000 + self.rng.normal(0, 2)

        # Implied vols
        base_vol = 8.0 if regime == "normal" else 12.0
        if regime == "vol_spike":
            base_vol = 16.0
        elif regime == "em_crisis":
            base_vol = 14.0

        for i in range(N_PAIRS):
            vol_mult = 1.0 if i < 7 else 1.5  # EM higher vol
            s.implied_vol[i] = base_vol * vol_mult + self.rng.normal(0, 1.5)
            s.realized_vol[i] = s.implied_vol[i] * self.rng.uniform(0.7, 1.3)

        # Risk reversals (positive = calls over puts, bullish bias)
        for i in range(N_PAIRS):
            s.risk_reversals[i] = self.rng.normal(0, 1.0)
            if regime == "risk_off":
                s.risk_reversals[i] -= 0.5 * PAIR_CONV[i]

        # Cross-currency basis
        for i in range(N_PAIRS):
            s.xccy_basis[i] = self.rng.normal(-15, 10)
            if regime == "risk_off":
                s.xccy_basis[i] -= 10

        s.dxy = self.rng.uniform(90, 115)
        if regime == "usd_rally":
            s.dxy = self.rng.uniform(105, 115)
        elif regime == "usd_sell":
            s.dxy = self.rng.uniform(88, 98)

        s.vix = self.rng.uniform(12, 20) if regime == "normal" else \
            self.rng.uniform(18, 40)
        s.em_stress = self.rng.uniform(1, 4) if regime != "em_crisis" else \
            self.rng.uniform(6, 10)

        return s

    def step_scenario(self, s: FXScenario) -> FXScenario:
        new = FXScenario(regime=s.regime)

        for i in range(N_PAIRS):
            daily_vol = s.realized_vol[i] / 100 / np.sqrt(252)
            shock = self.rng.normal(0, daily_vol)
            new.spots[i] = s.spots[i] * (1 + shock)

        new.cb_rates = s.cb_rates.copy()
        new.fwd_points = s.fwd_points + self.rng.normal(0, 1, (N_PAIRS, N_TENORS))
        new.implied_vol = np.maximum(3, s.implied_vol + self.rng.normal(0, 0.3, N_PAIRS))
        new.realized_vol = np.maximum(3, s.realized_vol + self.rng.normal(0, 0.4, N_PAIRS))
        new.risk_reversals = s.risk_reversals + self.rng.normal(0, 0.1, N_PAIRS)
        new.xccy_basis = s.xccy_basis + self.rng.normal(0, 1, N_PAIRS)
        new.dxy = s.dxy + self.rng.normal(0, 0.3)
        new.vix = max(8, s.vix + self.rng.normal(0, 0.5))
        new.em_stress = np.clip(s.em_stress + self.rng.normal(0, 0.3), 0, 10)

        return new


class FXEnv(gym.Env):
    """
    Gymnasium environment for FX trading desk.

    Action space: MultiDiscrete([14, 8, 5, 10])
      - action_type (14): NOOP through CLOSE_DAY
      - pair_idx (8): currency pair
      - tenor_idx (5): forward tenor
      - size_bucket (10): position size

    Observation: 165-dim
      - Market (91) + Book (74)

    Reward: Daily P&L from spot moves + carry + vol
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 20,
        max_actions_per_day: int = 12,
        delta_limit: float = 500.0,  # $M total delta
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.delta_limit = delta_limit
        self.gen = FXScenarioGenerator(seed=seed)

        self.action_space = spaces.MultiDiscrete([14, 8, 5, 10])

        self._market_dim = 91
        self._book_dim = 74
        obs_dim = self._market_dim + self._book_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._size_buckets = np.linspace(5, 100, 10)  # $M

        self.scenario: Optional[FXScenario] = None
        self.prev_scenario: Optional[FXScenario] = None
        self.book: Optional[FXBook] = None
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
            self.gen = FXScenarioGenerator(seed=seed)
        self.scenario = self.gen.generate()
        self.prev_scenario = None
        self.book = FXBook()
        self.day = 0
        self.actions_today = 0
        return self._get_obs(), {"day": 0, "regime": self.scenario.regime}

    def step(self, action):
        assert self.scenario is not None and self.book is not None

        act = FXAction(action[0])
        pair = min(action[1], N_PAIRS - 1)
        tenor = min(action[2], N_TENORS - 1)
        size = float(self._size_buckets[min(action[3], 9)])

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if act == FXAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == FXAction.NOOP:
            pass
        elif act == FXAction.BUY_SPOT:
            reward = self._trade_spot(pair, size)
        elif act == FXAction.SELL_SPOT:
            reward = self._trade_spot(pair, -size)
        elif act == FXAction.BUY_FORWARD:
            reward = self._trade_fwd(pair, tenor, size)
        elif act == FXAction.SELL_FORWARD:
            reward = self._trade_fwd(pair, tenor, -size)
        elif act == FXAction.CARRY_TRADE:
            reward = self._carry_trade(size)
        elif act == FXAction.REVERSE_CARRY:
            reward = self._carry_trade(-size)
        elif act == FXAction.BUY_CALL:
            reward = self._trade_option(pair, size, is_call=True)
        elif act == FXAction.BUY_PUT:
            reward = self._trade_option(pair, size, is_call=False)
        elif act == FXAction.SELL_STRADDLE:
            reward = self._sell_straddle(pair, size)
        elif act == FXAction.BUY_RISK_REV:
            reward = self._risk_reversal(pair, size)
        elif act == FXAction.HEDGE_DELTA:
            reward = self._delta_hedge(pair)
        elif act == FXAction.FLATTEN:
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
            "total_vega": self.book.total_vega,
            "realized_pnl": self.book.realized_pnl,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _trade_spot(self, pair: int, size: float) -> float:
        self.book.spot_positions[pair] += size
        # Bid/ask: ~1pip G10, ~3-5 pips EM
        spread_pips = 1.0 if pair < 7 else 5.0
        cost = abs(size) * spread_pips / self.scenario.spots[pair] / 100 * 1000
        return -cost

    def _trade_fwd(self, pair: int, tenor: int, size: float) -> float:
        self.book.fwd_positions[pair, tenor] += size
        spread = 2.0 if pair < 7 else 8.0  # Forward pips spread
        cost = abs(size) * spread / self.scenario.spots[pair] / 100 * 1000
        return -cost

    def _carry_trade(self, size: float) -> float:
        # Buy highest yielding, sell lowest
        rates = self.scenario.cb_rates.copy()
        buy_pair = int(np.argmax(rates))
        sell_pair = int(np.argmin(rates))
        self.book.spot_positions[buy_pair] += abs(size)
        self.book.spot_positions[sell_pair] -= abs(size) * 0.5
        cost = abs(size) * 3 / 100 * 1000  # ~3 pips combined
        return -cost

    def _trade_option(self, pair: int, size: float, is_call: bool) -> float:
        vol = self.scenario.implied_vol[pair]
        premium = vol / 100 * np.sqrt(30 / 365) * 0.4 * abs(size) * 1000  # $K approx
        delta_sign = 0.5 if is_call else -0.5
        self.book.option_delta[pair] += delta_sign * abs(size)
        self.book.option_vega[pair] += abs(size) * 0.3
        self.book.option_gamma[pair] += abs(size) * 0.1
        return -premium * 0.1  # Bid/ask = 10% of premium

    def _sell_straddle(self, pair: int, size: float) -> float:
        vol = self.scenario.implied_vol[pair]
        premium = vol / 100 * np.sqrt(30 / 365) * 0.8 * abs(size) * 1000
        self.book.option_vega[pair] -= abs(size) * 0.6
        self.book.option_gamma[pair] -= abs(size) * 0.2
        return premium * 0.05  # Collect half the bid/ask

    def _risk_reversal(self, pair: int, size: float) -> float:
        # Buy call, sell put (or vice versa)
        rr = self.scenario.risk_reversals[pair]
        cost = abs(rr) * abs(size) * 0.01 * 1000  # $K
        self.book.option_delta[pair] += abs(size) * 0.3
        self.book.option_vega[pair] += abs(size) * 0.1
        return -cost * 0.1

    def _delta_hedge(self, pair: int) -> float:
        # Hedge option delta with spot
        delta = self.book.option_delta[pair]
        if abs(delta) < 0.5:
            return 0.0
        self.book.spot_positions[pair] -= delta
        self.book.option_delta[pair] = 0.0
        spread = 1.0 if pair < 7 else 5.0
        cost = abs(delta) * spread / self.scenario.spots[pair] / 100 * 1000
        return -cost

    def _flatten(self) -> float:
        cost = 0.0
        for i in range(N_PAIRS):
            spread = 1.0 if i < 7 else 5.0
            spot = max(0.01, self.scenario.spots[i])
            cost += abs(self.book.spot_positions[i]) * spread / spot / 100 * 1000
            self.book.spot_positions[i] *= 0.1
            self.book.fwd_positions[i] *= 0.1
        self.book.option_delta *= 0.1
        self.book.option_vega *= 0.1
        self.book.option_gamma *= 0.1
        return -cost

    def _end_of_day(self) -> float:
        if self.scenario is None or self.book is None:
            return 0.0

        self.prev_scenario = self.scenario
        self.scenario = self.gen.step_scenario(self.scenario)
        daily_pnl = 0.0

        # Spot P&L
        for i in range(N_PAIRS):
            pct_move = (self.scenario.spots[i] - self.prev_scenario.spots[i]) / \
                self.prev_scenario.spots[i]
            daily_pnl += self.book.spot_positions[i] * pct_move * PAIR_CONV[i] * 1000

        # Forward carry
        for i in range(N_PAIRS):
            total_fwd = float(np.sum(self.book.fwd_positions[i]))
            if abs(total_fwd) > 0.01:
                # Earn rate differential
                rate_diff = self.prev_scenario.cb_rates[i] / 100 / 252
                carry = total_fwd * rate_diff * PAIR_CONV[i] * 1000
                daily_pnl += carry
                # Spot component
                pct_move = (self.scenario.spots[i] - self.prev_scenario.spots[i]) / \
                    self.prev_scenario.spots[i]
                daily_pnl += total_fwd * pct_move * PAIR_CONV[i] * 1000

        # Option gamma P&L
        for i in range(N_PAIRS):
            pct_move = (self.scenario.spots[i] - self.prev_scenario.spots[i]) / \
                self.prev_scenario.spots[i]
            daily_pnl += 0.5 * self.book.option_gamma[i] * (pct_move * 100) ** 2

        # Vega P&L
        for i in range(N_PAIRS):
            vol_change = self.scenario.implied_vol[i] - self.prev_scenario.implied_vol[i]
            daily_pnl += self.book.option_vega[i] * vol_change

        # Delta P&L from options
        for i in range(N_PAIRS):
            pct_move = (self.scenario.spots[i] - self.prev_scenario.spots[i]) / \
                self.prev_scenario.spots[i]
            daily_pnl += self.book.option_delta[i] * pct_move * PAIR_CONV[i] * 1000

        self.book.realized_pnl += daily_pnl
        return float(np.clip(daily_pnl, -100, 100))

    def render(self):
        if self.scenario is None or self.book is None:
            return
        print(f"Day {self.day}/{self.max_days} | Regime: {self.scenario.regime}")
        for i in range(N_PAIRS):
            if abs(self.book.spot_positions[i]) > 0.5 or \
               abs(np.sum(self.book.fwd_positions[i])) > 0.5:
                print(f"  {PAIR_NAMES[i]}: {self.scenario.spots[i]:.4f} "
                      f"pos=${self.book.spot_positions[i]:.0f}M "
                      f"fwd=${np.sum(self.book.fwd_positions[i]):.0f}M")
        print(f"DXY: {self.scenario.dxy:.1f} | VIX: {self.scenario.vix:.1f}")
        print(f"P&L: ${self.book.realized_pnl:.0f}K")


def make_fx_env(seed: Optional[int] = None, **kwargs) -> FXEnv:
    return FXEnv(seed=seed, **kwargs)
