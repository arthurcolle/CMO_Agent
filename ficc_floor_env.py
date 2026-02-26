"""
Unified FICC Trading Floor RL Environment.

Meta-environment that wraps all 9 FICC trading desks into a single
multi-desk trading floor with cross-desk risk management.

Desks:
  1. MBS/CMO Structuring (yield_book_env)
  2. US Treasuries (treasuries_env)
  3. Distressed Credit (distressed_credit_env)
  4. Rates Swaps & Options (rates_env)
  5. IG Credit (ig_credit_env)
  6. Municipals (munis_env)
  7. Repo / Financing (repo_env)
  8. FX (fx_env)
  9. Commodities (commodities_env)

The meta-agent allocates risk capital across desks, manages firm-wide
VaR and P&L limits, and coordinates cross-desk hedges.

References:
  - Basak & Shapiro (2001) "Value-at-Risk Based Risk Management" RFS
  - Daníelsson, Shin & Zigrand (2004) "The Impact of Risk Regulation" JFE
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

from cmo_agent.treasuries_env import TreasuriesEnv
from cmo_agent.distressed_credit_env import DistressedCreditEnv
from cmo_agent.rates_env import RatesEnv
from cmo_agent.ig_credit_env import IGCreditEnv
from cmo_agent.munis_env import MunisEnv
from cmo_agent.repo_env import RepoEnv
from cmo_agent.fx_env import FXEnv
from cmo_agent.commodities_env import CommoditiesEnv


# ─── Desk Registry ───────────────────────────────────────────────────────

N_DESKS = 9
DESK_NAMES = [
    "MBS/CMO", "Treasuries", "Distressed", "Rates",
    "IG Credit", "Munis", "Repo", "FX", "Commodities",
]

# Risk weights by desk (contribution to firm VaR, rough)
RISK_WEIGHTS = [1.0, 0.8, 2.0, 1.5, 0.7, 0.5, 0.3, 1.2, 1.5]

# Typical daily P&L volatility ($K) by desk
DESK_VOL = [80, 50, 120, 100, 40, 30, 15, 60, 80]

# Cross-desk correlation matrix (simplified)
# Rates and Treasuries highly correlated, credit desks correlated, etc.
DESK_CORRELATION = np.array([
    #  MBS   TSY  DIST  RATE  IG    MUNI  REPO  FX    CMDTY
    [1.00, 0.70, 0.20, 0.65, 0.30, 0.50, 0.40, 0.10, 0.05],  # MBS
    [0.70, 1.00, 0.15, 0.80, 0.25, 0.55, 0.50, 0.15, 0.05],  # TSY
    [0.20, 0.15, 1.00, 0.10, 0.60, 0.15, 0.05, 0.10, 0.15],  # DIST
    [0.65, 0.80, 0.10, 1.00, 0.30, 0.45, 0.35, 0.20, 0.10],  # RATE
    [0.30, 0.25, 0.60, 0.30, 1.00, 0.25, 0.10, 0.15, 0.10],  # IG
    [0.50, 0.55, 0.15, 0.45, 0.25, 1.00, 0.20, 0.05, 0.05],  # MUNI
    [0.40, 0.50, 0.05, 0.35, 0.10, 0.20, 1.00, 0.10, 0.05],  # REPO
    [0.10, 0.15, 0.10, 0.20, 0.15, 0.05, 0.10, 1.00, 0.30],  # FX
    [0.05, 0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0.30, 1.00],  # CMDTY
])


@dataclass
class FloorState:
    """Aggregated state of the entire FICC floor."""
    # Per-desk metrics
    desk_pnl: np.ndarray = field(default_factory=lambda: np.zeros(N_DESKS))
    desk_risk_usage: np.ndarray = field(default_factory=lambda: np.zeros(N_DESKS))  # 0-1
    desk_capital_alloc: np.ndarray = field(default_factory=lambda: np.ones(N_DESKS) / N_DESKS)
    # Firm-wide
    total_pnl: float = 0.0
    firm_var_95: float = 0.0  # 1-day 95% VaR ($K)
    firm_var_99: float = 0.0
    capital_utilization: float = 0.0  # 0-1
    # Limits
    daily_loss_limit: float = -500.0  # $K
    var_limit: float = 1000.0  # $K
    # Running stats
    sharpe_running: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    day: int = 0


class FloorAction(IntEnum):
    """Meta-level actions for the floor manager."""
    NOOP = 0
    INCREASE_CAPITAL = 1    # Give more capital to a desk
    DECREASE_CAPITAL = 2    # Pull capital from a desk
    STEP_DESK = 3           # Let a desk take its action
    HEDGE_RATES_TSY = 4     # Cross-desk hedge: rates vs TSY
    HEDGE_CREDIT_INDEX = 5  # Cross-desk hedge: HY vs IG
    HEDGE_FX_RATES = 6      # Cross-desk hedge: FX vs rates
    CUT_ALL_RISK = 7        # Emergency: flatten all desks
    CLOSE_DAY = 8           # End trading day for all desks


class FICCFloorEnv(gym.Env):
    """
    Meta-environment for the entire FICC trading floor.

    The agent manages capital allocation and risk limits across all 9 desks.
    Each desk runs its own sub-environment with heuristic or trained policies.

    Action space: MultiDiscrete([9, 9, 10])
      - meta_action (9): Floor-level action
      - desk_idx (9): Which desk to target
      - size (10): How much capital/risk to adjust

    Observation: 63-dim floor state + 9 desk summary stats = 72-dim
      - Per-desk: [pnl, risk_usage, capital_alloc] x 9 = 27
      - Cross-desk correlations (compressed): 9
      - Firm-wide: [total_pnl, var95, var99, capital_util, day, sharpe, drawdown] = 7
      - Per-desk regime indicators: 9
      Total: 52

    Reward: Firm-wide daily P&L adjusted for VaR usage (Sharpe-like)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_days: int = 60,
        max_actions_per_day: int = 20,
        var_limit: float = 2000.0,
        daily_loss_limit: float = -800.0,
        include_mbs: bool = False,  # MBS env has different API
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.include_mbs = include_mbs
        self.rng = np.random.RandomState(seed)

        # Initialize sub-environments (skip MBS by default — different API)
        self.desks: dict[int, gym.Env] = {}
        self.desk_active = np.ones(N_DESKS, dtype=bool)

        if not include_mbs:
            self.desk_active[0] = False  # MBS desk inactive by default

        desk_classes = [
            None,                 # MBS - handled separately
            TreasuriesEnv,
            DistressedCreditEnv,
            RatesEnv,
            IGCreditEnv,
            MunisEnv,
            RepoEnv,
            FXEnv,
            CommoditiesEnv,
        ]

        for i, cls in enumerate(desk_classes):
            if cls is not None and self.desk_active[i]:
                self.desks[i] = cls(seed=seed)

        # Action space
        self.action_space = spaces.MultiDiscrete([9, 9, 10])

        # Observation space
        self._obs_dim = 52
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32)

        self._capital_buckets = np.linspace(0.05, 0.30, 10)

        self.state: Optional[FloorState] = None
        self.day = 0
        self.actions_today = 0
        self._daily_pnls: list[float] = []
        self._desk_obs: dict[int, np.ndarray] = {}

        # Heuristic desk policies (random for now, can be replaced with trained)
        self._desk_policies: dict[int, object] = {}

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(self._obs_dim, dtype=np.float32)

        obs = np.concatenate([
            self.state.desk_pnl / 5000.0,             # 9
            self.state.desk_risk_usage,                # 9
            self.state.desk_capital_alloc,             # 9
            np.diag(DESK_CORRELATION) * 0 + \
                np.array([DESK_CORRELATION[i].mean() for i in range(N_DESKS)]),  # 9 avg corr
            [self.state.total_pnl / 10000.0],          # 1
            [self.state.firm_var_95 / 5000.0],          # 1
            [self.state.firm_var_99 / 5000.0],          # 1
            [self.state.capital_utilization],            # 1
            [self.state.day / self.max_days],            # 1
            [self.state.sharpe_running / 3.0],           # 1
            [self.state.max_drawdown / 5000.0],          # 1
            # Desk regime indicators (1 if active)
            self.desk_active.astype(float),             # 9
        ])
        return obs[:self._obs_dim].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.state = FloorState()
        self.state.var_limit = 2000.0
        self.state.daily_loss_limit = -800.0
        self.state.desk_capital_alloc = np.ones(N_DESKS) / N_DESKS

        # Reset all active desks
        for i, env in self.desks.items():
            obs, _ = env.reset(seed=seed)
            self._desk_obs[i] = obs

        self.day = 0
        self.actions_today = 0
        self._daily_pnls = []

        return self._get_obs(), {"day": 0, "desks_active": int(sum(self.desk_active))}

    def step(self, action):
        assert self.state is not None

        meta_act = FloorAction(action[0])
        desk_idx = min(action[1], N_DESKS - 1)
        size_idx = min(action[2], 9)

        reward = 0.0
        terminated = False
        truncated = False
        self.actions_today += 1

        if meta_act == FloorAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.state.day = self.day
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif meta_act == FloorAction.NOOP:
            pass
        elif meta_act == FloorAction.INCREASE_CAPITAL:
            self._adjust_capital(desk_idx, self._capital_buckets[size_idx])
        elif meta_act == FloorAction.DECREASE_CAPITAL:
            self._adjust_capital(desk_idx, -self._capital_buckets[size_idx])
        elif meta_act == FloorAction.STEP_DESK:
            reward = self._step_desk(desk_idx)
        elif meta_act == FloorAction.HEDGE_RATES_TSY:
            reward = self._cross_hedge_rates_tsy()
        elif meta_act == FloorAction.HEDGE_CREDIT_INDEX:
            reward = self._cross_hedge_credit()
        elif meta_act == FloorAction.HEDGE_FX_RATES:
            reward = self._cross_hedge_fx_rates()
        elif meta_act == FloorAction.CUT_ALL_RISK:
            reward = self._emergency_flatten()

        # Update firm-wide metrics
        self._update_firm_metrics()

        # VaR limit breach penalty
        if self.state.firm_var_95 > self.state.var_limit:
            reward -= 10.0

        # Daily loss limit breach
        daily_pnl = self.state.total_pnl - (self._daily_pnls[-1] if self._daily_pnls else 0)
        if daily_pnl < self.state.daily_loss_limit:
            reward -= 20.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.state.day = self.day
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        info = {
            "day": self.day,
            "total_pnl": self.state.total_pnl,
            "var_95": self.state.firm_var_95,
            "capital_util": self.state.capital_utilization,
            "sharpe": self.state.sharpe_running,
            "drawdown": self.state.max_drawdown,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _adjust_capital(self, desk_idx: int, delta: float):
        """Adjust capital allocation to a desk."""
        alloc = self.state.desk_capital_alloc.copy()
        alloc[desk_idx] = np.clip(alloc[desk_idx] + delta, 0.02, 0.50)
        # Renormalize
        alloc = alloc / alloc.sum()
        self.state.desk_capital_alloc = alloc

    def _step_desk(self, desk_idx: int) -> float:
        """Let a desk take one step with its policy."""
        if desk_idx not in self.desks or not self.desk_active[desk_idx]:
            return 0.0

        env = self.desks[desk_idx]
        # Use random policy (can be replaced with trained policy)
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        self._desk_obs[desk_idx] = obs

        # Scale reward by capital allocation
        scaled_reward = reward * self.state.desk_capital_alloc[desk_idx]
        self.state.desk_pnl[desk_idx] += reward

        # Update risk usage
        if hasattr(info, 'get'):
            gross = info.get('gross_exposure', info.get('gross', info.get('total_dv01', 0)))
            if gross:
                self.state.desk_risk_usage[desk_idx] = min(1.0, gross / 1000)

        # If desk episode ended, reset it
        if term or trunc:
            obs, _ = env.reset()
            self._desk_obs[desk_idx] = obs

        return float(scaled_reward)

    def _cross_hedge_rates_tsy(self) -> float:
        """Offset rates desk duration with treasury desk."""
        # Simplified: flatten 10% of rates DV01 and Tsy DV01
        cost = 0.0
        if 3 in self.desks:
            rates_env = self.desks[3]
            if hasattr(rates_env, 'book') and rates_env.book is not None:
                rates_env.book.swap_dv01 *= 0.9
                cost += float(np.sum(np.abs(rates_env.book.swap_dv01))) * 0.01
        if 1 in self.desks:
            tsy_env = self.desks[1]
            if hasattr(tsy_env, 'book') and tsy_env.book is not None:
                tsy_env.book.dv01_by_tenor *= 0.9
                cost += float(np.sum(np.abs(tsy_env.book.dv01_by_tenor))) * 0.01
        return -cost

    def _cross_hedge_credit(self) -> float:
        """Offset distressed long with IG short."""
        cost = 0.0
        if 2 in self.desks and 4 in self.desks:
            dist_env = self.desks[2]
            ig_env = self.desks[4]
            if hasattr(dist_env, 'book') and dist_env.book is not None:
                net = float(np.sum(dist_env.book.positions))
                if hasattr(ig_env, 'book') and ig_env.book is not None:
                    ig_env.book.index_position -= net * 0.3
                    cost += abs(net) * 0.001
        return -cost

    def _cross_hedge_fx_rates(self) -> float:
        """Hedge FX rate exposure with rates swaps."""
        return 0.0  # Simplified

    def _emergency_flatten(self) -> float:
        """Emergency risk cut across all desks — scale positions to 10%."""
        total_cost = 0.0
        for i, env in self.desks.items():
            if hasattr(env, 'book') and env.book is not None:
                book = env.book
                # Flatten all position arrays by 90%
                for attr in dir(book):
                    val = getattr(book, attr, None)
                    if isinstance(val, np.ndarray) and 'pnl' not in attr and 'realized' not in attr:
                        size_before = float(np.sum(np.abs(val)))
                        setattr(book, attr, val * 0.1)
                        total_cost += size_before * 0.001  # 10bps liquidation cost
                # Flatten scalar position fields
                for attr in ['index_position', 'basis_position', 'trs_notional',
                             'block_pending', 'dispersion']:
                    if hasattr(book, attr):
                        val = getattr(book, attr)
                        if isinstance(val, (int, float)):
                            total_cost += abs(val) * 0.001
                            setattr(book, attr, val * 0.1)
        self.state.desk_risk_usage *= 0.1
        return -total_cost

    def _update_firm_metrics(self):
        """Compute firm-wide risk metrics."""
        # Total P&L
        self.state.total_pnl = float(np.sum(self.state.desk_pnl))

        # Firm VaR (parametric, using correlation matrix)
        desk_vars = np.array(DESK_VOL) * self.state.desk_risk_usage * \
            self.state.desk_capital_alloc
        # Portfolio VaR = sqrt(w' * Sigma * w)
        cov = np.outer(desk_vars, desk_vars) * DESK_CORRELATION
        portfolio_var = np.sqrt(max(0, np.sum(cov)))
        self.state.firm_var_95 = portfolio_var * 1.645
        self.state.firm_var_99 = portfolio_var * 2.326

        # Capital utilization
        self.state.capital_utilization = self.state.firm_var_95 / \
            max(1, self.state.var_limit)

        # Running Sharpe
        if len(self._daily_pnls) > 5:
            returns = np.diff(self._daily_pnls)
            if np.std(returns) > 0:
                self.state.sharpe_running = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Drawdown
        self.state.peak_pnl = max(self.state.peak_pnl, self.state.total_pnl)
        self.state.max_drawdown = max(self.state.max_drawdown,
                                       self.state.peak_pnl - self.state.total_pnl)

    def _end_of_day(self) -> float:
        """Process end of day for all desks."""
        total_daily_pnl = 0.0

        for i, env in self.desks.items():
            # Step each desk's end-of-day
            if hasattr(env, '_end_of_day'):
                desk_pnl = env._end_of_day()
                self.state.desk_pnl[i] += desk_pnl
                total_daily_pnl += desk_pnl * self.state.desk_capital_alloc[i]

        self._daily_pnls.append(self.state.total_pnl)
        self._update_firm_metrics()

        # Sharpe-adjusted reward: P&L / VaR usage
        var_adj = max(0.1, self.state.capital_utilization)
        risk_adjusted = total_daily_pnl / var_adj

        return float(np.clip(risk_adjusted, -200, 200))

    def render(self):
        if self.state is None:
            return
        print(f"\n{'='*60}")
        print(f"FICC TRADING FLOOR - Day {self.day}/{self.max_days}")
        print(f"{'='*60}")
        print(f"Total P&L: ${self.state.total_pnl:,.0f}K | "
              f"VaR95: ${self.state.firm_var_95:,.0f}K | "
              f"Sharpe: {self.state.sharpe_running:.2f}")
        print(f"Capital Util: {self.state.capital_utilization:.0%} | "
              f"Drawdown: ${self.state.max_drawdown:,.0f}K")
        print(f"\n{'Desk':<15} {'P&L ($K)':>10} {'Risk':>8} {'Capital':>8}")
        print("-" * 45)
        for i in range(N_DESKS):
            if self.desk_active[i]:
                print(f"{DESK_NAMES[i]:<15} {self.state.desk_pnl[i]:>10,.0f} "
                      f"{self.state.desk_risk_usage[i]:>7.0%} "
                      f"{self.state.desk_capital_alloc[i]:>7.0%}")
        print(f"{'='*60}\n")

    def get_desk_summary(self) -> dict:
        """Get a summary dict of all desk states."""
        if self.state is None:
            return {}
        return {
            DESK_NAMES[i]: {
                "pnl": float(self.state.desk_pnl[i]),
                "risk": float(self.state.desk_risk_usage[i]),
                "capital": float(self.state.desk_capital_alloc[i]),
                "active": bool(self.desk_active[i]),
            }
            for i in range(N_DESKS)
        }


def make_ficc_floor_env(seed: Optional[int] = None, **kwargs) -> FICCFloorEnv:
    return FICCFloorEnv(seed=seed, **kwargs)
