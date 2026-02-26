"""
Unified Investment Bank RL Environment.

Meta-environment combining all businesses into a full-service investment bank:
  - FICC Floor (9 desks): MBS, Treasuries, Distressed, Rates, IG Credit,
    Munis, Repo, FX, Commodities
  - Equities (3 desks): Cash, Derivatives, Prime Brokerage
  - Investment Banking Division (3 desks): ECM/IPO, DCM/LevFin, M&A Advisory

The meta-agent is the CEO/CRO managing:
  - Capital allocation across divisions
  - Firm-wide risk limits (VaR, leverage, concentration)
  - Cross-divisional synergies (IB pipeline -> trading flow)
  - ROE optimization
  - Regulatory capital (Basel III/IV)

References:
  - Duffie (2010) "The Failure Mechanics of Dealer Banks" JEP
  - Adrian & Shin (2010) "Liquidity and Leverage" JFI
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

from cmo_agent.ficc_floor_env import FICCFloorEnv
from cmo_agent.equities_env import EquitiesEnv
from cmo_agent.eq_derivatives_env import EqDerivativesEnv
from cmo_agent.prime_brokerage_env import PrimeBrokerageEnv
from cmo_agent.ecm_env import ECMEnv
from cmo_agent.dcm_env import DCMEnv
from cmo_agent.ma_advisory_env import MAAdvisoryEnv


N_DIVISIONS = 3  # FICC, Equities, IBD
N_TOTAL_DESKS = 15  # 9 FICC + 3 Equities + 3 IBD
DIV_NAMES = ["FICC", "Equities", "IBD"]

DESK_NAMES = [
    # FICC (0-8)
    "MBS/CMO", "Treasuries", "Distressed", "Rates", "IG Credit",
    "Munis", "Repo", "FX", "Commodities",
    # Equities (9-11)
    "Eq Cash", "Eq Deriv", "Prime Brok",
    # IBD (12-14)
    "ECM/IPO", "DCM/LevFin", "M&A Advisory",
]


@dataclass
class IBState:
    """Full investment bank state."""
    # Per-desk P&L ($M)
    desk_pnl: np.ndarray = field(default_factory=lambda: np.zeros(N_TOTAL_DESKS))
    # Division P&L ($M)
    div_pnl: np.ndarray = field(default_factory=lambda: np.zeros(N_DIVISIONS))
    # Capital allocation by division (fraction)
    div_capital: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.30, 0.25]))
    # Risk usage by division (0-1)
    div_risk: np.ndarray = field(default_factory=lambda: np.zeros(N_DIVISIONS))
    # Firm-wide
    total_pnl: float = 0.0
    total_revenue: float = 0.0
    total_rwa: float = 0.0  # Risk-weighted assets ($B)
    leverage_ratio: float = 10.0
    cet1_ratio: float = 0.12  # Common Equity Tier 1 (>=4.5% required)
    roe: float = 0.0  # Return on equity (annualized %)
    # Running stats
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    day: int = 0


class IBAction(IntEnum):
    NOOP = 0
    ALLOC_FICC = 1        # Increase FICC capital
    ALLOC_EQUITIES = 2    # Increase Equities capital
    ALLOC_IBD = 3         # Increase IBD capital
    CUT_FICC = 4          # Cut FICC capital
    CUT_EQUITIES = 5
    CUT_IBD = 6
    STEP_DIVISION = 7     # Step a division forward
    CROSS_SELL = 8        # Generate cross-divisional synergy
    EMERGENCY_DELEVERAGE = 9
    CLOSE_DAY = 10


class InvestmentBankEnv(gym.Env):
    """
    Full investment bank meta-environment.

    Action space: MultiDiscrete([11, 3, 10])
      - action_type (11): NOOP through CLOSE_DAY
      - division (3): FICC, Equities, IBD
      - magnitude (10): size of action

    Observation: 35-dim
      - Per-desk P&L: 15
      - Division metrics: 9 (3x [pnl, capital, risk])
      - Firm-wide: 8 (total_pnl, rwa, leverage, cet1, roe, sharpe, dd, day)
      - Cross-div synergy indicators: 3

    Reward: Risk-adjusted ROE
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_days=60, max_actions_per_day=15, seed=None):
        super().__init__()
        self.max_days = max_days
        self.max_actions_per_day = max_actions_per_day
        self.rng = np.random.RandomState(seed)
        self._seed = seed

        # Sub-environments
        self.ficc = FICCFloorEnv(seed=seed, max_days=max_days)
        self.eq_cash = EquitiesEnv(seed=seed)
        self.eq_deriv = EqDerivativesEnv(seed=seed)
        self.prime_brok = PrimeBrokerageEnv(seed=seed)
        self.ecm = ECMEnv(seed=seed)
        self.dcm = DCMEnv(seed=seed)
        self.ma = MAAdvisoryEnv(seed=seed)

        self._eq_desks = [self.eq_cash, self.eq_deriv, self.prime_brok]
        self._ibd_desks = [self.ecm, self.dcm, self.ma]

        self.action_space = spaces.MultiDiscrete([11, 3, 10])
        self._obs_dim = 35
        self.observation_space = spaces.Box(-10, 10, (self._obs_dim,), np.float32)

        self.state: Optional[IBState] = None
        self.day = 0
        self.actions_today = 0
        self._daily_pnls: list[float] = []

    def _get_obs(self):
        if self.state is None:
            return np.zeros(self._obs_dim, dtype=np.float32)
        obs = np.concatenate([
            self.state.desk_pnl / 1000.0,          # 15
            self.state.div_pnl / 5000.0,            # 3
            self.state.div_capital,                  # 3
            self.state.div_risk,                     # 3
            [self.state.total_pnl / 10000.0],        # 1
            [self.state.total_rwa / 100.0],          # 1
            [self.state.leverage_ratio / 30.0],      # 1
            [self.state.cet1_ratio / 0.20],          # 1
            [self.state.roe / 20.0],                 # 1
            [self.state.sharpe / 3.0],               # 1
            [self.state.max_drawdown / 10000.0],     # 1
            [self.state.day / self.max_days],         # 1
            # Cross-div synergy indicators
            [0.5, 0.5, 0.5],                         # 3 (placeholder)
        ])[:self._obs_dim].astype(np.float32)
        return np.clip(obs, -10, 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.state = IBState()
        self.ficc.reset(seed=seed)
        for d in self._eq_desks:
            d.reset(seed=seed)
        for d in self._ibd_desks:
            d.reset(seed=seed)

        self.day = 0
        self.actions_today = 0
        self._daily_pnls = []

        return self._get_obs(), {"day": 0}

    def step(self, action):
        assert self.state is not None
        act = IBAction(action[0])
        div = min(action[1], N_DIVISIONS - 1)
        mag = (action[2] + 1) / 10.0  # 0.1 to 1.0

        reward = 0.0
        terminated = False
        self.actions_today += 1

        if act == IBAction.CLOSE_DAY:
            reward = self._end_of_day()
            self.day += 1
            self.state.day = self.day
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True
        elif act == IBAction.NOOP:
            pass
        elif act in (IBAction.ALLOC_FICC, IBAction.ALLOC_EQUITIES, IBAction.ALLOC_IBD):
            idx = act - IBAction.ALLOC_FICC
            self.state.div_capital[idx] = min(0.6, self.state.div_capital[idx] + mag * 0.05)
            self.state.div_capital /= self.state.div_capital.sum()
        elif act in (IBAction.CUT_FICC, IBAction.CUT_EQUITIES, IBAction.CUT_IBD):
            idx = act - IBAction.CUT_FICC
            self.state.div_capital[idx] = max(0.1, self.state.div_capital[idx] - mag * 0.05)
            self.state.div_capital /= self.state.div_capital.sum()
        elif act == IBAction.STEP_DIVISION:
            reward = self._step_division(div)
        elif act == IBAction.CROSS_SELL:
            reward = self._cross_sell(div, mag)
        elif act == IBAction.EMERGENCY_DELEVERAGE:
            reward = self._emergency_deleverage()

        self._update_firm_metrics()

        # Regulatory penalties
        if self.state.cet1_ratio < 0.06:
            reward -= 20.0  # Close to regulatory minimum
        if self.state.leverage_ratio > 25:
            reward -= 10.0

        if self.actions_today >= self.max_actions_per_day:
            reward += self._end_of_day()
            self.day += 1
            self.state.day = self.day
            self.actions_today = 0
            if self.day >= self.max_days:
                terminated = True

        return self._get_obs(), reward, terminated, False, {
            "day": self.day, "total_pnl": self.state.total_pnl,
            "roe": self.state.roe, "cet1": self.state.cet1_ratio,
            "sharpe": self.state.sharpe}

    def _step_division(self, div: int) -> float:
        """Step one division forward."""
        if div == 0:  # FICC
            action = self.ficc.action_space.sample()
            _, r, term, _, info = self.ficc.step(action)
            if term:
                self.ficc.reset()
            self.state.div_pnl[0] += r
            return float(r) * self.state.div_capital[0]
        elif div == 1:  # Equities
            total_r = 0
            for i, desk in enumerate(self._eq_desks):
                action = desk.action_space.sample()
                _, r, term, _, info = desk.step(action)
                if term:
                    desk.reset()
                self.state.desk_pnl[9 + i] += r
                total_r += r
            self.state.div_pnl[1] += total_r
            return float(total_r) * self.state.div_capital[1]
        else:  # IBD
            total_r = 0
            for i, desk in enumerate(self._ibd_desks):
                action = desk.action_space.sample()
                _, r, term, _, info = desk.step(action)
                if term:
                    desk.reset()
                self.state.desk_pnl[12 + i] += r
                total_r += r
            self.state.div_pnl[2] += total_r
            return float(total_r) * self.state.div_capital[2]

    def _cross_sell(self, div: int, mag: float) -> float:
        """Generate cross-divisional synergy."""
        # IBD deal -> trading flow (ECM IPO -> equities trading commissions)
        synergy = mag * self.rng.uniform(0, 5)
        self.state.total_revenue += synergy
        return synergy

    def _emergency_deleverage(self) -> float:
        """Cut firm-wide risk aggressively."""
        cost = 0.0
        # Flatten FICC
        action = np.array([8, 0, 0])  # CUT_ALL_RISK
        _, r, _, _, _ = self.ficc.step(action)
        cost += abs(r)
        # Flatten equity desks
        for desk in self._eq_desks:
            if hasattr(desk, 'book') and desk.book is not None:
                desk.book.realized_pnl -= 10  # Rough liquidation cost
                cost += 10
        self.state.leverage_ratio *= 0.7
        return -cost

    def _update_firm_metrics(self):
        self.state.total_pnl = float(np.sum(self.state.div_pnl))
        self.state.total_revenue = max(self.state.total_revenue,
                                        self.state.total_pnl)

        # RWA based on gross exposure, not P&L (Basel standardized approach)
        # FICC: notional from sub-desk gross positions, 50-100% risk weight
        # Equities: 100% risk weight on gross
        # IBD: low RWA (advisory is off-balance-sheet, only underwriting commitments)
        ficc_gross = self._get_division_gross(0)
        eq_gross = self._get_division_gross(1)
        ibd_gross = self._get_division_gross(2)
        self.state.total_rwa = max(1.0,
            ficc_gross * 0.5 +    # FICC: ~50% average risk weight
            eq_gross * 1.0 +      # Equities: 100% risk weight
            ibd_gross * 0.2)      # IBD: mostly off-balance-sheet

        # Equity absorbs cumulative P&L ($B base + retained earnings)
        equity = 10.0 + self.state.total_pnl / 1000.0  # P&L in $M -> $B
        equity = max(1.0, equity)  # Floor at $1B (avoid div by zero)

        # Leverage = total assets / equity
        total_assets = max(1, ficc_gross + eq_gross + ibd_gross + 100)
        self.state.leverage_ratio = total_assets / equity

        # CET1 = equity / RWA
        self.state.cet1_ratio = equity / max(1, self.state.total_rwa)

        # ROE (annualized)
        if len(self._daily_pnls) > 20:
            recent_pnl = self.state.total_pnl - self._daily_pnls[-20]
            self.state.roe = recent_pnl / equity / 20 * 252  # Annualized

        # Sharpe
        if len(self._daily_pnls) > 5:
            returns = np.diff(self._daily_pnls[-60:])
            if len(returns) > 0 and np.std(returns) > 0:
                self.state.sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Drawdown
        self.state.peak_pnl = max(self.state.peak_pnl, self.state.total_pnl)
        self.state.max_drawdown = max(self.state.max_drawdown,
                                       self.state.peak_pnl - self.state.total_pnl)

    def _get_division_gross(self, div: int) -> float:
        """Estimate gross exposure ($B) for a division from sub-envs."""
        if div == 0:  # FICC
            gross = 0.0
            for i, env in self.ficc.desks.items():
                if hasattr(env, 'book') and env.book is not None:
                    if hasattr(env.book, 'gross_exposure'):
                        gross += env.book.gross_exposure / 1000.0
                    elif hasattr(env.book, 'total_dv01'):
                        gross += abs(env.book.total_dv01) * 0.01
                    else:
                        gross += 1.0  # Default $1B per active desk
            return max(5.0, gross)  # FICC floor minimum $5B
        elif div == 1:  # Equities
            gross = 0.0
            for desk in self._eq_desks:
                if hasattr(desk, 'book') and desk.book is not None:
                    if hasattr(desk.book, 'gross_exposure'):
                        gross += desk.book.gross_exposure / 1000.0
                    elif hasattr(desk.book, 'total_exposure'):
                        gross += desk.book.total_exposure
                    elif hasattr(desk.book, 'total_vega'):
                        gross += desk.book.total_vega / 100.0
                    else:
                        gross += 1.0
            return max(3.0, gross)
        else:  # IBD
            # IBD is mostly advisory (off-balance-sheet)
            # Only bridge loans and underwriting commitments use balance sheet
            gross = 0.0
            for desk in self._ibd_desks:
                if hasattr(desk, 'book') and desk.book is not None:
                    if hasattr(desk.book, 'bridge_outstanding'):
                        gross += desk.book.bridge_outstanding / 1000.0
                    elif hasattr(desk.book, 'warehouse_total'):
                        gross += desk.book.warehouse_total / 1000.0
                    else:
                        gross += 0.5
            return max(1.0, gross)

    def _end_of_day(self) -> float:
        self._daily_pnls.append(self.state.total_pnl)

        # Step all divisions forward (end-of-day processing)
        total = 0.0
        for div in range(N_DIVISIONS):
            pnl = self._step_division(div)
            total += pnl

        self._update_firm_metrics()

        # ROE-adjusted reward
        var_adj = max(0.1, self.state.leverage_ratio / 15.0)
        return float(np.clip(total / var_adj, -100, 100))

    def render(self):
        if self.state is None:
            return
        print(f"\n{'='*70}")
        print(f"INVESTMENT BANK — Day {self.day}/{self.max_days}")
        print(f"{'='*70}")
        print(f"Total P&L: ${self.state.total_pnl:,.0f}M | "
              f"ROE: {self.state.roe:.1f}% | Sharpe: {self.state.sharpe:.2f}")
        print(f"CET1: {self.state.cet1_ratio:.1%} | "
              f"Leverage: {self.state.leverage_ratio:.1f}x | "
              f"Drawdown: ${self.state.max_drawdown:,.0f}M")
        print()
        for i, name in enumerate(DIV_NAMES):
            print(f"  {name:12s}: P&L=${self.state.div_pnl[i]:>10,.0f}M  "
                  f"Capital={self.state.div_capital[i]:>5.0%}  "
                  f"Risk={self.state.div_risk[i]:>5.0%}")
        print()
        print(f"  {'Desk':<16} {'P&L ($M)':>10}")
        print(f"  {'-'*28}")
        for i, name in enumerate(DESK_NAMES):
            if abs(self.state.desk_pnl[i]) > 0.1:
                print(f"  {name:<16} {self.state.desk_pnl[i]:>10,.1f}")
        print(f"{'='*70}\n")


def make_investment_bank_env(seed=None, **kwargs):
    return InvestmentBankEnv(seed=seed, **kwargs)
