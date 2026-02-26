"""
Yield Book RL Environment for CMO Structuring.

Gymnasium-compatible reinforcement learning environment that mimics
the Yield Book Structuring Tool. Agents learn to structure CMO/REMIC deals
by selecting tranche types, sizing, and coupons to maximize deal arbitrage.

Supports both Agency CMO and Non-Agency (CMBS) deal modes.

Observation space: Market conditions + current deal state
Action space: Add/modify/remove tranches, set coupons, execute deal
Reward: Deal arbitrage in ticks (32nds) = (CMO Proceeds - Collateral Cost)

From Fuster, Lucca & Vickery (2022) and the Yield Book Structuring Tool.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

from .yield_curve import YieldCurve
from .spec_pool import SpecPool, PoolCharacteristic, project_pool_cashflows, PoolCashFlows, spec_pool_payup
from .cmo_structure import (
    TrancheSpec, PrincipalType, InterestType,
    structure_cmo, CMOCashFlows,
)
from .pricing import price_tranche, price_deal, structuring_profit
from .market_simulator import MarketSimulator, MarketScenario
from .tba import build_tba_price_grid
from .deal_economics import compute_deal_pnl


# ─── Action Encoding ──────────────────────────────────────────────────────

class ActionType(IntEnum):
    """High-level action types for the CMO structuring agent."""
    ADD_SEQ = 0        # Add sequential tranche
    ADD_PAC = 1        # Add PAC tranche
    ADD_TAC = 2        # Add TAC tranche
    ADD_SUPPORT = 3    # Add support tranche
    ADD_Z_BOND = 4     # Add Z-bond
    ADD_IO = 5         # Add IO strip
    ADD_PO = 6         # Add PO strip
    ADD_FLOATER = 7    # Add floater
    ADD_INV_FLOAT = 8  # Add inverse floater
    MODIFY_SIZE = 9    # Modify tranche size
    MODIFY_COUPON = 10 # Modify tranche coupon
    REMOVE_TRANCHE = 11  # Remove a tranche
    EXECUTE_DEAL = 12  # Execute the deal (terminal action)
    NOOP = 13          # No operation
    SELECT_POOL = 14   # Select collateral pool type (uses tranche_idx as pool type)


# Tranche type mapping
_PRINCIPAL_MAP = {
    ActionType.ADD_SEQ: PrincipalType.SEQUENTIAL,
    ActionType.ADD_PAC: PrincipalType.PAC,
    ActionType.ADD_TAC: PrincipalType.TAC,
    ActionType.ADD_SUPPORT: PrincipalType.SUPPORT,
    ActionType.ADD_Z_BOND: PrincipalType.SEQUENTIAL,
    ActionType.ADD_IO: PrincipalType.PASSTHROUGH,
    ActionType.ADD_PO: PrincipalType.PASSTHROUGH,
    ActionType.ADD_FLOATER: PrincipalType.SEQUENTIAL,
    ActionType.ADD_INV_FLOAT: PrincipalType.SEQUENTIAL,
}

_INTEREST_MAP = {
    ActionType.ADD_SEQ: InterestType.FIXED,
    ActionType.ADD_PAC: InterestType.FIXED,
    ActionType.ADD_TAC: InterestType.FIXED,
    ActionType.ADD_SUPPORT: InterestType.FIXED,
    ActionType.ADD_Z_BOND: InterestType.Z_ACCRUAL,
    ActionType.ADD_IO: InterestType.IO_ONLY,
    ActionType.ADD_PO: InterestType.PO_ONLY,
    ActionType.ADD_FLOATER: InterestType.FLOATING,
    ActionType.ADD_INV_FLOAT: InterestType.INVERSE_FLOATING,
}


# ─── Deal State ────────────────────────────────────────────────────────────

@dataclass
class DealState:
    """Current state of the deal being structured."""
    tranches: list[TrancheSpec] = field(default_factory=list)
    collateral_balance: float = 0.0
    collateral_coupon: float = 5.5
    collateral_wac: float = 6.0
    collateral_wam: int = 357
    collateral_wala: int = 3
    deal_mode: str = "AGENCY"  # AGENCY or CMBS

    @property
    def n_tranches(self) -> int:
        return len(self.tranches)

    @property
    def total_tranche_balance(self) -> float:
        return sum(t.original_balance for t in self.tranches if not t.is_io)

    @property
    def unallocated_balance(self) -> float:
        return self.collateral_balance - self.total_tranche_balance

    @property
    def allocation_pct(self) -> float:
        if self.collateral_balance <= 0:
            return 0.0
        return self.total_tranche_balance / self.collateral_balance

    def to_vector(self) -> np.ndarray:
        """Encode deal state as fixed-size vector."""
        # Encode up to 10 tranches
        max_tranches = 10
        tranche_vec = np.zeros(max_tranches * 6)  # 6 features per tranche

        for i, t in enumerate(self.tranches[:max_tranches]):
            offset = i * 6
            tranche_vec[offset] = t.principal_type.value.encode()[0] / 100.0  # type encoding
            tranche_vec[offset + 1] = t.interest_type.value.encode()[0] / 100.0
            tranche_vec[offset + 2] = t.original_balance / max(1, self.collateral_balance)
            tranche_vec[offset + 3] = t.coupon / 10.0
            tranche_vec[offset + 4] = t.priority / 10.0
            tranche_vec[offset + 5] = 1.0  # active flag

        summary = np.array([
            self.n_tranches / 10.0,
            self.allocation_pct,
            self.unallocated_balance / max(1, self.collateral_balance),
            self.collateral_coupon / 10.0,
            self.collateral_wac / 10.0,
            self.collateral_wam / 360.0,
            1.0 if self.deal_mode == "AGENCY" else 0.0,
        ])

        return np.concatenate([summary, tranche_vec])

    @property
    def state_dim(self) -> int:
        return 7 + 10 * 6  # 67


# ─── Yield Book Environment ───────────────────────────────────────────────

class YieldBookEnv(gym.Env):
    """
    Gymnasium environment that mimics the Yield Book Structuring Tool.

    The agent structures a CMO deal by:
    1. Observing market conditions (yield curve, TBA prices, collateral)
    2. Adding tranches (SEQ, PAC, TAC, Support, Z-bond, IO/PO, Floater/Inv)
    3. Setting sizes and coupons for each tranche
    4. Executing the deal when satisfied

    Reward = Deal Arbitrage in ticks (32nds)
           = (Sum of tranche proceeds - Collateral cost) / collateral_face * 32 * 100

    Episode ends when:
    - Agent executes the deal (EXECUTE_DEAL action)
    - Maximum steps exceeded
    - Invalid deal state (e.g. over-allocation)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        max_steps: int = 30,
        max_tranches: int = 10,
        collateral_balance: float = 100_000_000,
        seed: Optional[int] = None,
        deal_mode: str = "AGENCY",
        render_mode: Optional[str] = None,
        market_simulator: Optional[MarketSimulator] = None,
        data_provider: Optional["RealMarketDataProvider"] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.max_tranches = max_tranches
        self.collateral_balance = collateral_balance
        self.deal_mode = deal_mode
        self.render_mode = render_mode

        self.market_sim = market_simulator or MarketSimulator(seed=seed)
        self.data_provider = data_provider
        self._seed = seed

        # Action space: MultiDiscrete
        # [action_type (15), tranche_idx (10), size_bucket (20), coupon_bucket (20)]
        # action_type 14 = SELECT_POOL: uses tranche_idx as pool type
        self.action_space = spaces.MultiDiscrete([15, 10, 20, 20])

        # Observation space
        # Market obs (yield curve 10 + TBA prices 10 + rates 5 + pools 35 + deal_mode 1 + financing 1)
        # = 62 market dims + 6 desk state dims = 68 market dims
        # + 40 ecosystem dims (real market data)
        # + Deal state = 67 dims
        # = 175 total (108 market + 67 deal)
        self._market_dim = 68  # 62 base + 6 desk state (Song & Zhu)
        self._ecosystem_dim = 40  # Real market ecosystem vector
        self._deal_dim = 67
        obs_dim = self._market_dim + self._ecosystem_dim + self._deal_dim

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self.scenario: Optional[MarketScenario] = None
        self.deal: Optional[DealState] = None
        self.collateral_cf: Optional[PoolCashFlows] = None
        self.pool: Optional[SpecPool] = None
        self.step_count = 0
        self._episode_reward = 0.0

        # Size and coupon buckets
        self._size_buckets = np.linspace(0.02, 0.50, 20)  # 2% to 50% of collateral
        # Coupon bucket: offset from collateral coupon (-2% to +1%)
        self._coupon_offsets = np.linspace(-2.0, 1.0, 20)

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from market + ecosystem + deal state."""
        total_dim = self._market_dim + self._ecosystem_dim + self._deal_dim
        if self.scenario is None or self.deal is None:
            return np.zeros(total_dim, dtype=np.float32)

        market_obs = self.scenario.to_observation()

        # Flatten market observation (62 base + 6 desk state = 68)
        market_vec = np.concatenate([
            np.array(market_obs["curve_yields"]) / 10.0,  # 10
            np.array(market_obs["tba_prices"]) / 110.0,   # 10
            np.array(market_obs["rates"]) / 10.0,          # 5
            np.array(market_obs["pools"]),                  # 35
            [market_obs["deal_mode"]],                      # 1
            [market_obs["financing_rate"] / 10.0],          # 1
            np.array(market_obs["desk_state"]),             # 6 (Song & Zhu desk P&L)
        ])

        # Ecosystem vector (40 dims from real market data)
        ecosystem_vec = np.array(market_obs.get("ecosystem", [0.0] * 40))

        deal_vec = self.deal.to_vector()
        obs = np.concatenate([market_vec, ecosystem_vec, deal_vec]).astype(np.float32)

        # Clamp to obs space bounds
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def _get_info(self) -> dict:
        """Return auxiliary info."""
        if self.deal is None:
            return {}
        return {
            "n_tranches": self.deal.n_tranches,
            "allocation_pct": round(self.deal.allocation_pct * 100, 1),
            "unallocated": round(self.deal.unallocated_balance, 2),
            "step": self.step_count,
            "deal_mode": self.deal.deal_mode,
            "regime": self.scenario.regime if self.scenario else "",
        }

    def reset(self, seed=None, options=None):
        """Reset environment with a new market scenario."""
        super().reset(seed=seed)
        if seed is not None:
            self.market_sim = MarketSimulator(seed=seed)

        # Generate market scenario: real data or synthetic
        if self.data_provider is not None and self.data_provider.n_dates > 0:
            rng = np.random.RandomState(seed) if seed is not None else self.market_sim.rng
            random_date = self.data_provider.get_random_historical_date(rng)
            self.scenario = self.data_provider.get_scenario_for_date(
                random_date, collateral_balance=self.collateral_balance, rng=rng
            )
            # Fallback to synthetic if date returned None
            if self.scenario is None:
                regime = options.get("regime") if options else None
                self.scenario = self.market_sim.generate_scenario(regime=regime)
        else:
            regime = options.get("regime") if options else None
            self.scenario = self.market_sim.generate_scenario(regime=regime)

        # Pick first collateral pool (or synthesize one)
        if self.scenario.collateral_pools:
            self.pool = self.scenario.collateral_pools[0]
            # Scale to target balance
            scale = self.collateral_balance / max(1, self.pool.current_balance)
            self.pool.original_balance = self.collateral_balance
            self.pool.current_balance = self.collateral_balance
        else:
            from .spec_pool import AgencyType, CollateralType
            self.pool = SpecPool(
                pool_id="SIM_POOL",
                agency=AgencyType.FNMA,
                collateral_type=CollateralType.FN,
                coupon=round(self.scenario.mortgage_rate - 0.5, 1),
                wac=round(self.scenario.mortgage_rate, 3),
                wam=357,
                wala=3,
                original_balance=self.collateral_balance,
                current_balance=self.collateral_balance,
            )

        # Project collateral cash flows
        # Both wac and mortgage_rate are in percentage points (e.g. 7.5, 7.95)
        from .prepayment import estimate_psa_speed
        psa = estimate_psa_speed(
            self.pool.wac,
            self.scenario.mortgage_rate,
            self.pool.wala,
        )
        self.collateral_cf = project_pool_cashflows(self.pool, psa_speed=psa)

        # Initialize deal state
        deal_mode = self.scenario.deal_mode if self.deal_mode == "AGENCY" else self.deal_mode
        self.deal = DealState(
            collateral_balance=self.collateral_balance,
            collateral_coupon=self.pool.coupon,
            collateral_wac=self.pool.wac,
            collateral_wam=self.pool.wam,
            collateral_wala=self.pool.wala,
            deal_mode=deal_mode,
        )

        self.step_count = 0
        self._episode_reward = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Take a structuring action."""
        assert self.deal is not None and self.scenario is not None

        action_type = ActionType(action[0])
        tranche_idx = action[1]
        size_bucket = action[2]
        coupon_bucket = action[3]

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Decode size and coupon (coupon is offset from collateral coupon)
        size_pct = float(self._size_buckets[size_bucket])
        size = size_pct * self.collateral_balance
        base_coupon = self.deal.collateral_coupon if self.deal else 5.5
        coupon = base_coupon + float(self._coupon_offsets[coupon_bucket])
        coupon = max(0.5, round(coupon * 2) / 2)  # Round to nearest 0.5, floor at 0.5

        # Process action
        if action_type == ActionType.EXECUTE_DEAL:
            reward, terminated = self._execute_deal()
        elif action_type == ActionType.NOOP:
            reward = -0.1  # Small penalty for wasting time
        elif action_type == ActionType.SELECT_POOL:
            reward = self._select_pool(tranche_idx)  # tranche_idx encodes pool type
        elif action_type == ActionType.REMOVE_TRANCHE:
            reward = self._remove_tranche(tranche_idx)
        elif action_type == ActionType.MODIFY_SIZE:
            reward = self._modify_tranche_size(tranche_idx, size)
        elif action_type == ActionType.MODIFY_COUPON:
            reward = self._modify_tranche_coupon(tranche_idx, coupon)
        elif action_type.value <= ActionType.ADD_INV_FLOAT.value:
            reward = self._add_tranche(action_type, size, coupon)
        else:
            reward = -0.5  # Invalid action

        # Step penalty to encourage efficiency
        reward -= 0.05

        # Check truncation
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            # Auto-execute if we have tranches
            if self.deal.n_tranches > 0 and self.deal.allocation_pct > 0.5:
                exec_reward, _ = self._execute_deal()
                reward += exec_reward
            else:
                reward -= 5.0  # Penalty for not completing a deal

        self._episode_reward += reward
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _add_tranche(self, action_type: ActionType, size: float, coupon: float) -> float:
        """Add a tranche to the deal."""
        if self.deal is None:
            return -1.0

        if self.deal.n_tranches >= self.max_tranches:
            return -2.0  # Too many tranches

        # Check if size would over-allocate (except IO which is notional)
        is_io = (action_type == ActionType.ADD_IO)
        if not is_io and size > self.deal.unallocated_balance * 1.01:
            # Clamp to available
            size = max(0, self.deal.unallocated_balance)
            if size < self.collateral_balance * 0.01:
                return -1.0  # Too small

        principal_type = _PRINCIPAL_MAP[action_type]
        interest_type = _INTEREST_MAP[action_type]
        priority = self.deal.n_tranches

        name = f"T{priority + 1}_{interest_type.value}"

        # Default coupon to collateral coupon for most tranche types
        if is_io:
            # IO coupon = excess coupon (collateral - tranche weighted avg)
            io_coupon = max(0.5, self.deal.collateral_coupon - coupon + 1.0)
            coupon = io_coupon

        spec = TrancheSpec(
            name=name,
            principal_type=principal_type,
            interest_type=interest_type,
            original_balance=0.0 if is_io else size,
            coupon=coupon,
            notional_balance=size if is_io else 0.0,
            priority=priority,
        )

        # Set PAC bands for PAC tranches
        if action_type == ActionType.ADD_PAC:
            spec.pac_lower_band = 100.0
            spec.pac_upper_band = 300.0

        # Set floater params
        if action_type == ActionType.ADD_FLOATER:
            spec.index_spread = 50.0
            spec.rate_cap = 10.0
            spec.rate_floor = 0.0

        if action_type == ActionType.ADD_INV_FLOAT:
            spec.inverse_constant = 20.0
            spec.inverse_multiplier = 3.0
            spec.rate_cap = 20.0
            spec.rate_floor = 0.0

        self.deal.tranches.append(spec)

        # Shaping reward: encourage reasonable allocation
        # IO strips don't change allocation — only reward the FIRST IO
        if is_io:
            n_io = sum(1 for t in self.deal.tranches
                       if t.interest_type == InterestType.IO_ONLY)
            return 0.5 if n_io == 1 else -0.3  # Penalize redundant IO strips

        alloc = self.deal.allocation_pct
        if 0.95 <= alloc <= 1.01:
            return 1.0  # Good: fully allocated
        elif 0.5 <= alloc < 0.95:
            return 0.3  # Okay: partially allocated
        else:
            return 0.1  # Just started

    def _remove_tranche(self, idx: int) -> float:
        """Remove a tranche by index."""
        if self.deal is None or idx >= self.deal.n_tranches:
            return -0.5
        self.deal.tranches.pop(idx)
        # Renumber priorities
        for i, t in enumerate(self.deal.tranches):
            t.priority = i
        return -0.2  # Small cost for removing

    def _modify_tranche_size(self, idx: int, new_size: float) -> float:
        """Modify the size of a tranche."""
        if self.deal is None or idx >= self.deal.n_tranches:
            return -0.5

        t = self.deal.tranches[idx]
        if t.is_io:
            t.notional_balance = new_size
        else:
            old_size = t.original_balance
            # Check if new size fits
            new_total = self.deal.total_tranche_balance - old_size + new_size
            if new_total > self.collateral_balance * 1.01:
                new_size = self.collateral_balance - (self.deal.total_tranche_balance - old_size)
                new_size = max(0, new_size)
            t.original_balance = new_size
        return 0.1

    def _modify_tranche_coupon(self, idx: int, new_coupon: float) -> float:
        """Modify the coupon of a tranche."""
        if self.deal is None or idx >= self.deal.n_tranches:
            return -0.5
        self.deal.tranches[idx].coupon = new_coupon
        return 0.0

    # Pool type mapping: tranche_idx value -> PoolCharacteristic
    _POOL_TYPE_MAP = {
        0: PoolCharacteristic.TBA,
        1: PoolCharacteristic.LOW_LOAN_BALANCE,
        2: PoolCharacteristic.HIGH_LTV,
        3: PoolCharacteristic.NEW_YORK,
        4: PoolCharacteristic.LOW_FICO,
        5: PoolCharacteristic.INVESTOR,
        6: PoolCharacteristic.GEO_CONCENTRATED,
    }

    def _select_pool(self, pool_type_idx: int) -> float:
        """Select collateral pool type. Affects payup and prepayment speed.

        Only effective once per episode (first call sets the pool; subsequent calls
        are penalized).
        """
        if self.scenario is None or self.pool is None:
            return -1.0

        # Only allow pool selection if we haven't already selected a non-TBA pool
        if self.scenario.selected_pool_type != 0:
            return -0.5  # Already selected; penalize re-selection

        pool_type_idx = min(pool_type_idx, 6)  # clamp to valid range
        characteristic = self._POOL_TYPE_MAP.get(pool_type_idx, PoolCharacteristic.TBA)

        # Apply characteristic to the pool
        self.pool.characteristic = characteristic

        # Set characteristic-appropriate pool attributes
        if characteristic == PoolCharacteristic.LOW_LOAN_BALANCE:
            self.pool.avg_loan_size = max(50000, self.pool.avg_loan_size * 0.3)
        elif characteristic == PoolCharacteristic.HIGH_LTV:
            self.pool.avg_ltv = max(91.0, min(98.0, self.pool.avg_ltv * 1.2))
        elif characteristic == PoolCharacteristic.NEW_YORK:
            self.pool.geography = "NY"
        elif characteristic == PoolCharacteristic.LOW_FICO:
            self.pool.avg_fico = max(620, min(680, self.pool.avg_fico * 0.9))
        elif characteristic == PoolCharacteristic.INVESTOR:
            pass  # Investor properties already handled by payup model

        # Calculate payup
        payup = spec_pool_payup(self.pool)
        self.scenario.selected_pool_type = pool_type_idx
        self.scenario.selected_pool_payup = payup

        # Recalculate collateral cost: TBA cost + payup
        # Payup increases collateral cost but also provides prepay protection
        # The agent earns this payup as income if it selected well

        # Reward: small positive for selecting spec pool (encourages exploration)
        # Real P&L impact comes at EXECUTE_DEAL
        if characteristic == PoolCharacteristic.TBA:
            return 0.0  # No benefit to selecting TBA explicitly
        return 0.3  # Small reward for making a spec pool selection

    def _execute_deal(self) -> tuple[float, bool]:
        """Execute the deal and compute reward via deal_economics.

        Delegates ALL P&L computation to compute_deal_pnl() which enforces
        cash flow conservation, investor demand model, and IO cap rules.
        No ad-hoc bonuses — the economics determine everything.
        """
        if self.deal is None or self.collateral_cf is None or self.scenario is None:
            return -10.0, True

        if self.deal.n_tranches == 0:
            return -10.0, True

        # ─── Normalize allocation ────────────────────────────────────────
        # Fill gap into a fixed-rate tranche (SEQ/PAC/TAC/SUPPORT),
        # NOT into exotics (inverse, floater, Z) which would distort demand.
        # If no fixed-rate tranche exists, distribute gap PROPORTIONALLY.
        gap = self.deal.unallocated_balance
        if gap > self.collateral_balance * 0.001:
            filled = False
            for t in reversed(self.deal.tranches):
                if not t.is_io and t.interest_type == InterestType.FIXED:
                    t.original_balance += gap
                    filled = True
                    break
            if not filled:
                # Proportional fill across all non-IO tranches
                non_io = [t for t in self.deal.tranches if not t.is_io]
                total_bal = sum(t.original_balance for t in non_io)
                if total_bal > 0:
                    for t in non_io:
                        t.original_balance += gap * (t.original_balance / total_bal)
                elif non_io:
                    non_io[0].original_balance += gap

        if self.deal.total_tranche_balance > self.collateral_balance * 1.001:
            excess = self.deal.total_tranche_balance - self.collateral_balance
            for t in reversed(self.deal.tranches):
                if not t.is_io:
                    reduce = min(t.original_balance, excess)
                    t.original_balance -= reduce
                    excess -= reduce
                    if excess <= 0:
                        break

        # ─── Structure the deal ──────────────────────────────────────────
        try:
            cmo_cf = structure_cmo(
                deal_id="RL_DEAL",
                collateral_flows=self.collateral_cf,
                tranche_specs=self.deal.tranches,
                collateral_coupon=self.deal.collateral_coupon,
            )
        except Exception:
            return -10.0, True

        # ─── Price tranches ──────────────────────────────────────────────
        # We need pricing_results for WAL-based duration estimates.
        # Pass empty spreads for initial pricing — deal_economics will
        # compute its own market-clearing spreads.
        curve = self.scenario.curve
        try:
            pricing_results = price_deal(cmo_cf, curve, {})
        except Exception:
            return -5.0, True

        # ─── Compute P&L via deal_economics ──────────────────────────────
        tba_price = 100.0
        if self.scenario.tba_grid:
            tba_price = self.scenario.tba_grid.get_price(self.deal.collateral_coupon)

        collateral_wal = self.collateral_cf.wal if self.collateral_cf else 5.0

        pnl = compute_deal_pnl(
            tranches=self.deal.tranches,
            pricing_results=pricing_results,
            collateral_wac=self.deal.collateral_wac,
            collateral_balance=self.collateral_balance,
            collateral_wal=collateral_wal,
            dollar_roll_specialness=self.scenario.dollar_roll_specialness,
            tba_price=tba_price,
            selected_pool_payup=self.scenario.selected_pool_payup,
        )

        return pnl.reward, True

    def render(self):
        """Render the current deal state."""
        if self.deal is None:
            return

        lines = [
            f"═══ Yield Book RL Environment ═══",
            f"Mode: {self.deal.deal_mode} | Step: {self.step_count}/{self.max_steps}",
            f"Regime: {self.scenario.regime if self.scenario else 'N/A'}",
            f"Collateral: ${self.collateral_balance:,.0f} @ {self.deal.collateral_coupon:.1f}%",
            f"Allocation: {self.deal.allocation_pct*100:.1f}% | Unallocated: ${self.deal.unallocated_balance:,.0f}",
            f"",
            f"{'Name':<12} {'Type':<6} {'Int':<5} {'Balance':>14} {'Coupon':>8} {'Pri':>4}",
            f"{'─'*12} {'─'*6} {'─'*5} {'─'*14} {'─'*8} {'─'*4}",
        ]
        for t in self.deal.tranches:
            bal = t.notional_balance if t.is_io else t.original_balance
            lines.append(
                f"{t.name:<12} {t.principal_type.value:<6} {t.interest_type.value:<5} "
                f"${bal:>12,.0f} {t.coupon:>7.2f}% {t.priority:>4d}"
            )
        lines.append(f"{'═'*55}")

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


# ─── CMBS Extension ───────────────────────────────────────────────────────

class CMBSYieldBookEnv(YieldBookEnv):
    """
    Extended environment for CMBS (Commercial MBS) structuring.

    Adds CMBS-specific features from the 666 Fifth Avenue case study:
    - Credit subordination (senior/mezz/B-piece)
    - Special servicing triggers (DSCR, LTV, occupancy)
    - Workout scenarios for distressed loans
    - B-piece buyer dynamics

    The waterfall follows CMBS conventions:
    1. Senior tranches (AAA) get paid first
    2. Mezzanine tranches (AA to BBB) absorb initial losses
    3. B-piece (BB and below) absorbs first-loss
    """

    def __init__(
        self,
        max_steps: int = 40,
        max_tranches: int = 12,
        collateral_balance: float = 500_000_000,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        n_loans: int = 50,
        default_rate: float = 0.03,
    ):
        super().__init__(
            max_steps=max_steps,
            max_tranches=max_tranches,
            collateral_balance=collateral_balance,
            seed=seed,
            deal_mode="CMBS",
            render_mode=render_mode,
        )
        self.n_loans = n_loans
        self.base_default_rate = default_rate

        # Additional CMBS observation dimensions
        # DSCR, LTV, occupancy, property_type_mix, vintage
        self._cmbs_dim = 10
        obs_dim = self._market_dim + self._ecosystem_dim + self._deal_dim + self._cmbs_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # CMBS loan pool state
        self.loan_pool_dscr: float = 1.0
        self.loan_pool_ltv: float = 0.0
        self.loan_pool_occupancy: float = 0.0
        self.property_type_mix: np.ndarray = np.zeros(5)  # office, retail, hotel, multi, industrial

    def reset(self, seed=None, options=None):
        # Generate CMBS loan pool characteristics BEFORE super().reset()
        # because super().reset() calls self._get_obs() which needs these set
        rng = np.random.RandomState(seed)
        self.loan_pool_dscr = rng.uniform(1.1, 2.5)
        self.loan_pool_ltv = rng.uniform(55, 80)
        self.loan_pool_occupancy = rng.uniform(70, 98)
        self.property_type_mix = rng.dirichlet([3, 2, 1, 2, 1])

        obs, info = super().reset(seed=seed, options=options)
        # obs already includes CMBS dims via self._get_obs()

        info["cmbs"] = {
            "dscr": round(self.loan_pool_dscr, 2),
            "ltv": round(self.loan_pool_ltv, 1),
            "occupancy": round(self.loan_pool_occupancy, 1),
            "n_loans": self.n_loans,
        }
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        cmbs_vec = np.array([
            self.loan_pool_dscr / 3.0,
            self.loan_pool_ltv / 100.0,
            self.loan_pool_occupancy / 100.0,
            self.base_default_rate * 10,
            self.n_loans / 100.0,
            *self.property_type_mix,
        ], dtype=np.float32)
        return np.concatenate([base_obs, cmbs_vec])

    def _execute_deal(self) -> tuple[float, bool]:
        """CMBS deal execution with credit subordination analysis."""
        reward, terminated = super()._execute_deal()

        if self.deal is None:
            return reward, terminated

        # CMBS-specific reward adjustments

        # Check credit subordination levels
        sorted_tranches = sorted(self.deal.tranches, key=lambda t: t.priority)
        total_balance = self.deal.total_tranche_balance

        if total_balance > 0:
            senior_pct = 0.0
            mezz_pct = 0.0
            sub_pct = 0.0

            for t in sorted_tranches:
                if t.is_io:
                    continue
                pct = t.original_balance / total_balance
                if t.priority <= 1:
                    senior_pct += pct
                elif t.priority <= 4:
                    mezz_pct += pct
                else:
                    sub_pct += pct

            # Reward for proper CMBS subordination
            # Senior should be 70-80%, mezz 10-20%, sub 5-10%
            if 0.65 <= senior_pct <= 0.85:
                reward += 1.0
            if 0.08 <= mezz_pct <= 0.25:
                reward += 0.5
            if sub_pct >= 0.03:
                reward += 0.5  # B-piece must exist

            # Stress test: would the structure survive defaults?
            expected_loss = self.base_default_rate * (1 - 0.4)  # 40% recovery
            if sub_pct >= expected_loss:
                reward += 2.0  # Structure can absorb expected losses
            else:
                reward -= 3.0  # Under-subordinated

        return reward, terminated


# ─── Vectorized Environment for Training ──────────────────────────────────

def make_yield_book_env(
    mode: str = "AGENCY",
    collateral_balance: float = 100_000_000,
    seed: Optional[int] = None,
    max_steps: int = 30,
    render_mode: Optional[str] = None,
    data_provider=None,
) -> gym.Env:
    """Factory function to create Yield Book environments.

    Args:
        data_provider: Optional RealMarketDataProvider for historical scenarios.
    """
    if mode == "CMBS":
        return CMBSYieldBookEnv(
            max_steps=max_steps,
            collateral_balance=collateral_balance,
            seed=seed,
            render_mode=render_mode,
        )
    return YieldBookEnv(
        max_steps=max_steps,
        collateral_balance=collateral_balance,
        seed=seed,
        deal_mode=mode,
        render_mode=render_mode,
        data_provider=data_provider,
    )


def make_vec_env(
    n_envs: int = 8,
    mode: str = "AGENCY",
    seed: int = 42,
    **kwargs,
) -> list[gym.Env]:
    """Create multiple environments for parallel training."""
    return [
        make_yield_book_env(mode=mode, seed=seed + i, **kwargs)
        for i in range(n_envs)
    ]
