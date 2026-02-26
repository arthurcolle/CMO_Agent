"""
Multi-Client Yield Book RL Environment for CMO Structuring.

Extends YieldBookEnv to simulate a realistic dealer desk where 2-5 investor
clients with different needs simultaneously want to buy tranches. The agent
structures deals that satisfy multiple competing demands while maximizing
desk P&L.

Key design:
- Action space unchanged (agent structures the deal normally)
- Environment auto-matches tranches to clients using greedy algorithm
- Multi-objective reward: dealer profit + client satisfaction + fill rate
- Curriculum learning via difficulty levels
"""
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional

from .yield_book_env import YieldBookEnv
from .cmo_structure import InterestType, PrincipalType, TrancheSpec
from .market_simulator import MarketSimulator
from .client_demand import (
    ClientType, ClientDemand, DifficultyLevel,
    generate_client_scenario, compute_match_score,
)


@dataclass
class MatchResult:
    """Result of matching a tranche to a client."""
    tranche_name: str
    client_idx: int
    client_type: ClientType
    match_score: float
    accepted: bool


class MultiClientYieldBookEnv(YieldBookEnv):
    """
    Multi-client CMO structuring environment.

    Observation space: base 129 dims + (max_clients x 15) client dims.
    Action space: unchanged MultiDiscrete([14, 10, 20, 20]).

    The agent structures the deal as before. Upon EXECUTE_DEAL, the environment:
    1. Prices all tranches (via parent class)
    2. Greedy-matches tranches to clients based on match scores
    3. Computes multi-objective reward combining dealer profit, client
       satisfaction, and inventory fill rate
    """

    def __init__(
        self,
        max_steps: int = 30,
        max_tranches: int = 10,
        collateral_balance: float = 100_000_000,
        seed: Optional[int] = None,
        deal_mode: str = "AGENCY",
        render_mode: Optional[str] = None,
        market_simulator: Optional[MarketSimulator] = None,
        # Multi-client params
        max_clients: int = 5,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        # Reward weights
        dealer_arb_weight: float = 0.6,
        client_satisfaction_weight: float = 0.3,
        completion_weight: float = 0.1,
        unsold_penalty_per_10m: float = 5.0,
    ):
        super().__init__(
            max_steps=max_steps,
            max_tranches=max_tranches,
            collateral_balance=collateral_balance,
            seed=seed,
            deal_mode=deal_mode,
            render_mode=render_mode,
            market_simulator=market_simulator,
        )

        self.max_clients = max_clients
        self.difficulty = difficulty
        self.dealer_arb_weight = dealer_arb_weight
        self.client_satisfaction_weight = client_satisfaction_weight
        self.completion_weight = completion_weight
        self.unsold_penalty_per_10m = unsold_penalty_per_10m

        # Client state
        self.clients: list[ClientDemand] = []
        self._match_results: list[MatchResult] = []

        # Expand observation space: base + client dims
        self._client_dim = max_clients * 15
        obs_dim = self._market_dim + self._deal_dim + self._client_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

    def _get_client_obs(self) -> np.ndarray:
        """Build client observation vector."""
        client_vecs = []
        for i in range(self.max_clients):
            if i < len(self.clients):
                client_vecs.append(self.clients[i].to_vector())
            else:
                client_vecs.append(np.zeros(15, dtype=np.float32))
        return np.concatenate(client_vecs)

    def _get_obs(self) -> np.ndarray:
        """Build observation: market + deal + clients."""
        base_obs = super()._get_obs()
        client_obs = self._get_client_obs()
        return np.concatenate([base_obs, client_obs]).astype(np.float32)

    def _get_info(self) -> dict:
        """Return auxiliary info including client details."""
        info = super()._get_info()
        active_clients = [c for c in self.clients if c.active]
        info["n_clients"] = len(active_clients)
        info["client_types"] = [ClientType(c.client_type).name for c in active_clients]
        info["difficulty"] = DifficultyLevel(self.difficulty).name
        if self._match_results:
            info["matches"] = [
                {
                    "tranche": m.tranche_name,
                    "client": m.client_type.name,
                    "score": round(m.match_score, 3),
                    "accepted": m.accepted,
                }
                for m in self._match_results
            ]
        return info

    def reset(self, seed=None, options=None):
        """Reset with new market scenario and client mix."""
        obs, info = super().reset(seed=seed, options=options)

        # Override difficulty if provided in options
        difficulty = self.difficulty
        if options and "difficulty" in options:
            difficulty = DifficultyLevel(options["difficulty"])

        # Generate client scenario
        rng = np.random.RandomState(seed if seed is not None else self._seed)
        regime = self.scenario.regime if self.scenario else "normal"
        self.clients = generate_client_scenario(
            difficulty=difficulty,
            regime=regime,
            rng=rng,
            collateral_balance=self.collateral_balance,
            max_clients=self.max_clients,
        )
        self._match_results = []

        # Rebuild obs with client dims
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _estimate_tranche_wal(self, tranche: TrancheSpec) -> float:
        """Rough WAL estimate based on tranche type and priority."""
        base_wal = 5.0 + tranche.priority * 1.5

        if tranche.is_z_bond:
            base_wal = 15.0 + tranche.priority * 2.0
        elif tranche.principal_type == PrincipalType.PAC:
            base_wal = 3.0 + tranche.priority * 1.0
        elif tranche.principal_type == PrincipalType.SUPPORT:
            base_wal = 4.0 + tranche.priority * 2.0
        elif tranche.principal_type in (PrincipalType.VADM, PrincipalType.Z_PAC):
            base_wal = 12.0 + tranche.priority * 1.5
        elif tranche.principal_type == PrincipalType.NAS:
            base_wal = 2.0 + tranche.lockout_months / 12.0
        elif tranche.is_io:
            base_wal = 5.0
        elif tranche.is_po:
            base_wal = 7.0

        # Cap to reasonable range
        return float(np.clip(base_wal, 0.5, 30.0))

    def _estimate_tranche_duration(self, tranche: TrancheSpec) -> float:
        """Rough duration estimate based on tranche type."""
        wal = self._estimate_tranche_wal(tranche)

        if tranche.interest_type == InterestType.FLOATING:
            return min(wal, 0.5)  # Floaters have near-zero duration
        elif tranche.interest_type == InterestType.INVERSE_FLOATING:
            return wal * 2.5  # Inverse floaters have leveraged duration
        elif tranche.is_io:
            return -wal * 0.8  # IO has negative duration
        elif tranche.is_po:
            return wal * 1.3  # PO has high positive duration
        elif tranche.is_z_bond:
            return wal * 1.1  # Z-bonds slightly longer than WAL
        else:
            return wal * 0.85  # Fixed-rate ~ 85% of WAL

    def _match_tranches_to_clients(self) -> tuple[list[MatchResult], float, float]:
        """Greedy matching: for each client, find best unallocated tranche.

        Returns:
            matches: list of MatchResult
            avg_score: average match score across accepted matches
            fill_rate: fraction of active clients that got a tranche
        """
        if self.deal is None:
            return [], 0.0, 0.0

        active_clients = [(i, c) for i, c in enumerate(self.clients) if c.active]
        if not active_clients:
            return [], 0.0, 0.0

        allocated_tranches: set[str] = set()
        matches: list[MatchResult] = []
        accepted_scores: list[float] = []

        # For each client, find best available tranche
        for client_idx, client in active_clients:
            best_tranche = None
            best_score = -1.0

            for tranche in self.deal.tranches:
                if tranche.name in allocated_tranches:
                    continue

                wal = self._estimate_tranche_wal(tranche)
                dur = self._estimate_tranche_duration(tranche)
                score = compute_match_score(tranche, client, wal, dur)

                if score > best_score:
                    best_score = score
                    best_tranche = tranche

            if best_tranche is not None:
                accepted = best_score >= client.acceptance_threshold
                match = MatchResult(
                    tranche_name=best_tranche.name,
                    client_idx=client_idx,
                    client_type=client.client_type,
                    match_score=best_score,
                    accepted=accepted,
                )
                matches.append(match)
                if accepted:
                    allocated_tranches.add(best_tranche.name)
                    accepted_scores.append(best_score)

        avg_score = float(np.mean(accepted_scores)) if accepted_scores else 0.0
        n_accepted = len(accepted_scores)
        fill_rate = n_accepted / len(active_clients) if active_clients else 0.0

        return matches, avg_score, fill_rate

    def _execute_deal(self) -> tuple[float, bool]:
        """Execute deal with multi-objective reward.

        Reward = dealer_arb_weight * dealer_profit_ticks
               + client_satisfaction_weight * avg_match_score * 10
               + completion_weight * fill_rate * 5
               - unsold_penalty
        """
        # Get base dealer profit from parent
        base_reward, terminated = super()._execute_deal()

        if self.deal is None:
            return base_reward, terminated

        # Match tranches to clients
        matches, avg_score, fill_rate = self._match_tranches_to_clients()
        self._match_results = matches

        # Multi-objective reward
        dealer_component = self.dealer_arb_weight * base_reward
        satisfaction_component = self.client_satisfaction_weight * avg_score * 10.0
        completion_component = self.completion_weight * fill_rate * 5.0

        # Unsold inventory penalty
        unsold_notional = 0.0
        allocated_names = {m.tranche_name for m in matches if m.accepted}
        for t in self.deal.tranches:
            if t.name not in allocated_names:
                face = t.notional_balance if t.is_io else t.original_balance
                unsold_notional += face
        unsold_penalty = (unsold_notional / 10_000_000) * self.unsold_penalty_per_10m

        reward = dealer_component + satisfaction_component + completion_component - unsold_penalty

        return reward, terminated

    def render(self):
        """Render deal state plus client information."""
        output = super().render()
        if output is None:
            return None

        lines = [output, ""]

        # Client info
        lines.append("─── Client Demands ───")
        for i, c in enumerate(self.clients):
            if not c.active:
                continue
            lines.append(
                f"  [{i}] {ClientType(c.client_type).name:15s} "
                f"${c.notional:>12,.0f}  "
                f"WAL [{c.wal_min:.0f}-{c.wal_max:.0f}yr]  "
                f"Thresh {c.acceptance_threshold:.2f}"
            )

        # Match results
        if self._match_results:
            lines.append("")
            lines.append("─── Tranche-Client Matches ───")
            for m in self._match_results:
                status = "ACCEPTED" if m.accepted else "REJECTED"
                lines.append(
                    f"  {m.tranche_name:<12s} -> {m.client_type.name:<15s} "
                    f"score={m.match_score:.3f} [{status}]"
                )

        full_output = "\n".join(lines)
        if self.render_mode == "human":
            print(full_output)
        return full_output


def make_multi_client_env(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    max_clients: int = 5,
    collateral_balance: float = 100_000_000,
    seed: Optional[int] = None,
    max_steps: int = 30,
    render_mode: Optional[str] = None,
) -> MultiClientYieldBookEnv:
    """Factory function for multi-client environments."""
    return MultiClientYieldBookEnv(
        max_steps=max_steps,
        collateral_balance=collateral_balance,
        seed=seed,
        difficulty=difficulty,
        max_clients=max_clients,
        render_mode=render_mode,
    )
