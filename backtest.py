"""
Backtesting CMO Agent Against Historical Dealer Performance.

Replays 742 real REMIC deals through the RL environment and compares:
- Agent's structuring profit vs what dealers actually achieved
- Performance by dealer, period, and deal complexity
- Alpha generation metrics

Usage:
    from cmo_agent.backtest import backtest_agent_vs_experts
    results = backtest_agent_vs_experts("models/gse_shelf_200k.pt")
"""
import json
import os
import time
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .yield_book_env import YieldBookEnv, make_yield_book_env, ActionType
from .train import PolicyNetwork


@dataclass
class DealBacktestResult:
    """Result of backtesting a single deal."""
    deal_id: str
    dealer: str
    period: str
    expert_reward: float
    agent_reward: float
    expert_n_actions: int
    agent_n_actions: int
    alpha: float  # agent_reward - expert_reward
    expert_structure_types: list[str] = field(default_factory=list)


@dataclass
class BacktestSummary:
    """Aggregate backtest results."""
    total_deals: int = 0
    agent_mean_reward: float = 0.0
    expert_mean_reward: float = 0.0
    mean_alpha: float = 0.0
    win_rate: float = 0.0  # fraction where agent > expert
    by_dealer: dict = field(default_factory=dict)
    by_period: dict = field(default_factory=dict)
    by_complexity: dict = field(default_factory=dict)
    deal_results: list[DealBacktestResult] = field(default_factory=list)


def replay_expert_demo(
    env: YieldBookEnv,
    actions: list[list[int]],
    seed: int = 42,
) -> tuple[float, int]:
    """Replay an expert action sequence through the env.

    Returns:
        total_reward: cumulative reward
        n_steps: number of steps taken
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_steps = 0

    for action in actions:
        # Ensure action is valid for the env
        action_clipped = [
            min(action[0], 13),
            min(action[1], 9),
            min(action[2], 19),
            min(action[3], 19),
        ]
        obs, reward, terminated, truncated, _ = env.step(action_clipped)
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            break

    return total_reward, n_steps


def _adapt_obs(obs: np.ndarray, policy_obs_dim: int) -> np.ndarray:
    """Adapt observation to match policy's expected dimension.

    Handles cases where env obs space changed after model was trained.
    Truncates extra dims or pads with zeros as needed.
    """
    if len(obs) == policy_obs_dim:
        return obs
    elif len(obs) > policy_obs_dim:
        return obs[:policy_obs_dim]
    else:
        return np.concatenate([obs, np.zeros(policy_obs_dim - len(obs), dtype=obs.dtype)])


def replay_agent_policy(
    env: YieldBookEnv,
    policy: PolicyNetwork,
    seed: int = 42,
    deterministic: bool = True,
) -> tuple[float, int]:
    """Run the trained RL policy through the env.

    Returns:
        total_reward: cumulative reward
        n_steps: number of steps taken
    """
    # Detect policy's expected obs dim from its first layer
    policy_obs_dim = policy.shared[0].in_features

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_steps = 0
    done = False

    while not done:
        adapted = _adapt_obs(obs, policy_obs_dim)
        obs_tensor = torch.tensor(adapted, dtype=torch.float32)
        action, _, _ = policy.get_action(obs_tensor, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        n_steps += 1
        done = terminated or truncated

    return total_reward, n_steps


def load_policy(
    model_path: str,
    obs_dim: int = 129,
    action_dims: Optional[list[int]] = None,
    hidden_dim: int = 256,
) -> PolicyNetwork:
    """Load a trained policy from checkpoint.

    Auto-detects obs_dim from checkpoint if possible to handle
    env observation space changes between training and inference.
    Checks model registry first for metadata.
    """
    if action_dims is None:
        action_dims = [14, 10, 20, 20]

    # Try registry first for metadata
    try:
        from .model_registry import get_model
        model_id = os.path.basename(model_path).replace(".pt", "")
        meta = get_model(model_id)
        if meta:
            obs_dim = meta["obs_dim"]
            action_dims = meta["action_dims"]
            hidden_dim = meta["hidden_dim"]
    except Exception:
        pass

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    # Auto-detect obs_dim from saved weights
    if "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = checkpoint

    # The first linear layer shape tells us the obs_dim used during training
    if "shared.0.weight" in state_dict:
        saved_obs_dim = state_dict["shared.0.weight"].shape[1]
        if saved_obs_dim != obs_dim:
            obs_dim = saved_obs_dim

    # Detect hidden dim from weights
    if "shared.0.weight" in state_dict:
        hidden_dim = state_dict["shared.0.weight"].shape[0]

    # Auto-detect action_dims from policy head output layers
    detected_dims = []
    for i in range(10):  # Up to 10 heads
        key = f"policy_heads.{i}.2.weight"
        if key in state_dict:
            detected_dims.append(state_dict[key].shape[0])
        else:
            break
    if detected_dims:
        action_dims = detected_dims

    policy = PolicyNetwork(obs_dim, action_dims, hidden_dim)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def backtest_agent_vs_experts(
    model_path: str,
    demo_file: str = "expert_demonstrations.json",
    n_demos: int = 0,
    seed_base: int = 42,
    verbose: bool = True,
) -> BacktestSummary:
    """Backtest trained agent against 742 real dealer demonstrations.

    For each deal:
    1. Reset env with the same seed
    2. Replay expert actions → expert_reward
    3. Run agent policy → agent_reward
    4. Compute alpha = agent - expert

    Args:
        model_path: Path to trained .pt model
        demo_file: Path to expert_demonstrations.json
        n_demos: Max demos to test (0 = all)
        seed_base: Base seed for reproducibility
        verbose: Print progress
    """
    # Load demos
    with open(demo_file) as f:
        demos = json.load(f)
    if n_demos > 0:
        demos = demos[:n_demos]

    if verbose:
        print(f"Backtesting {len(demos)} deals from {demo_file}")
        print(f"Loading model from {model_path}")

    # Create env and load policy
    env = make_yield_book_env(mode="AGENCY", seed=seed_base)
    policy = load_policy(model_path, obs_dim=env.observation_space.shape[0])

    summary = BacktestSummary()
    dealer_results = defaultdict(list)
    period_results = defaultdict(list)
    complexity_results = defaultdict(list)

    t0 = time.time()

    for i, demo in enumerate(demos):
        deal_seed = seed_base + i

        # Replay expert actions
        expert_reward, expert_steps = replay_expert_demo(
            env, demo["actions"], seed=deal_seed
        )

        # Run agent policy
        agent_reward, agent_steps = replay_agent_policy(
            env, policy, seed=deal_seed
        )

        alpha = agent_reward - expert_reward
        deal_id = demo.get("deal_id", f"deal_{i}")
        dealer = demo.get("dealer", "Unknown")
        period = demo.get("period", "Unknown")
        structure_types = demo.get("structure_types", [])

        result = DealBacktestResult(
            deal_id=deal_id,
            dealer=dealer,
            period=period,
            expert_reward=expert_reward,
            agent_reward=agent_reward,
            expert_n_actions=expert_steps,
            agent_n_actions=agent_steps,
            alpha=alpha,
            expert_structure_types=structure_types,
        )
        summary.deal_results.append(result)

        # Aggregate by dealer
        dealer_results[dealer].append(result)

        # Aggregate by period (year)
        year = period.split()[-1] if period else "Unknown"
        period_results[year].append(result)

        # Aggregate by complexity
        n_types = len(set(structure_types))
        if n_types <= 2:
            complexity = "simple"
        elif n_types <= 4:
            complexity = "moderate"
        else:
            complexity = "complex"
        complexity_results[complexity].append(result)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg_alpha = np.mean([r.alpha for r in summary.deal_results])
            print(
                f"  [{i+1}/{len(demos)}] "
                f"Avg alpha: {avg_alpha:+.2f} ticks | "
                f"Win rate: {sum(1 for r in summary.deal_results if r.alpha > 0)/len(summary.deal_results)*100:.1f}% | "
                f"{elapsed:.1f}s"
            )

    # Compute summary stats
    n = len(summary.deal_results)
    summary.total_deals = n
    summary.agent_mean_reward = float(np.mean([r.agent_reward for r in summary.deal_results]))
    summary.expert_mean_reward = float(np.mean([r.expert_reward for r in summary.deal_results]))
    summary.mean_alpha = float(np.mean([r.alpha for r in summary.deal_results]))
    summary.win_rate = sum(1 for r in summary.deal_results if r.alpha > 0) / max(1, n)

    # By dealer
    for dealer, results in dealer_results.items():
        # Normalize dealer names (many variants for same bank)
        clean_dealer = _normalize_dealer(dealer)
        if clean_dealer not in summary.by_dealer:
            summary.by_dealer[clean_dealer] = {
                "n_deals": 0, "agent_mean": 0, "expert_mean": 0,
                "mean_alpha": 0, "win_rate": 0,
            }
        entry = summary.by_dealer[clean_dealer]
        all_results = results
        entry["n_deals"] += len(all_results)
        entry["agent_mean"] = round(float(np.mean([r.agent_reward for r in all_results])), 2)
        entry["expert_mean"] = round(float(np.mean([r.expert_reward for r in all_results])), 2)
        entry["mean_alpha"] = round(float(np.mean([r.alpha for r in all_results])), 2)
        entry["win_rate"] = round(sum(1 for r in all_results if r.alpha > 0) / max(1, len(all_results)), 3)

    # By period
    for period, results in sorted(period_results.items()):
        summary.by_period[period] = {
            "n_deals": len(results),
            "agent_mean": round(float(np.mean([r.agent_reward for r in results])), 2),
            "expert_mean": round(float(np.mean([r.expert_reward for r in results])), 2),
            "mean_alpha": round(float(np.mean([r.alpha for r in results])), 2),
            "win_rate": round(sum(1 for r in results if r.alpha > 0) / max(1, len(results)), 3),
        }

    # By complexity
    for complexity, results in complexity_results.items():
        summary.by_complexity[complexity] = {
            "n_deals": len(results),
            "agent_mean": round(float(np.mean([r.agent_reward for r in results])), 2),
            "expert_mean": round(float(np.mean([r.expert_reward for r in results])), 2),
            "mean_alpha": round(float(np.mean([r.alpha for r in results])), 2),
            "win_rate": round(sum(1 for r in results if r.alpha > 0) / max(1, len(results)), 3),
        }

    elapsed = time.time() - t0
    if verbose:
        print(f"\n{'='*60}")
        print(f"Backtest Complete: {n} deals in {elapsed:.1f}s")
        print(f"  Agent mean reward:  {summary.agent_mean_reward:>7.2f} ticks")
        print(f"  Expert mean reward: {summary.expert_mean_reward:>7.2f} ticks")
        print(f"  Mean alpha:         {summary.mean_alpha:>+7.2f} ticks")
        print(f"  Win rate:           {summary.win_rate*100:.1f}%")
        print(f"\n  By Dealer:")
        for dealer, stats in sorted(summary.by_dealer.items(), key=lambda x: -x[1]["n_deals"]):
            print(f"    {dealer:30s}: {stats['n_deals']:>3d} deals | "
                  f"alpha={stats['mean_alpha']:>+6.2f} | "
                  f"win={stats['win_rate']*100:.0f}%")
        print(f"\n  By Year:")
        for year, stats in sorted(summary.by_period.items()):
            print(f"    {year}: {stats['n_deals']:>3d} deals | "
                  f"alpha={stats['mean_alpha']:>+6.2f} | "
                  f"win={stats['win_rate']*100:.0f}%")
        print(f"\n  By Complexity:")
        for comp, stats in summary.by_complexity.items():
            print(f"    {comp:10s}: {stats['n_deals']:>3d} deals | "
                  f"alpha={stats['mean_alpha']:>+6.2f} | "
                  f"win={stats['win_rate']*100:.0f}%")
        print(f"{'='*60}")

    return summary


def _normalize_dealer(name: str) -> str:
    """Normalize dealer name variants to canonical form."""
    name_lower = name.lower()
    if "jp morgan" in name_lower or "j.p. morgan" in name_lower:
        return "JP Morgan"
    if "goldman" in name_lower:
        return "Goldman Sachs"
    if "citi" in name_lower:
        return "Citigroup"
    if "bmo" in name_lower:
        return "BMO"
    if "mizuho" in name_lower:
        return "Mizuho"
    if "morgan stanley" in name_lower:
        return "Morgan Stanley"
    if "bofa" in name_lower or "bank of america" in name_lower:
        return "BofA"
    if "barclays" in name_lower:
        return "Barclays"
    if "wells fargo" in name_lower:
        return "Wells Fargo"
    if "nomura" in name_lower:
        return "Nomura"
    if "truist" in name_lower:
        return "Truist"
    if "credit suisse" in name_lower:
        return "Credit Suisse"
    if "bnp" in name_lower:
        return "BNP Paribas"
    if "santander" in name_lower:
        return "Santander"
    return name


def backtest_to_json(summary: BacktestSummary) -> dict:
    """Convert backtest summary to JSON-serializable dict."""
    return {
        "total_deals": summary.total_deals,
        "agent_mean_reward": round(summary.agent_mean_reward, 2),
        "expert_mean_reward": round(summary.expert_mean_reward, 2),
        "mean_alpha": round(summary.mean_alpha, 2),
        "win_rate": round(summary.win_rate, 3),
        "by_dealer": summary.by_dealer,
        "by_period": summary.by_period,
        "by_complexity": summary.by_complexity,
        "top_deals": [
            {
                "deal_id": r.deal_id,
                "dealer": r.dealer,
                "period": r.period,
                "alpha": round(r.alpha, 2),
                "agent_reward": round(r.agent_reward, 2),
                "expert_reward": round(r.expert_reward, 2),
            }
            for r in sorted(summary.deal_results, key=lambda x: -x.alpha)[:20]
        ],
        "worst_deals": [
            {
                "deal_id": r.deal_id,
                "dealer": r.dealer,
                "period": r.period,
                "alpha": round(r.alpha, 2),
                "agent_reward": round(r.agent_reward, 2),
                "expert_reward": round(r.expert_reward, 2),
            }
            for r in sorted(summary.deal_results, key=lambda x: x.alpha)[:10]
        ],
    }
