"""
Multi-Client Curriculum Training Pipeline for CMO Structuring.

Trains agents via curriculum learning across increasing difficulty:
  Stage 1: EASY   (1-2 aligned clients,    100k steps)
  Stage 2: MEDIUM (2-3 moderate conflicts, 150k steps)
  Stage 3: HARD   (3-4 competing demands,  150k steps)
  Stage 4: EXPERT (4-5 adversarial,        100k steps)

Transfers weights between stages for progressive learning.
"""
import numpy as np
import torch
import time
from typing import Optional
from dataclasses import dataclass

from .train import PPOTrainer, PPOConfig
from .multi_client_env import MultiClientYieldBookEnv, make_multi_client_env
from .client_demand import DifficultyLevel, ClientType


# ─── Curriculum Config ────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    """Configuration for one stage of curriculum training."""
    difficulty: DifficultyLevel
    total_timesteps: int
    lr: float
    entropy_coeff: float
    label: str


DEFAULT_CURRICULUM = [
    CurriculumStage(DifficultyLevel.EASY,   100_000, 3e-4, 0.03, "Stage 1: EASY"),
    CurriculumStage(DifficultyLevel.MEDIUM, 150_000, 1e-4, 0.02, "Stage 2: MEDIUM"),
    CurriculumStage(DifficultyLevel.HARD,   150_000, 1e-4, 0.015, "Stage 3: HARD"),
    CurriculumStage(DifficultyLevel.EXPERT, 100_000, 5e-5, 0.01, "Stage 4: EXPERT"),
]


# ─── Multi-Client Heuristic Agent ─────────────────────────────────────────

class MultiClientHeuristicAgent:
    """
    Heuristic agent adapted for multi-client environments.

    Adjusts strategy based on which client types are present:
    - If hedge fund present: include inverse floater + IO
    - If insurance/pension present: include Z-bond
    - If bank present: include floater
    - Default to kitchen sink for best coverage
    """

    def __init__(self, env: MultiClientYieldBookEnv):
        self.env = env

    def _pick_strategy(self, clients: list) -> str:
        """Choose strategy based on active client types."""
        active_types = set()
        for c in clients:
            if c.active:
                active_types.add(ClientType(c.client_type))

        has_hf = ClientType.HEDGE_FUND in active_types
        has_ins = ClientType.INSURANCE in active_types or ClientType.PENSION_FUND in active_types
        has_bank = ClientType.BANK in active_types

        # Kitchen sink covers the most ground
        if has_hf and has_ins:
            return "kitchen_sink"
        elif has_hf:
            return "floater_inverse"
        elif has_ins:
            return "pac_io_z"
        elif has_bank:
            return "floater_inverse"
        else:
            return "kitchen_sink"

    def act(self, obs: np.ndarray, step: int, clients: list) -> list[int]:
        """Select action based on client-aware heuristics."""
        strategy = self._pick_strategy(clients)
        at_collateral = 13
        below_collateral = 10
        above_collateral = 16

        if strategy == "kitchen_sink":
            return self._kitchen_sink(step, at_collateral, below_collateral, above_collateral)
        elif strategy == "floater_inverse":
            return self._floater_inverse(step, at_collateral, below_collateral, above_collateral)
        elif strategy == "pac_io_z":
            return self._pac_io_z(step, at_collateral, below_collateral, above_collateral)
        else:
            return self._pac_support(step, at_collateral)

    def _kitchen_sink(self, step, at, below, above):
        if step == 0: return [1, 0, 11, below]    # PAC 28%
        if step == 1: return [7, 1, 8, at]         # Floater 20%
        if step == 2: return [8, 2, 5, above]      # Inv Floater 12%
        if step == 3: return [3, 3, 7, above]      # Support 18%
        if step == 4: return [4, 4, 5, at]         # Z-bond 12%
        if step == 5: return [5, 5, 11, at]        # IO strip
        if step == 6: return [0, 6, 3, at]         # SEQ remainder
        return [12, 0, 0, 0]

    def _floater_inverse(self, step, at, below, above):
        if step == 0: return [1, 0, 14, below]
        if step == 1: return [7, 1, 8, at]
        if step == 2: return [8, 2, 5, above]
        if step == 3: return [3, 3, 8, above]
        if step == 4: return [0, 4, 4, at]
        return [12, 0, 0, 0]

    def _pac_io_z(self, step, at, below, above):
        if step == 0: return [1, 0, 15, below]
        if step == 1: return [3, 1, 10, above]
        if step == 2: return [4, 2, 7, at]
        if step == 3: return [5, 3, 12, at]
        if step == 4: return [0, 4, 5, at]
        return [12, 0, 0, 0]

    def _pac_support(self, step, at):
        if step == 0: return [1, 0, 16, at]
        if step == 1: return [3, 1, 11, at]
        if step == 2: return [0, 2, 7, at]
        return [12, 0, 0, 0]

    def evaluate(self, n_episodes: int = 50, difficulty: Optional[DifficultyLevel] = None) -> dict:
        """Run heuristic agent and collect stats."""
        rewards = []
        match_scores = []
        fill_rates = []

        for ep in range(n_episodes):
            options = {"difficulty": int(difficulty)} if difficulty is not None else None
            obs, info = self.env.reset(seed=ep, options=options)
            total_reward = 0.0
            done = False
            step = 0
            while not done:
                action = self.act(obs, step, self.env.clients)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
            rewards.append(total_reward)

            # Extract match info if available
            if "matches" in info:
                scores = [m["score"] for m in info["matches"] if m["accepted"]]
                if scores:
                    match_scores.append(np.mean(scores))
                fill = sum(1 for m in info["matches"] if m["accepted"]) / max(1, info.get("n_clients", 1))
                fill_rates.append(fill)

        result = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }
        if match_scores:
            result["mean_match_score"] = float(np.mean(match_scores))
        if fill_rates:
            result["mean_fill_rate"] = float(np.mean(fill_rates))
        return result


# ─── Curriculum Training ──────────────────────────────────────────────────

def train_multi_client_agent(
    curriculum: Optional[list[CurriculumStage]] = None,
    seed: int = 42,
    max_clients: int = 5,
    save_path: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """Train agent with 4-stage curriculum learning.

    Transfers weights between stages for progressive difficulty scaling.
    """
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM

    all_results = {}
    policy_state = None
    total_start = time.time()

    for stage_idx, stage in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"{stage.label} | difficulty={stage.difficulty.name} | "
              f"{stage.total_timesteps:,} steps | lr={stage.lr}")
        print(f"{'='*60}")

        # Create env for this difficulty
        env = make_multi_client_env(
            difficulty=stage.difficulty,
            max_clients=max_clients,
            seed=seed + stage_idx * 1000,
        )
        eval_env = make_multi_client_env(
            difficulty=stage.difficulty,
            max_clients=max_clients,
            seed=seed + stage_idx * 1000 + 500,
        )

        config = PPOConfig(
            total_timesteps=stage.total_timesteps,
            lr=stage.lr,
            entropy_coeff=stage.entropy_coeff,
            entropy_min=max(0.001, stage.entropy_coeff * 0.1),
            entropy_anneal_steps=stage.total_timesteps // 2,
            rollout_steps=512,
            hidden_dim=256,
            n_epochs=4,
            batch_size=64,
            device=device,
        )

        trainer = PPOTrainer(env, config, eval_env)

        # Transfer weights from previous stage
        if policy_state is not None:
            try:
                trainer.policy.load_state_dict(policy_state)
                print("  Transferred weights from previous stage")
            except RuntimeError:
                print("  Weight transfer failed (architecture mismatch), starting fresh")

        # Update learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = stage.lr

        results = trainer.train()
        policy_state = trainer.policy.state_dict()

        all_results[stage.label] = {
            "final_eval_reward": results["eval_rewards"][-1] if results["eval_rewards"] else 0.0,
            "mean_train_reward": float(np.mean(results["episode_rewards"][-50:])) if results["episode_rewards"] else 0.0,
            "total_episodes": len(results["episode_rewards"]),
            "timesteps": results["total_timesteps"],
            "training_time": results["training_time"],
        }

    total_time = time.time() - total_start

    if save_path and policy_state is not None:
        torch.save({
            "policy_state_dict": policy_state,
            "curriculum_results": all_results,
            "max_clients": max_clients,
        }, save_path)
        print(f"\nModel saved to {save_path}")

    print(f"\n{'='*60}")
    print(f"Curriculum training complete in {total_time:.1f}s")
    for label, res in all_results.items():
        print(f"  {label}: eval={res['final_eval_reward']:.2f} train={res['mean_train_reward']:.2f}")
    print(f"{'='*60}")

    return all_results


# ─── Benchmarking ─────────────────────────────────────────────────────────

def benchmark_multi_client_agents(
    n_episodes: int = 50,
    seed: int = 42,
    max_clients: int = 5,
) -> dict:
    """Benchmark heuristic agent across all difficulty levels."""
    print("=== Multi-Client Agent Benchmark ===\n")

    all_results = {}

    for difficulty in DifficultyLevel:
        env = make_multi_client_env(
            difficulty=difficulty,
            max_clients=max_clients,
            seed=seed,
        )
        agent = MultiClientHeuristicAgent(env)
        result = agent.evaluate(n_episodes=n_episodes, difficulty=difficulty)
        all_results[difficulty.name] = result

        fill_str = f"fill={result.get('mean_fill_rate', 0):.2f}" if "mean_fill_rate" in result else ""
        match_str = f"match={result.get('mean_match_score', 0):.3f}" if "mean_match_score" in result else ""
        print(
            f"  {difficulty.name:8s}: mean={result['mean_reward']:>7.2f} "
            f"std={result['std_reward']:>6.2f} "
            f"{match_str} {fill_str}"
        )

    best = max(all_results.items(), key=lambda x: x[1]["mean_reward"])
    print(f"\nBest difficulty: {best[0]} (mean={best[1]['mean_reward']:.2f})")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-client CMO structuring training")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    parser.add_argument("--train", action="store_true", help="Run curriculum training")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--save", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_multi_client_agents(n_episodes=args.episodes, seed=args.seed, max_clients=args.clients)
    elif args.train:
        train_multi_client_agent(seed=args.seed, max_clients=args.clients, save_path=args.save, device=args.device)
    else:
        # Default: benchmark then train
        benchmark_multi_client_agents(n_episodes=args.episodes, seed=args.seed, max_clients=args.clients)
