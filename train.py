"""
RL Training Pipeline for CMO Structuring Agent.

Trains agents to structure CMO deals using the Yield Book RL environment.
Supports:
- PPO (Proximal Policy Optimization) via custom implementation
- DQN (Deep Q-Network) for discrete action selection
- Behavioral cloning from expert dealer demonstrations
- Curriculum learning with regime scheduling
- Cosine LR scheduling with warmup
- Running observation/reward normalization
- Vectorized parallel rollout collection
- EMA policy for stable evaluation
- Best-model checkpointing
- Multi-desk FICC training coordination
- CMBS-specific training pipeline

No external RL library dependency - pure PyTorch implementation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional
import json
import time
import math
import copy
import os

from .yield_book_env import YieldBookEnv, CMBSYieldBookEnv, make_yield_book_env, ActionType


# ─── Neural Network Architectures ─────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Actor-Critic network for PPO with layer norm for stable training."""

    def __init__(self, obs_dim: int, action_dims: list[int], hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Separate heads for each action dimension
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, dim),
            ) for dim in action_dims
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        features = self.shared(x)
        logits = [head(features) for head in self.policy_heads]
        value = self.value_head(features)
        return logits, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        with torch.no_grad():
            logits, value = self.forward(obs.unsqueeze(0))

        actions = []
        log_probs = []
        for head_logits in logits:
            dist = Categorical(logits=head_logits.squeeze(0))
            if deterministic:
                action = dist.probs.argmax()
            else:
                action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))

        total_log_prob = torch.stack(log_probs).sum()
        return actions, total_log_prob.item(), value.item()

    def evaluate_actions(self, obs, actions):
        """Evaluate log probs and values for given obs/actions."""
        logits, values = self.forward(obs)

        total_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        total_entropy = torch.zeros(obs.shape[0], device=obs.device)

        for i, head_logits in enumerate(logits):
            dist = Categorical(logits=head_logits)
            total_log_prob += dist.log_prob(actions[:, i])
            total_entropy += dist.entropy()

        return total_log_prob, values.squeeze(-1), total_entropy


class QNetwork(nn.Module):
    """Q-Network for DQN - flattened action space."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─── Experience Buffer ─────────────────────────────────────────────────────

@dataclass
class Transition:
    obs: np.ndarray
    action: list[int]
    reward: float
    next_obs: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


class RolloutBuffer:
    """Buffer for PPO rollouts."""

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[list[int]] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def add(self, t: Transition):
        self.obs.append(t.obs)
        self.actions.append(t.action)
        self.rewards.append(t.reward)
        self.dones.append(t.done)
        self.log_probs.append(t.log_prob)
        self.values.append(t.value)

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages."""
        n = len(self.rewards)
        returns = np.zeros(n)
        advantages = np.zeros(n)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = self.values[t + 1] if t + 1 < n else last_value

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

    def to_tensors(self, device="cpu"):
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns, advantages = self.compute_returns()
        return obs, actions, log_probs, returns.to(device), advantages.to(device)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.obs)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ─── Training Utilities ──────────────────────────────────────────────────

class RunningMeanStd:
    """Welford online algorithm for running mean/std of observations or rewards."""

    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch)
        if batch.ndim == 1 and self.mean.ndim == 0:
            # Scalar case (rewards)
            batch_mean = batch.mean()
            batch_var = batch.var()
            batch_count = batch.shape[0]
        else:
            batch_mean = batch.mean(axis=0)
            batch_var = batch.var(axis=0)
            batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class CosineSchedule:
    """Cosine annealing with linear warmup for learning rate."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0,
                 lr_min: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        if self._step < self.warmup_steps:
            return self.base_lr * self._step / max(1, self.warmup_steps)
        progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class EMAPolicy:
    """Exponential moving average of policy weights for stable evaluation."""

    def __init__(self, policy: nn.Module, decay: float = 0.995):
        self.ema_policy = copy.deepcopy(policy)
        self.decay = decay
        for p in self.ema_policy.parameters():
            p.requires_grad_(False)

    def update(self, policy: nn.Module):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_policy.parameters(), policy.parameters()):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)

    def get_action(self, obs, deterministic=True):
        return self.ema_policy.get_action(obs, deterministic=deterministic)

    def state_dict(self):
        return self.ema_policy.state_dict()


class VecEnvWrapper:
    """Lightweight vectorized environment wrapper for parallel rollout collection.

    Runs N envs in lockstep (no multiprocessing — avoids pickle/fork issues).
    Still provides 2-4x throughput improvement via batched numpy ops.
    """

    def __init__(self, envs: list):
        self.envs = envs
        self.n_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(self):
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list

    def step(self, actions: list):
        """Step all envs with corresponding actions.

        Args:
            actions: list of N action arrays

        Returns:
            obs (N, obs_dim), rewards (N,), dones (N,), infos (list of N dicts)
        """
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            if done:
                # Auto-reset
                final_reward = r
                obs, reset_info = env.reset()
                info["terminal_reward"] = final_reward
            rewards.append(r)
            dones.append(done)
            obs_list.append(obs)
            infos.append(info)
        return np.array(obs_list), np.array(rewards), np.array(dones), infos


class CurriculumScheduler:
    """Schedule training difficulty by controlling market regime distribution.

    Starts with calm/normal regimes, gradually introduces crisis/inverted/vol scenarios.
    """

    REGIME_SCHEDULE = [
        # (frac_of_training, regime_weights)
        (0.0,  {"normal": 0.7, "steep": 0.2, "flat": 0.1}),
        (0.2,  {"normal": 0.4, "steep": 0.2, "flat": 0.15, "inverted": 0.15, "volatile": 0.1}),
        (0.5,  {"normal": 0.25, "steep": 0.15, "flat": 0.15, "inverted": 0.15, "volatile": 0.15, "crisis": 0.15}),
        (0.8,  {"normal": 0.15, "steep": 0.1, "flat": 0.1, "inverted": 0.2, "volatile": 0.2, "crisis": 0.25}),
    ]

    def __init__(self, total_steps: int):
        self.total_steps = total_steps

    def sample_regime(self, current_step: int, rng: np.random.RandomState = None) -> str:
        frac = current_step / max(1, self.total_steps)
        # Find the right bracket
        weights = self.REGIME_SCHEDULE[0][1]
        for threshold, w in self.REGIME_SCHEDULE:
            if frac >= threshold:
                weights = w
        regimes = list(weights.keys())
        probs = np.array([weights[r] for r in regimes])
        probs /= probs.sum()
        if rng is not None:
            return rng.choice(regimes, p=probs)
        return np.random.choice(regimes, p=probs)


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics collected per update."""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    imitation_loss: float = 0.0
    imitation_coeff: float = 0.0
    learning_rate: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    episode_length_mean: float = 0.0
    fps: float = 0.0
    n_updates: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v != 0.0 or k in ("policy_loss", "value_loss")}


class DealAnalyzer:
    """Analyze what deal structures the trained agent prefers.

    Runs the policy through N episodes and records action distributions,
    tranche type frequencies, sizing patterns, and per-regime performance.
    """

    ACTION_NAMES = [
        "ADD_SEQ", "ADD_PAC", "ADD_TAC", "ADD_SUPPORT", "ADD_Z_BOND",
        "ADD_IO", "ADD_PO", "ADD_FLOATER", "ADD_INV_FLOAT",
        "MODIFY_SIZE", "MODIFY_COUPON", "REMOVE_TRANCHE",
        "EXECUTE_DEAL", "NOOP", "SELECT_POOL",
    ]

    POOL_NAMES = ["TBA", "LLB", "HLTV", "NY", "LFICO", "INV", "GEO"]

    def __init__(self, policy: nn.Module, env: YieldBookEnv, device="cpu"):
        self.policy = policy
        self.env = env
        self.device = device

    def analyze(self, n_episodes: int = 200) -> dict:
        """Run N episodes and collect deal structure statistics."""
        action_counts = defaultdict(int)
        tranche_type_counts = defaultdict(int)
        pool_type_counts = defaultdict(int)
        size_buckets = []
        coupon_offsets = []
        rewards_by_regime = defaultdict(list)
        episode_lengths = []
        n_tranches_per_deal = []
        rewards = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            n_tranche_adds = 0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, _, _ = self.policy.get_action(obs_t, deterministic=True)
                action_type = action[0]
                action_counts[action_type] += 1

                if action_type <= 8:  # Tranche add actions
                    tranche_type_counts[action_type] += 1
                    size_buckets.append(action[2])
                    coupon_offsets.append(action[3])
                    n_tranche_adds += 1
                elif action_type == 14:  # SELECT_POOL
                    pool_type_counts[min(action[1], 6)] += 1

                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1

            rewards.append(total_reward)
            episode_lengths.append(step)
            n_tranches_per_deal.append(n_tranche_adds)
            regime = info.get("regime", "unknown")
            rewards_by_regime[regime].append(total_reward)

        # Compile statistics
        total_actions = sum(action_counts.values())
        result = {
            "n_episodes": n_episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_tranches_per_deal": float(np.mean(n_tranches_per_deal)),
            "action_distribution": {
                self.ACTION_NAMES[k]: round(v / total_actions * 100, 1)
                for k, v in sorted(action_counts.items())
            },
            "tranche_type_distribution": {
                self.ACTION_NAMES[k]: round(v / max(1, sum(tranche_type_counts.values())) * 100, 1)
                for k, v in sorted(tranche_type_counts.items())
            },
            "pool_selection_distribution": {
                self.POOL_NAMES[k]: round(v / max(1, sum(pool_type_counts.values())) * 100, 1)
                for k, v in sorted(pool_type_counts.items())
            },
            "avg_size_bucket": float(np.mean(size_buckets)) if size_buckets else 0,
            "avg_coupon_offset_bucket": float(np.mean(coupon_offsets)) if coupon_offsets else 0,
            "regime_performance": {
                regime: {
                    "mean": float(np.mean(rews)),
                    "std": float(np.std(rews)),
                    "n": len(rews),
                }
                for regime, rews in sorted(rewards_by_regime.items())
            },
        }
        return result

    def print_analysis(self, results: dict):
        """Pretty-print deal analysis results."""
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║           TRAINED AGENT DEAL ANALYSIS               ║")
        print("╚══════════════════════════════════════════════════════╝")
        print(f"  Episodes: {results['n_episodes']}")
        print(f"  Mean Reward: {results['mean_reward']:.2f} (std={results['std_reward']:.2f})")
        print(f"  Median Reward: {results['median_reward']:.2f}")
        print(f"  Avg Episode Length: {results['mean_episode_length']:.1f} steps")
        print(f"  Avg Tranches/Deal: {results['mean_tranches_per_deal']:.1f}")

        print("\n  Action Distribution:")
        for name, pct in results["action_distribution"].items():
            bar = "█" * int(pct / 2)
            print(f"    {name:18s} {pct:5.1f}% {bar}")

        print("\n  Tranche Type Preferences:")
        for name, pct in results["tranche_type_distribution"].items():
            bar = "█" * int(pct / 2)
            print(f"    {name:18s} {pct:5.1f}% {bar}")

        if results["pool_selection_distribution"]:
            print("\n  Pool Selection:")
            for name, pct in results["pool_selection_distribution"].items():
                bar = "█" * int(pct / 2)
                print(f"    {name:6s} {pct:5.1f}% {bar}")

        print("\n  Performance by Regime:")
        for regime, stats in results["regime_performance"].items():
            print(f"    {regime:12s}: mean={stats['mean']:>7.2f} std={stats['std']:>6.2f} (n={stats['n']})")


# ─── Expert Demo Buffer for Imitation Learning ───────────────────────────

class ExpertDemoBuffer:
    """Buffer of expert demonstrations for behavioral cloning / imitation.

    Loads 742 real REMIC dealer action sequences from expert_demonstrations.json,
    rolls them through the RL env to generate (observation, action) pairs,
    then provides mini-batch sampling for the imitation loss.
    """

    def __init__(self, demo_file: str, env: YieldBookEnv, max_demos: int = 0):
        self.obs_list: list[np.ndarray] = []
        self.action_list: list[list[int]] = []
        self._load_demos(demo_file, env, max_demos)

    def _load_demos(self, demo_file: str, env: YieldBookEnv, max_demos: int):
        """Roll expert action sequences through env to generate (obs, action) pairs."""
        with open(demo_file) as f:
            demos = json.load(f)
        if max_demos > 0:
            demos = demos[:max_demos]

        for i, demo in enumerate(demos):
            actions = demo["actions"]
            try:
                obs, _ = env.reset()
            except Exception:
                continue
            for action in actions:
                action_type = action[0]
                # Skip the EXECUTE_DEAL action for BC (we want to learn structuring, not when to stop)
                if action_type == 12:  # EXECUTE_DEAL
                    continue
                self.obs_list.append(obs.copy())
                self.action_list.append(action[:4])  # [type, idx, size, coupon]
                try:
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                    obs = next_obs
                except Exception:
                    break

        print(f"  ExpertDemoBuffer: {len(self.obs_list)} (obs, action) pairs from {len(demos)} demos")

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of (obs, action) pairs for imitation loss."""
        n = len(self.obs_list)
        indices = np.random.choice(n, min(batch_size, n), replace=(batch_size > n))
        obs = torch.tensor(
            np.array([self.obs_list[i] for i in indices]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [self.action_list[i] for i in indices],
            dtype=torch.long,
        )
        return obs, actions

    def __len__(self):
        return len(self.obs_list)


# ─── PPO Trainer ───────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.02         # Higher initial entropy for exploration
    entropy_min: float = 0.001          # Minimum entropy after annealing
    entropy_anneal_steps: int = 50_000  # Anneal entropy over this many steps
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 512            # Longer rollouts for more complete episodes
    total_timesteps: int = 100_000
    hidden_dim: int = 256
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    device: str = "cpu"
    # Imitation learning from expert
    imitation_coeff: float = 0.0        # Weight for expert demonstration loss
    imitation_anneal_steps: int = 20_000  # Anneal imitation over this many steps
    # LR scheduling
    use_lr_schedule: bool = True        # Cosine annealing with warmup
    warmup_steps: int = 5_000           # Linear warmup steps
    lr_min: float = 1e-6               # Minimum LR at end of cosine
    # Normalization
    normalize_obs: bool = False         # Running observation normalization
    normalize_rewards: bool = True      # Running reward normalization (std only)
    # Checkpointing
    save_best: bool = True             # Save best eval model
    save_freq: int = 50_000            # Save periodic checkpoints
    save_dir: str = "models"           # Directory for checkpoints
    # EMA
    use_ema: bool = True               # Exponential moving average policy for eval
    ema_decay: float = 0.995
    # Vectorized envs
    n_envs: int = 1                    # Number of parallel environments
    # KL early stopping per epoch
    target_kl: float = 0.03            # Stop epoch early if KL exceeds this
    # Curriculum
    use_curriculum: bool = False        # Use curriculum learning


class PPOTrainer:
    """Proximal Policy Optimization trainer for CMO structuring.

    Features:
    - Cosine LR scheduling with linear warmup
    - Running reward normalization
    - EMA policy for stable evaluation
    - Best-model checkpointing
    - KL-based epoch early stopping
    - Rich training diagnostics (clip fraction, approx KL, explained variance)
    - Optional curriculum learning
    - Vectorized env support
    """

    def __init__(
        self,
        env: YieldBookEnv,
        config: Optional[PPOConfig] = None,
        eval_env: Optional[YieldBookEnv] = None,
        expert_buffer: Optional[ExpertDemoBuffer] = None,
        vec_env: Optional[VecEnvWrapper] = None,
    ):
        self.env = env
        self.eval_env = eval_env or env
        self.config = config or PPOConfig()
        self.vec_env = vec_env

        obs_dim = env.observation_space.shape[0]
        action_dims = list(env.action_space.nvec)

        self.policy = PolicyNetwork(
            obs_dim, action_dims, self.config.hidden_dim
        ).to(self.config.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()
        self.expert_buffer = expert_buffer

        # LR scheduler
        self.lr_scheduler = None
        if self.config.use_lr_schedule:
            self.lr_scheduler = CosineSchedule(
                self.optimizer,
                total_steps=self.config.total_timesteps // self.config.rollout_steps,
                warmup_steps=self.config.warmup_steps // self.config.rollout_steps,
                lr_min=self.config.lr_min,
            )

        # EMA policy for stable evaluation
        self.ema = None
        if self.config.use_ema:
            self.ema = EMAPolicy(self.policy, decay=self.config.ema_decay)

        # Running reward normalization
        self.reward_rms = RunningMeanStd() if self.config.normalize_rewards else None
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if self.config.normalize_obs else None

        # Curriculum
        self.curriculum = None
        if self.config.use_curriculum:
            self.curriculum = CurriculumScheduler(self.config.total_timesteps)

        # Best model tracking
        self.best_eval_reward = -float("inf")
        self.best_model_path = None

        # Logging
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.eval_rewards: list[float] = []
        self.ema_eval_rewards: list[float] = []
        self.losses: list[dict] = []
        self.metrics_history: list[TrainingMetrics] = []
        self._timestep = 0

    def _current_entropy_coeff(self) -> float:
        """Anneal entropy coefficient over training."""
        if self._timestep >= self.config.entropy_anneal_steps:
            return self.config.entropy_min
        frac = self._timestep / self.config.entropy_anneal_steps
        return self.config.entropy_coeff * (1 - frac) + self.config.entropy_min * frac

    def _current_imitation_coeff(self) -> float:
        """Anneal imitation coefficient to 0 over training."""
        if self.config.imitation_coeff <= 0:
            return 0.0
        if self._timestep >= self.config.imitation_anneal_steps:
            return 0.0
        frac = self._timestep / self.config.imitation_anneal_steps
        return self.config.imitation_coeff * (1 - frac)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running stats."""
        if self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(1, -1) if obs.ndim == 1 else obs)
            return self.obs_rms.normalize(obs).astype(np.float32)
        return obs

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running std (not mean — preserves sign)."""
        if self.reward_rms is not None:
            self.reward_rms.update(np.array([reward]))
            return reward / max(np.sqrt(self.reward_rms.var + 1e-8), 1e-4)
        return reward

    def train(self) -> dict:
        """Run full training loop with all enhancements."""
        obs, info = self.env.reset()
        obs = self._normalize_obs(obs)
        episode_reward = 0.0
        episode_raw_reward = 0.0
        episode_length = 0
        timestep = 0

        start_time = time.time()
        last_log_time = start_time

        while timestep < self.config.total_timesteps:
            # Collect rollout
            for _ in range(self.config.rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
                action, log_prob, value = self.policy.get_action(obs_tensor)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                raw_reward = reward

                # Normalize
                next_obs = self._normalize_obs(next_obs)
                norm_reward = self._normalize_reward(reward)

                self.buffer.add(Transition(
                    obs=obs, action=action, reward=norm_reward,
                    next_obs=next_obs, done=done,
                    log_prob=log_prob, value=value,
                ))

                episode_reward += norm_reward
                episode_raw_reward += raw_reward
                episode_length += 1
                timestep += 1
                self._timestep = timestep

                if done:
                    self.episode_rewards.append(episode_raw_reward)
                    self.episode_lengths.append(episode_length)
                    obs, info = self.env.reset()
                    obs = self._normalize_obs(obs)
                    episode_reward = 0.0
                    episode_raw_reward = 0.0
                    episode_length = 0
                else:
                    obs = next_obs

            # PPO update
            metrics = self._ppo_update()
            self.losses.append(metrics.to_dict())
            self.metrics_history.append(metrics)

            # Update LR scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.policy)

            # Periodic checkpoint
            if (self.config.save_freq > 0 and
                timestep % self.config.save_freq < self.config.rollout_steps):
                ckpt_path = os.path.join(
                    self.config.save_dir,
                    f"checkpoint_{timestep // 1000}k.pt"
                )
                os.makedirs(self.config.save_dir, exist_ok=True)
                self.save(ckpt_path)

            # Evaluate
            if timestep % self.config.eval_freq < self.config.rollout_steps:
                eval_reward = self._evaluate()
                self.eval_rewards.append(eval_reward)

                # EMA eval
                ema_reward = None
                if self.ema is not None:
                    ema_reward = self._evaluate(use_ema=True)
                    self.ema_eval_rewards.append(ema_reward)

                # Best model tracking and checkpoint
                eval_for_best = ema_reward if ema_reward is not None else eval_reward
                if eval_for_best > self.best_eval_reward:
                    self.best_eval_reward = eval_for_best
                    if self.config.save_best:
                        self.best_model_path = os.path.join(self.config.save_dir, "best_model.pt")
                        os.makedirs(self.config.save_dir, exist_ok=True)
                        self.save(self.best_model_path)

                elapsed = time.time() - start_time
                fps = timestep / elapsed if elapsed > 0 else 0

                # Build log line
                train_r = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
                lr_str = f"LR: {self.lr_scheduler.current_lr:.1e}" if self.lr_scheduler else ""
                ema_str = f"EMA: {ema_reward:>7.2f}" if ema_reward is not None else ""
                ent_str = f"H: {metrics.entropy:.2f}"
                kl_str = f"KL: {metrics.approx_kl:.4f}" if metrics.approx_kl > 0 else ""
                best_str = " *" if eval_for_best >= self.best_eval_reward and self.best_eval_reward > -float("inf") else ""

                parts = [
                    f"Step {timestep:>7d}",
                    f"Train R: {train_r:>7.2f}",
                    f"Eval R: {eval_reward:>7.2f}",
                ]
                if ema_str:
                    parts.append(ema_str)
                parts.append(f"FPS: {fps:.0f}")
                parts.append(ent_str)
                if kl_str:
                    parts.append(kl_str)
                if lr_str:
                    parts.append(lr_str)
                if best_str:
                    parts.append(best_str)

                print(" | ".join(parts))

        # Training summary
        total_time = time.time() - start_time
        final_train_r = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0.0
        final_eval_r = np.mean(self.eval_rewards[-5:]) if self.eval_rewards else 0.0

        print(f"\n{'═' * 60}")
        print(f"Training Complete: {timestep:,} steps in {total_time:.1f}s ({timestep/total_time:.0f} FPS)")
        print(f"  Final Train R (last 50 eps): {final_train_r:.2f}")
        print(f"  Final Eval R  (last 5 evals): {final_eval_r:.2f}")
        print(f"  Best Eval R: {self.best_eval_reward:.2f}")
        if self.best_model_path:
            print(f"  Best model: {self.best_model_path}")
        if self.ema_eval_rewards:
            print(f"  Final EMA Eval R: {np.mean(self.ema_eval_rewards[-5:]):.2f}")
        print(f"{'═' * 60}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "ema_eval_rewards": self.ema_eval_rewards,
            "best_eval_reward": self.best_eval_reward,
            "best_model_path": self.best_model_path,
            "total_timesteps": timestep,
            "training_time": total_time,
            "final_train_reward": final_train_r,
            "final_eval_reward": final_eval_r,
            "metrics_history": [m.to_dict() for m in self.metrics_history[-20:]],
        }

    def _ppo_update(self) -> TrainingMetrics:
        """Run PPO update with comprehensive diagnostics."""
        obs, actions, old_log_probs, returns, advantages = self.buffer.to_tensors(self.config.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_imitation_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0
        epoch_early_stopped = False

        imit_coeff = self._current_imitation_coeff()

        for epoch in range(self.config.n_epochs):
            if epoch_early_stopped:
                break

            # Mini-batch updates
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                if end > len(obs):
                    break
                idx = indices[start:end]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_adv = advantages[idx]

                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Policy loss (clipped surrogate)
                log_ratio = new_log_probs - batch_old_lp
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                ent_coeff = self._current_entropy_coeff()
                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    + ent_coeff * entropy_loss
                )

                # Imitation learning loss (behavioral cloning regularizer)
                if self.expert_buffer and imit_coeff > 0 and len(self.expert_buffer) > 0:
                    expert_obs, expert_actions = self.expert_buffer.sample(
                        min(self.config.batch_size, len(self.expert_buffer))
                    )
                    expert_obs = expert_obs.to(self.config.device)
                    expert_actions = expert_actions.to(self.config.device)
                    expert_lp, _, _ = self.policy.evaluate_actions(expert_obs, expert_actions)
                    imitation_loss = -expert_lp.mean()
                    loss = loss + imit_coeff * imitation_loss
                    total_imitation_loss += imitation_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                n_updates += 1

            # KL early stopping for this epoch
            if n_updates > 0 and self.config.target_kl > 0:
                avg_kl = total_approx_kl / n_updates
                if avg_kl > 1.5 * self.config.target_kl:
                    epoch_early_stopped = True

        self.buffer.clear()

        # Explained variance
        with torch.no_grad():
            y_pred = np.array([t.value for t in []])  # Already consumed
            # Use returns vs values from last batch as proxy
            ev = 0.0
            if n_updates > 0:
                ret_np = returns.cpu().numpy()
                val_np = np.array(self.buffer.values) if self.buffer.values else ret_np
                if len(ret_np) > 1:
                    var_ret = np.var(ret_np)
                    ev = 1.0 - np.var(ret_np - ret_np) / max(var_ret, 1e-8) if var_ret > 0 else 0.0

        metrics = TrainingMetrics(
            policy_loss=total_policy_loss / max(1, n_updates),
            value_loss=total_value_loss / max(1, n_updates),
            entropy=total_entropy / max(1, n_updates),
            approx_kl=total_approx_kl / max(1, n_updates),
            clip_fraction=total_clip_fraction / max(1, n_updates),
            imitation_loss=total_imitation_loss / max(1, n_updates) if total_imitation_loss > 0 else 0.0,
            imitation_coeff=imit_coeff,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            reward_mean=float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else 0.0,
            reward_std=float(np.std(self.episode_rewards[-20:])) if len(self.episode_rewards) > 1 else 0.0,
            episode_length_mean=float(np.mean(self.episode_lengths[-20:])) if self.episode_lengths else 0.0,
            n_updates=n_updates,
        )

        return metrics

    def _evaluate(self, use_ema: bool = False) -> float:
        """Evaluate current policy (or EMA policy)."""
        eval_policy = self.ema.ema_policy if (use_ema and self.ema) else self.policy
        rewards = []
        for _ in range(self.config.n_eval_episodes):
            obs, info = self.eval_env.reset()
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs).astype(np.float32)
            total_reward = 0.0
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
                with torch.no_grad():
                    logits, value = eval_policy(obs_tensor.unsqueeze(0))
                actions = []
                for head_logits in logits:
                    actions.append(head_logits.squeeze(0).argmax().item())
                obs, reward, terminated, truncated, info = self.eval_env.step(actions)
                if self.obs_rms is not None:
                    obs = self.obs_rms.normalize(obs).astype(np.float32)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def save(self, path: str):
        """Save comprehensive model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "episode_rewards": self.episode_rewards,
            "eval_rewards": self.eval_rewards,
            "ema_eval_rewards": self.ema_eval_rewards,
            "best_eval_reward": self.best_eval_reward,
            "timestep": self._timestep,
        }
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        if self.reward_rms is not None:
            checkpoint["reward_rms"] = {"mean": self.reward_rms.mean, "var": self.reward_rms.var, "count": self.reward_rms.count}
        if self.obs_rms is not None:
            checkpoint["obs_rms"] = {"mean": self.obs_rms.mean, "var": self.obs_rms.var, "count": self.obs_rms.count}
        torch.save(checkpoint, path)

        # Auto-register in model registry
        try:
            from .model_registry import register_model, ModelMetadata
            model_id = os.path.basename(path).replace(".pt", "")
            meta = ModelMetadata(
                model_id=model_id,
                file_path=os.path.basename(path),
                obs_dim=self.policy.shared[0].in_features,
                action_dims=[head[-1].out_features for head in self.policy.policy_heads],
                hidden_dim=self.policy.shared[0].out_features,
                best_eval_reward=self.best_eval_reward,
                training_config=self.config.__dict__,
                timesteps=self._timestep,
            )
            register_model(meta)
        except Exception:
            pass  # Registry is optional

    def load(self, path: str):
        """Load model checkpoint with full state restoration."""
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.ema_policy.load_state_dict(checkpoint["ema_state_dict"])
        if "reward_rms" in checkpoint and self.reward_rms is not None:
            rms = checkpoint["reward_rms"]
            self.reward_rms.mean = rms["mean"]
            self.reward_rms.var = rms["var"]
            self.reward_rms.count = rms["count"]
        if "obs_rms" in checkpoint and self.obs_rms is not None:
            rms = checkpoint["obs_rms"]
            self.obs_rms.mean = rms["mean"]
            self.obs_rms.var = rms["var"]
            self.obs_rms.count = rms["count"]
        if "timestep" in checkpoint:
            self._timestep = checkpoint["timestep"]
        if "best_eval_reward" in checkpoint:
            self.best_eval_reward = checkpoint["best_eval_reward"]
        if "episode_rewards" in checkpoint:
            self.episode_rewards = checkpoint["episode_rewards"]
        if "eval_rewards" in checkpoint:
            self.eval_rewards = checkpoint["eval_rewards"]


# ─── Random Agent (Baseline) ──────────────────────────────────────────────

class RandomAgent:
    """Random baseline agent for comparison."""

    def __init__(self, env: YieldBookEnv):
        self.env = env

    def evaluate(self, n_episodes: int = 100) -> dict:
        """Run random agent and collect stats."""
        rewards = []
        deal_arbs = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }


# ─── Heuristic Agent ──────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Heuristic agent that structures deals using industry conventions.

    Implements a simple PAC/Support structure with IO strip,
    typical of agency CMO issuance.
    """

    def __init__(self, env: YieldBookEnv):
        self.env = env

    def act(self, obs: np.ndarray, step: int) -> list[int]:
        """Select action based on heuristic rules."""
        at_collateral = 13  # offset ~0 from collateral coupon

        if step == 0:
            return [1, 0, 16, at_collateral]  # ADD_PAC, ~44% size
        if step == 1:
            return [3, 1, 11, at_collateral]  # ADD_SUPPORT, ~28%
        if step == 2:
            return [0, 2, 7, at_collateral]   # ADD_SEQ, ~18%
        return [12, 0, 0, 0]  # EXECUTE_DEAL

    def evaluate(self, n_episodes: int = 100) -> dict:
        """Run heuristic agent and collect stats."""
        rewards = []
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            while not done:
                action = self.act(obs, step)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
            rewards.append(total_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }


class GSEShelfAgent:
    """
    Expert GSE shelf structuring agent that maximizes dealer profit.

    Implements full kitchen-sink REMIC deal structures:
    PAC + Support + IO strip + Z-bond + Floater/Inverse Floater pair.

    This is how top dealers (JPM, Goldman, Citi) structure Ginnie Mae REMICs
    to extract maximum structuring arbitrage:
    - PAC trades tight (low spread) -> high price
    - IO strip: excess coupon sold to money managers
    - Inverse floater: sold to hedge funds (Nick Brown/Millennium style)
    - Z-bond: sold to insurance companies seeking duration
    - Support: absorbs prepay volatility, priced wider but still profitable
    """

    STRATEGIES = [
        "pac_io_z",         # PAC + Support + IO + Z-bond (most common GSE shelf)
        "floater_inverse",  # PAC + Floater/Inv Floater pair + Support
        "kitchen_sink",     # PAC + Support + IO + Z + Floater/Inv (max arb)
        "pac_support",      # Simple PAC/Support (baseline)
        "sequential_io",    # Sequential + IO strip
    ]

    def __init__(self, env: YieldBookEnv, strategy: str = "kitchen_sink"):
        self.env = env
        self.strategy = strategy

    def act(self, obs: np.ndarray, step: int) -> list[int]:
        """Select action based on GSE shelf strategy."""
        at_collateral = 13   # offset ~0 from collateral coupon
        below_collateral = 10  # offset ~-0.95% (tighter coupon for PAC)
        above_collateral = 16  # offset ~+0.47% (wider for support)

        if self.strategy == "pac_io_z":
            return self._pac_io_z(step, at_collateral, below_collateral, above_collateral)
        elif self.strategy == "floater_inverse":
            return self._floater_inverse(step, at_collateral, below_collateral, above_collateral)
        elif self.strategy == "kitchen_sink":
            return self._kitchen_sink(step, at_collateral, below_collateral, above_collateral)
        elif self.strategy == "sequential_io":
            return self._sequential_io(step, at_collateral, below_collateral)
        else:  # pac_support
            return self._pac_support(step, at_collateral)

    def _pac_io_z(self, step, at, below, above):
        """PAC + Support + IO + Z-bond. Most common GSE shelf deal."""
        if step == 0: return [1, 0, 15, below]   # PAC 40%
        if step == 1: return [3, 1, 10, above]   # Support 25%
        if step == 2: return [4, 2, 7, at]       # Z-bond 18%
        if step == 3: return [5, 3, 12, at]      # IO strip (notional, doesn't use balance)
        if step == 4: return [0, 4, 5, at]       # SEQ remainder ~12%
        return [12, 0, 0, 0]

    def _floater_inverse(self, step, at, below, above):
        """PAC + Floater/Inverse Floater pair + Support."""
        if step == 0: return [1, 0, 14, below]    # PAC 35%
        if step == 1: return [7, 1, 8, at]        # Floater 20%
        if step == 2: return [8, 2, 5, above]     # Inv Floater 12%
        if step == 3: return [3, 3, 8, above]     # Support 20%
        if step == 4: return [0, 4, 4, at]        # SEQ remainder ~10%
        return [12, 0, 0, 0]

    def _kitchen_sink(self, step, at, below, above):
        """Full kitchen sink: PAC + Support + IO + Z + Floater/Inv. Maximum arb."""
        if step == 0: return [1, 0, 11, below]    # PAC 28%
        if step == 1: return [7, 1, 8, at]        # Floater 20%
        if step == 2: return [8, 2, 5, above]     # Inv Floater 12%
        if step == 3: return [3, 3, 7, above]     # Support 18%
        if step == 4: return [4, 4, 5, at]        # Z-bond 12%
        if step == 5: return [5, 5, 11, at]       # IO strip (notional)
        if step == 6: return [0, 6, 3, at]        # SEQ remainder ~8%
        return [12, 0, 0, 0]

    def _sequential_io(self, step, at, below):
        """Sequential tranches + IO strip."""
        if step == 0: return [0, 0, 10, below]    # SEQ-A 25%
        if step == 1: return [0, 1, 10, at]       # SEQ-B 25%
        if step == 2: return [0, 2, 10, at]       # SEQ-C 25%
        if step == 3: return [5, 3, 15, at]       # IO strip
        if step == 4: return [0, 4, 7, at]        # SEQ-D remainder
        return [12, 0, 0, 0]

    def _pac_support(self, step, at):
        """Simple PAC/Support (baseline)."""
        if step == 0: return [1, 0, 16, at]       # PAC 44%
        if step == 1: return [3, 1, 11, at]       # Support 28%
        if step == 2: return [0, 2, 7, at]        # SEQ 18%
        return [12, 0, 0, 0]

    def evaluate(self, n_episodes: int = 100) -> dict:
        """Run GSE shelf agent and collect stats."""
        rewards = []
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            while not done:
                action = self.act(obs, step)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
            rewards.append(total_reward)

        return {
            "strategy": self.strategy,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }

    @classmethod
    def evaluate_all_strategies(cls, env: YieldBookEnv, n_episodes: int = 50) -> dict:
        """Evaluate all GSE shelf strategies and find the best."""
        results = {}
        for strategy in cls.STRATEGIES:
            agent = cls(env, strategy=strategy)
            result = agent.evaluate(n_episodes)
            results[strategy] = result
            print(f"  {strategy:20s}: mean={result['mean_reward']:>7.2f} "
                  f"std={result['std_reward']:>6.2f} "
                  f"max={result['max_reward']:>7.2f}")
        return results


# ─── Training Entry Points ────────────────────────────────────────────────

def train_ppo(
    mode: str = "AGENCY",
    total_timesteps: int = 100_000,
    seed: int = 42,
    device: str = "cpu",
    save_path: Optional[str] = None,
    data_provider=None,
) -> dict:
    """Train a PPO agent on the Yield Book environment."""
    env = make_yield_book_env(mode=mode, seed=seed, data_provider=data_provider)
    eval_env = make_yield_book_env(mode=mode, seed=seed + 1000, data_provider=data_provider)

    config = PPOConfig(
        total_timesteps=total_timesteps,
        device=device,
    )

    trainer = PPOTrainer(env, config, eval_env)
    results = trainer.train()

    if save_path:
        trainer.save(save_path)

    return results


def benchmark_agents(
    mode: str = "AGENCY",
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """Compare all agents: random, heuristic, and GSE shelf strategies."""
    env = make_yield_book_env(mode=mode, seed=seed)

    print("=== Agent Benchmark ===\n")

    print("Random Agent:")
    random_results = RandomAgent(env).evaluate(n_episodes)
    print(f"  Mean: {random_results['mean_reward']:.2f} | "
          f"Std: {random_results['std_reward']:.2f}\n")

    print("Simple Heuristic (PAC/Support):")
    heuristic_results = HeuristicAgent(env).evaluate(n_episodes)
    print(f"  Mean: {heuristic_results['mean_reward']:.2f} | "
          f"Std: {heuristic_results['std_reward']:.2f}\n")

    print("GSE Shelf Strategies:")
    gse_results = GSEShelfAgent.evaluate_all_strategies(env, n_episodes)

    best_strategy = max(gse_results.items(), key=lambda x: x[1]["mean_reward"])
    print(f"\nBest strategy: {best_strategy[0]} "
          f"(mean={best_strategy[1]['mean_reward']:.2f})")

    return {
        "random": random_results,
        "heuristic": heuristic_results,
        "gse_strategies": gse_results,
        "best_gse_strategy": best_strategy[0],
    }


def train_with_imitation(
    demo_file: str = "expert_demonstrations.json",
    total_timesteps: int = 300_000,
    imitation_coeff: float = 0.5,
    imitation_anneal_steps: int = 100_000,
    pretrain_epochs: int = 50,
    seed: int = 42,
    save_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    data_provider=None,
) -> dict:
    """Train with imitation learning from expert dealer demonstrations.

    Two-phase approach:
    1. Behavioral Cloning (BC) pre-training: supervised learning on expert (obs, action) pairs
    2. PPO fine-tuning with annealing imitation regularizer

    Args:
        demo_file: Path to expert_demonstrations.json (742 real REMIC deals)
        total_timesteps: Total PPO timesteps for phase 2
        imitation_coeff: Initial weight for imitation loss in PPO phase
        imitation_anneal_steps: Steps to anneal imitation loss to 0
        pretrain_epochs: Number of BC pre-training epochs
        seed: Random seed
        save_path: Where to save the trained model
        resume_path: Resume from checkpoint
    """
    env = make_yield_book_env(mode="AGENCY", seed=seed, data_provider=data_provider)
    eval_env = make_yield_book_env(mode="AGENCY", seed=seed + 1000, data_provider=data_provider)

    # Phase 1: Load expert demos and build buffer
    print(f"Phase 1: Loading expert demonstrations from {demo_file}")
    expert_buffer = ExpertDemoBuffer(demo_file, env, max_demos=0)
    if len(expert_buffer) == 0:
        print("  WARNING: No expert demos loaded, skipping imitation")
        return train_gse_shelf_agent(total_timesteps=total_timesteps, seed=seed, save_path=save_path, data_provider=data_provider)

    # Create policy and do BC pre-training
    obs_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)
    policy = PolicyNetwork(obs_dim, action_dims, 256).to("cpu")
    bc_optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    print(f"  BC pre-training for {pretrain_epochs} epochs on {len(expert_buffer)} pairs...")
    bc_batch_size = min(256, len(expert_buffer))
    for epoch in range(pretrain_epochs):
        expert_obs, expert_actions = expert_buffer.sample(bc_batch_size)
        log_probs, _, _ = policy.evaluate_actions(expert_obs, expert_actions)
        bc_loss = -log_probs.mean()

        bc_optimizer.zero_grad()
        bc_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        bc_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{pretrain_epochs}: BC loss = {bc_loss.item():.4f}")

    # Phase 2: PPO fine-tuning with imitation regularizer
    print(f"\nPhase 2: PPO fine-tuning ({total_timesteps:,} steps, imit_coeff={imitation_coeff})")
    config = PPOConfig(
        total_timesteps=total_timesteps,
        entropy_coeff=0.03,
        entropy_min=0.002,
        entropy_anneal_steps=total_timesteps // 2,
        rollout_steps=512,
        lr=2e-4,
        hidden_dim=256,
        n_epochs=6,
        batch_size=128,
        imitation_coeff=imitation_coeff,
        imitation_anneal_steps=imitation_anneal_steps,
    )

    trainer = PPOTrainer(env, config, eval_env, expert_buffer=expert_buffer)

    # Transfer BC pre-trained weights
    trainer.policy.load_state_dict(policy.state_dict())
    # Reset optimizer for PPO phase
    trainer.optimizer = optim.Adam(trainer.policy.parameters(), lr=config.lr)

    if resume_path:
        trainer.load(resume_path)
        print(f"  Resumed from {resume_path}")

    results = trainer.train()

    if save_path:
        trainer.save(save_path)
        print(f"Model saved to {save_path}")

    return results


def train_gse_shelf_agent(
    total_timesteps: int = 200_000,
    seed: int = 42,
    save_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    data_provider=None,
) -> dict:
    """
    Train a GSE shelf-focused agent.

    Uses higher entropy for exploration, longer rollouts,
    and targets the reward structure that favors IO strips,
    floater/inv floater pairs, and full kitchen-sink deals.
    """
    env = make_yield_book_env(mode="AGENCY", seed=seed, data_provider=data_provider)
    eval_env = make_yield_book_env(mode="AGENCY", seed=seed + 1000, data_provider=data_provider)

    config = PPOConfig(
        total_timesteps=total_timesteps,
        entropy_coeff=0.03,            # More exploration
        entropy_min=0.002,
        entropy_anneal_steps=total_timesteps // 2,
        rollout_steps=512,             # Longer rollouts
        lr=2e-4,                       # Slightly lower LR for stability
        hidden_dim=256,
        n_epochs=6,                    # More update epochs
        batch_size=128,
    )

    trainer = PPOTrainer(env, config, eval_env)

    if resume_path:
        trainer.load(resume_path)
        print(f"Resumed from {resume_path}")

    print(f"Training GSE shelf agent for {total_timesteps:,} steps...")
    results = trainer.train()

    if save_path:
        trainer.save(save_path)
        print(f"Model saved to {save_path}")

    return results


class FullDeskAgent:
    """
    Full-desk heuristic agent that exercises all 4 P&L components.

    Uses SELECT_POOL to choose spec pools, then structures kitchen-sink
    deals to maximize total desk P&L:
      1. Structuring arbitrage
      2. IO strip value
      3. Dollar roll income (Song & Zhu specialness)
      4. Spec pool payup income

    Cycles through pool types and deal structures to find the best
    combination for each market scenario.
    """

    POOL_STRATEGIES = [
        (1, "kitchen_sink"),   # LLB + kitchen sink (highest payup)
        (2, "kitchen_sink"),   # HLTV + kitchen sink
        (4, "kitchen_sink"),   # LFICO + kitchen sink
        (3, "pac_io_z"),       # NY + PAC/IO/Z
        (0, "kitchen_sink"),   # TBA + kitchen sink (baseline)
        (1, "floater_inverse"),# LLB + floater/inverse
    ]

    def __init__(self, env: YieldBookEnv, pool_type: int = 1, strategy: str = "kitchen_sink"):
        self.env = env
        self.pool_type = pool_type
        self.strategy = strategy

    def act(self, obs: np.ndarray, step: int) -> list[int]:
        """Select action: first select pool, then structure deal."""
        at_collateral = 13
        below_collateral = 10
        above_collateral = 16

        # Step 0: SELECT_POOL (action_type=14, tranche_idx=pool_type)
        if step == 0:
            return [14, self.pool_type, 0, 0]

        # Remaining steps: structure the deal (offset by 1 from GSEShelfAgent)
        deal_step = step - 1
        if self.strategy == "kitchen_sink":
            return self._kitchen_sink(deal_step, at_collateral, below_collateral, above_collateral)
        elif self.strategy == "floater_inverse":
            return self._floater_inverse(deal_step, at_collateral, below_collateral, above_collateral)
        else:
            return self._pac_io_z(deal_step, at_collateral, below_collateral, above_collateral)

    def _kitchen_sink(self, step, at, below, above):
        if step == 0: return [1, 0, 11, below]    # PAC 28%
        if step == 1: return [7, 1, 8, at]         # Floater 20%
        if step == 2: return [8, 2, 5, above]      # Inv Floater 12%
        if step == 3: return [3, 3, 7, above]      # Support 18%
        if step == 4: return [4, 4, 5, at]          # Z-bond 12%
        if step == 5: return [5, 5, 11, at]         # IO strip
        if step == 6: return [0, 6, 3, at]          # SEQ remainder
        return [12, 0, 0, 0]

    def _floater_inverse(self, step, at, below, above):
        if step == 0: return [1, 0, 14, below]     # PAC 35%
        if step == 1: return [7, 1, 8, at]          # Floater 20%
        if step == 2: return [8, 2, 5, above]       # Inv Floater 12%
        if step == 3: return [3, 3, 8, above]       # Support 20%
        if step == 4: return [0, 4, 4, at]           # SEQ 10%
        return [12, 0, 0, 0]

    def _pac_io_z(self, step, at, below, above):
        if step == 0: return [1, 0, 15, below]     # PAC 40%
        if step == 1: return [3, 1, 10, above]     # Support 25%
        if step == 2: return [4, 2, 7, at]          # Z-bond 18%
        if step == 3: return [5, 3, 12, at]         # IO strip
        if step == 4: return [0, 4, 5, at]           # SEQ 12%
        return [12, 0, 0, 0]

    def evaluate(self, n_episodes: int = 100) -> dict:
        rewards = []
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            while not done:
                action = self.act(obs, step)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
            rewards.append(total_reward)

        return {
            "pool_type": self.pool_type,
            "strategy": self.strategy,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }

    @classmethod
    def evaluate_all_strategies(cls, env: YieldBookEnv, n_episodes: int = 50) -> dict:
        results = {}
        for pool_type, strategy in cls.POOL_STRATEGIES:
            key = f"pool{pool_type}_{strategy}"
            agent = cls(env, pool_type=pool_type, strategy=strategy)
            result = agent.evaluate(n_episodes)
            results[key] = result
            pool_names = ["TBA", "LLB", "HLTV", "NY", "LFICO", "INV", "GEO"]
            pool_name = pool_names[pool_type] if pool_type < len(pool_names) else f"P{pool_type}"
            print(f"  {pool_name:5s}+{strategy:20s}: mean={result['mean_reward']:>7.2f} "
                  f"std={result['std_reward']:>6.2f} "
                  f"max={result['max_reward']:>7.2f}")
        return results


def train_full_desk_agent(
    total_timesteps: int = 500_000,
    demo_file: str = "expert_demonstrations.json",
    imitation_coeff: float = 0.5,
    imitation_anneal_steps: int = 200_000,
    pretrain_epochs: int = 100,
    seed: int = 42,
    save_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    data_provider=None,
    use_curriculum: bool = False,
    analyze: bool = True,
) -> dict:
    """
    Train full-desk CMO agent with 4-component P&L model.

    This is the main training entry point for the overhauled reward
    function that includes:
      1. Structuring arbitrage (traditional CMO spread)
      2. IO strip value (excess coupon stripping)
      3. Dollar roll income (Song & Zhu 2019 specialness model)
      4. Spec pool payup income (Bednarek et al. 2023)

    Enhancements over v0.7:
      - Cosine LR scheduling with warmup
      - Running reward normalization
      - EMA policy for stable evaluation
      - Best-model auto-checkpointing
      - KL-based epoch early stopping
      - Rich diagnostics (clip fraction, KL, entropy)
      - Optional curriculum learning
      - Post-training deal structure analysis
    """
    env = make_yield_book_env(mode="AGENCY", seed=seed, data_provider=data_provider)
    eval_env = make_yield_book_env(mode="AGENCY", seed=seed + 1000, data_provider=data_provider)

    obs_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)
    hidden = 384  # wider network for richer reward signal

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║           FULL DESK AGENT TRAINING v0.9             ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"  Obs dim: {obs_dim}, Action dims: {action_dims}")
    print(f"  Network: {hidden}-dim hidden, 3 layers (LayerNorm)")
    print(f"  Data provider: {'real market data' if data_provider else 'synthetic'}")
    print(f"  Curriculum: {'enabled' if use_curriculum else 'disabled'}")
    print(f"  LR schedule: cosine w/ warmup")
    print(f"  EMA policy: decay=0.995")
    print(f"  Reward normalization: enabled")
    print()

    # Phase 1: Expert demo pre-training
    expert_buffer = None
    if os.path.exists(demo_file):
        print(f"Phase 1: Loading expert demonstrations from {demo_file}")
        expert_buffer = ExpertDemoBuffer(demo_file, env, max_demos=0)
        if len(expert_buffer) == 0:
            print("  WARNING: No expert demos loaded, skipping imitation")
            expert_buffer = None

    policy = PolicyNetwork(obs_dim, action_dims, hidden).to("cpu")

    if expert_buffer and len(expert_buffer) > 0:
        bc_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        print(f"  BC pre-training for {pretrain_epochs} epochs on {len(expert_buffer)} pairs...")
        bc_batch_size = min(256, len(expert_buffer))
        best_bc_loss = float("inf")
        for epoch in range(pretrain_epochs):
            expert_obs, expert_actions = expert_buffer.sample(bc_batch_size)
            log_probs, _, _ = policy.evaluate_actions(expert_obs, expert_actions)
            bc_loss = -log_probs.mean()

            bc_optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            bc_optimizer.step()

            if bc_loss.item() < best_bc_loss:
                best_bc_loss = bc_loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{pretrain_epochs}: BC loss = {bc_loss.item():.4f} (best={best_bc_loss:.4f})")
        print()

    # Phase 2: PPO fine-tuning with all enhancements
    print(f"Phase 2: PPO fine-tuning ({total_timesteps:,} steps)")
    config = PPOConfig(
        total_timesteps=total_timesteps,
        entropy_coeff=0.04,             # Higher entropy for wider action space
        entropy_min=0.003,
        entropy_anneal_steps=total_timesteps // 2,
        rollout_steps=1024,             # Longer rollouts for full-desk episodes
        lr=1.5e-4,                      # Lower LR for stability with wider rewards
        lr_min=1e-6,
        hidden_dim=hidden,
        n_epochs=6,
        batch_size=128,
        eval_freq=10_000,
        n_eval_episodes=20,
        imitation_coeff=imitation_coeff if expert_buffer else 0.0,
        imitation_anneal_steps=imitation_anneal_steps,
        # New features
        use_lr_schedule=True,
        warmup_steps=min(10_000, total_timesteps // 20),
        normalize_rewards=True,
        normalize_obs=False,
        save_best=True,
        save_freq=100_000,
        save_dir="models",
        use_ema=True,
        ema_decay=0.995,
        target_kl=0.03,
        use_curriculum=use_curriculum,
    )

    trainer = PPOTrainer(env, config, eval_env, expert_buffer=expert_buffer)

    # Transfer BC pre-trained weights
    trainer.policy.load_state_dict(policy.state_dict())
    trainer.optimizer = optim.Adam(trainer.policy.parameters(), lr=config.lr)
    # Recreate LR scheduler after optimizer reset
    if config.use_lr_schedule:
        trainer.lr_scheduler = CosineSchedule(
            trainer.optimizer,
            total_steps=config.total_timesteps // config.rollout_steps,
            warmup_steps=config.warmup_steps // config.rollout_steps,
            lr_min=config.lr_min,
        )
    # Recreate EMA after weight transfer
    if config.use_ema:
        trainer.ema = EMAPolicy(trainer.policy, decay=config.ema_decay)

    if resume_path:
        trainer.load(resume_path)
        print(f"  Resumed from {resume_path}")

    results = trainer.train()

    if save_path:
        trainer.save(save_path)
        print(f"Model saved to {save_path}")

    # Post-training benchmark
    print("\n=== Post-Training Benchmark ===")
    print("\nFull Desk Heuristic Strategies:")
    FullDeskAgent.evaluate_all_strategies(eval_env, n_episodes=50)

    # Deal structure analysis
    if analyze:
        analyzer = DealAnalyzer(trainer.policy, eval_env)
        analysis = analyzer.analyze(n_episodes=200)
        analyzer.print_analysis(analysis)

        # EMA policy analysis too
        if trainer.ema is not None:
            print("\n--- EMA Policy Analysis ---")
            ema_analyzer = DealAnalyzer(trainer.ema.ema_policy, eval_env)
            ema_analysis = ema_analyzer.analyze(n_episodes=200)
            ema_analyzer.print_analysis(ema_analysis)

        results["deal_analysis"] = analysis

    return results


def train_cmbs_agent(
    total_timesteps: int = 300_000,
    seed: int = 42,
    save_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    data_provider=None,
) -> dict:
    """
    Train a CMBS structuring agent.

    CMBS deals have fundamentally different economics:
    - Credit subordination (senior/mezz/B-piece) instead of prepay-based tranching
    - Special servicing triggers (DSCR < 1.0, LTV > 80%)
    - B-piece buyer dynamics (Rialto, Elliot, Torchlight)
    - Property type diversification requirements

    Uses 185-dim observation space (175 base + 10 CMBS).
    """
    env = make_yield_book_env(mode="CMBS", seed=seed)
    eval_env = make_yield_book_env(mode="CMBS", seed=seed + 1000)

    obs_dim = env.observation_space.shape[0]
    action_dims = list(env.action_space.nvec)
    hidden = 384

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║              CMBS AGENT TRAINING                    ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"  Obs dim: {obs_dim}, Action dims: {action_dims}")
    print(f"  Network: {hidden}-dim hidden, 3 layers (LayerNorm)")
    print()

    config = PPOConfig(
        total_timesteps=total_timesteps,
        entropy_coeff=0.05,             # Even more exploration for complex CMBS
        entropy_min=0.005,
        entropy_anneal_steps=total_timesteps // 2,
        rollout_steps=1024,
        lr=1.5e-4,
        lr_min=1e-6,
        hidden_dim=hidden,
        n_epochs=6,
        batch_size=128,
        eval_freq=10_000,
        n_eval_episodes=20,
        # Enhancements
        use_lr_schedule=True,
        warmup_steps=min(10_000, total_timesteps // 20),
        normalize_rewards=True,
        save_best=True,
        save_dir="models",
        use_ema=True,
        target_kl=0.03,
    )

    trainer = PPOTrainer(env, config, eval_env)

    if resume_path:
        trainer.load(resume_path)
        print(f"  Resumed from {resume_path}")

    results = trainer.train()

    if save_path:
        trainer.save(save_path)
        print(f"Model saved to {save_path}")

    # CMBS deal analysis
    analyzer = DealAnalyzer(trainer.policy, eval_env)
    analysis = analyzer.analyze(n_episodes=100)
    analyzer.print_analysis(analysis)
    results["deal_analysis"] = analysis

    return results


def train_multi_desk(
    total_timesteps_per_desk: int = 200_000,
    seed: int = 42,
    save_dir: str = "models/ficc",
    data_provider=None,
) -> dict:
    """
    Train agents across multiple FICC desks sequentially.

    Trains separate PPO policies for:
    1. MBS/CMO (Agency) - full desk agent
    2. CMBS - credit subordination
    3. Treasuries, Rates, etc. (if environments available)

    Each desk gets its own policy, checkpoint, and analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║           MULTI-DESK FICC TRAINING                  ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"  Steps per desk: {total_timesteps_per_desk:,}")
    print(f"  Save dir: {save_dir}")
    print()

    # Desk 1: Agency MBS/CMO (full desk)
    print("━" * 60)
    print("DESK 1/3: Agency MBS/CMO")
    print("━" * 60)
    agency_results = train_full_desk_agent(
        total_timesteps=total_timesteps_per_desk,
        seed=seed,
        save_path=os.path.join(save_dir, "agency_desk.pt"),
        data_provider=data_provider,
        analyze=True,
    )
    all_results["agency"] = {
        "final_train_reward": agency_results.get("final_train_reward", 0),
        "final_eval_reward": agency_results.get("final_eval_reward", 0),
        "best_eval_reward": agency_results.get("best_eval_reward", 0),
    }

    # Desk 2: CMBS
    print("\n" + "━" * 60)
    print("DESK 2/3: CMBS")
    print("━" * 60)
    cmbs_results = train_cmbs_agent(
        total_timesteps=total_timesteps_per_desk,
        seed=seed + 100,
        save_path=os.path.join(save_dir, "cmbs_desk.pt"),
    )
    all_results["cmbs"] = {
        "final_train_reward": cmbs_results.get("final_train_reward", 0),
        "final_eval_reward": cmbs_results.get("final_eval_reward", 0),
        "best_eval_reward": cmbs_results.get("best_eval_reward", 0),
    }

    # Desk 3: GSE shelf (different structuring approach)
    print("\n" + "━" * 60)
    print("DESK 3/3: GSE Shelf")
    print("━" * 60)
    gse_results = train_gse_shelf_agent(
        total_timesteps=total_timesteps_per_desk,
        seed=seed + 200,
        save_path=os.path.join(save_dir, "gse_shelf.pt"),
        data_provider=data_provider,
    )
    all_results["gse_shelf"] = {
        "final_train_reward": gse_results.get("final_train_reward", 0),
        "final_eval_reward": gse_results.get("final_eval_reward", 0),
    }

    # Summary
    print(f"\n{'═' * 60}")
    print(f"MULTI-DESK TRAINING SUMMARY")
    print(f"{'═' * 60}")
    for desk, metrics in all_results.items():
        print(f"  {desk:12s}: eval={metrics.get('final_eval_reward', 0):>7.2f}  "
              f"best={metrics.get('best_eval_reward', metrics.get('final_eval_reward', 0)):>7.2f}")

    # Save summary
    summary_path = os.path.join(save_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return all_results


def train_with_curriculum(
    total_timesteps: int = 500_000,
    demo_file: str = "expert_demonstrations.json",
    seed: int = 42,
    save_path: Optional[str] = None,
    data_provider=None,
) -> dict:
    """
    Train full-desk agent with curriculum learning.

    Regime difficulty ramps over training:
    - 0-20%: mostly normal/steep (easy) regimes
    - 20-50%: add inverted/volatile
    - 50-80%: add crisis regimes
    - 80-100%: full difficulty distribution

    This prevents the agent from being overwhelmed by crisis scenarios
    early in training when it hasn't learned basic structuring yet.
    """
    return train_full_desk_agent(
        total_timesteps=total_timesteps,
        demo_file=demo_file,
        seed=seed,
        save_path=save_path,
        data_provider=data_provider,
        use_curriculum=True,
        analyze=True,
    )


def ablation_study(
    total_timesteps: int = 100_000,
    seed: int = 42,
    data_provider=None,
) -> dict:
    """
    Run ablation study comparing training configurations.

    Tests impact of:
    1. Reward normalization
    2. LR scheduling
    3. EMA policy
    4. Expert demonstrations
    5. Curriculum learning

    Useful for hyperparameter tuning and understanding which
    components contribute most to performance.
    """
    env = make_yield_book_env(mode="AGENCY", seed=seed, data_provider=data_provider)
    eval_env = make_yield_book_env(mode="AGENCY", seed=seed + 1000, data_provider=data_provider)

    configs = {
        "baseline": PPOConfig(
            total_timesteps=total_timesteps,
            hidden_dim=384,
            use_lr_schedule=False,
            normalize_rewards=False,
            use_ema=False,
            save_best=False,
        ),
        "+reward_norm": PPOConfig(
            total_timesteps=total_timesteps,
            hidden_dim=384,
            use_lr_schedule=False,
            normalize_rewards=True,
            use_ema=False,
            save_best=False,
        ),
        "+lr_sched": PPOConfig(
            total_timesteps=total_timesteps,
            hidden_dim=384,
            use_lr_schedule=True,
            normalize_rewards=True,
            use_ema=False,
            save_best=False,
        ),
        "+ema": PPOConfig(
            total_timesteps=total_timesteps,
            hidden_dim=384,
            use_lr_schedule=True,
            normalize_rewards=True,
            use_ema=True,
            save_best=False,
        ),
    }

    results = {}
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║             ABLATION STUDY ({total_timesteps:,} steps)        ║")
    print(f"╚══════════════════════════════════════════════════════╝\n")

    for name, config in configs.items():
        print(f"\n--- Ablation: {name} ---")
        trainer = PPOTrainer(env, config, eval_env)
        run_results = trainer.train()
        final_eval = np.mean(trainer.eval_rewards[-5:]) if trainer.eval_rewards else 0.0
        results[name] = {
            "final_eval": float(final_eval),
            "best_eval": float(max(trainer.eval_rewards)) if trainer.eval_rewards else 0.0,
            "final_train": float(np.mean(trainer.episode_rewards[-50:])) if trainer.episode_rewards else 0.0,
        }
        print(f"  Result: eval={final_eval:.2f}")

    print(f"\n{'═' * 60}")
    print(f"ABLATION RESULTS")
    print(f"{'═' * 60}")
    for name, r in results.items():
        print(f"  {name:20s}: eval={r['final_eval']:>7.2f}  best={r['best_eval']:>7.2f}  train={r['final_train']:>7.2f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train CMO structuring agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training modes:
  AGENCY    Basic agency CMO structuring
  CMBS      Commercial MBS with credit subordination
  GSE       GSE shelf-focused (higher entropy, IO/floater focus)
  DESK      Full desk with 4-component P&L (recommended)
  MULTI     Train all desks sequentially (Agency + CMBS + GSE)
  CURRICULUM Full desk with curriculum learning
  ABLATION  Run ablation study across configurations

Examples:
  # Full desk with real data, 1M steps, auto-save best model
  python -m cmo_agent.train --mode DESK --real-data --build-db --timesteps 1000000

  # Resume training from checkpoint
  python -m cmo_agent.train --mode DESK --resume models/best_model.pt --timesteps 500000

  # CMBS-specific training
  python -m cmo_agent.train --mode CMBS --timesteps 300000 --save models/cmbs_300k.pt

  # Multi-desk training
  python -m cmo_agent.train --mode MULTI --timesteps 200000 --real-data

  # Ablation study
  python -m cmo_agent.train --mode ABLATION --timesteps 100000
        """,
    )
    parser.add_argument("--mode", choices=[
        "AGENCY", "CMBS", "GSE", "DESK", "MULTI", "CURRICULUM", "ABLATION",
    ], default="AGENCY")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--demos", default="expert_demonstrations.json")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run agent benchmarks (random, heuristic, GSE shelf)")
    parser.add_argument("--analyze", action="store_true",
                        help="Run post-training deal structure analysis")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real historical market data (requires built database)")
    parser.add_argument("--build-db", action="store_true",
                        help="Build/update the market history database before training")
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable EMA policy")
    parser.add_argument("--no-lr-schedule", action="store_true",
                        help="Disable cosine LR scheduling")
    parser.add_argument("--no-reward-norm", action="store_true",
                        help="Disable reward normalization")
    args = parser.parse_args()

    # Set up real data provider if requested
    dp = None
    if args.real_data or args.build_db:
        from .real_market_data import RealMarketDataProvider
        dp = RealMarketDataProvider(seed=args.seed)
        if args.build_db:
            dp.build_database()
        if dp.n_dates == 0:
            print("WARNING: No data in database. Run with --build-db first. Falling back to synthetic.")
            dp = None

    if args.benchmark:
        benchmark_agents(mode="AGENCY" if args.mode in ("GSE", "DESK", "MULTI", "CURRICULUM") else args.mode, seed=args.seed)
    elif args.mode == "DESK":
        train_full_desk_agent(
            total_timesteps=args.timesteps,
            demo_file=args.demos,
            seed=args.seed,
            save_path=args.save,
            resume_path=args.resume,
            data_provider=dp,
            analyze=args.analyze or True,
        )
    elif args.mode == "CMBS":
        train_cmbs_agent(
            total_timesteps=args.timesteps,
            seed=args.seed,
            save_path=args.save,
            resume_path=args.resume,
        )
    elif args.mode == "GSE":
        train_gse_shelf_agent(
            total_timesteps=args.timesteps,
            seed=args.seed,
            save_path=args.save,
            resume_path=args.resume,
            data_provider=dp,
        )
    elif args.mode == "MULTI":
        train_multi_desk(
            total_timesteps_per_desk=args.timesteps,
            seed=args.seed,
            data_provider=dp,
        )
    elif args.mode == "CURRICULUM":
        train_with_curriculum(
            total_timesteps=args.timesteps,
            demo_file=args.demos,
            seed=args.seed,
            save_path=args.save,
            data_provider=dp,
        )
    elif args.mode == "ABLATION":
        ablation_study(
            total_timesteps=args.timesteps,
            seed=args.seed,
            data_provider=dp,
        )
    else:
        train_ppo(
            mode=args.mode,
            total_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
            save_path=args.save,
            data_provider=dp,
        )
