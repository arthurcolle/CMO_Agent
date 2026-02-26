"""
Unified Multi-Desk Training Pipeline for Investment Bank RL Environments.

Trains PPO agents across all 17 desk environments:
  FICC (9):  MBS/CMO, Treasuries, Distressed, Rates, IG Credit, Munis, Repo, FX, Commodities
  Equities (3): Cash, Derivatives, Prime Brokerage
  IBD (3): ECM/IPO, DCM/LevFin, M&A Advisory
  Meta (2): FICC Floor, Investment Bank

Supports:
  - Individual desk training
  - Parallel multi-desk training
  - Curriculum learning (desks -> floor -> bank)
  - Benchmarking across all desks
  - Model checkpointing and resume

References:
  - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
  - Bengio et al. (2009) "Curriculum Learning" ICML
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Optional
import time
import json
import os

# ─── Desk Registry ────────────────────────────────────────────────────────

def _make_env(module_name: str, factory_name: str, **kwargs):
    """Lazy import and create an environment."""
    import importlib
    mod = importlib.import_module(f"cmo_agent.{module_name}")
    factory = getattr(mod, factory_name)
    return factory(**kwargs)


@dataclass
class DeskConfig:
    """Per-desk training hyperparameters."""
    name: str
    module: str
    factory: str
    # Training
    total_timesteps: int = 200_000
    hidden_dim: int = 256
    lr: float = 2e-4
    entropy_coeff: float = 0.03
    rollout_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 4
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    # Reward scale (for normalizing across desks)
    reward_scale: float = 1.0


DESK_REGISTRY: dict[str, DeskConfig] = {
    # ─── FICC ───
    "mbs": DeskConfig(
        name="MBS/CMO", module="yield_book_env", factory="make_yield_book_env",
        total_timesteps=500_000, hidden_dim=384, lr=1.5e-4,
        rollout_steps=1024, batch_size=128, n_epochs=6,
    ),
    "treasuries": DeskConfig(
        name="Treasuries", module="treasuries_env", factory="make_treasuries_env",
        total_timesteps=300_000, hidden_dim=256,
    ),
    "distressed": DeskConfig(
        name="Distressed Credit", module="distressed_credit_env",
        factory="make_distressed_credit_env",
        total_timesteps=300_000, hidden_dim=384,
        entropy_coeff=0.04,  # Larger action space needs more exploration
    ),
    "rates": DeskConfig(
        name="Rates", module="rates_env", factory="make_rates_env",
        total_timesteps=400_000, hidden_dim=384, lr=1.5e-4,
        rollout_steps=1024,
    ),
    "ig_credit": DeskConfig(
        name="IG Credit", module="ig_credit_env", factory="make_ig_credit_env",
        total_timesteps=250_000, hidden_dim=256,
    ),
    "munis": DeskConfig(
        name="Munis", module="munis_env", factory="make_munis_env",
        total_timesteps=250_000, hidden_dim=256,
    ),
    "repo": DeskConfig(
        name="Repo", module="repo_env", factory="make_repo_env",
        total_timesteps=200_000, hidden_dim=256,
    ),
    "fx": DeskConfig(
        name="FX", module="fx_env", factory="make_fx_env",
        total_timesteps=300_000, hidden_dim=256,
    ),
    "commodities": DeskConfig(
        name="Commodities", module="commodities_env", factory="make_commodities_env",
        total_timesteps=300_000, hidden_dim=256,
    ),
    # ─── Equities ───
    "equities": DeskConfig(
        name="Equities Cash", module="equities_env", factory="make_equities_env",
        total_timesteps=300_000, hidden_dim=384,
        entropy_coeff=0.04, rollout_steps=1024,
    ),
    "eq_derivatives": DeskConfig(
        name="Eq Derivatives", module="eq_derivatives_env",
        factory="make_eq_derivatives_env",
        total_timesteps=400_000, hidden_dim=384, lr=1.5e-4,
        entropy_coeff=0.04,
    ),
    "prime_brokerage": DeskConfig(
        name="Prime Brokerage", module="prime_brokerage_env",
        factory="make_prime_brokerage_env",
        total_timesteps=200_000, hidden_dim=256,
    ),
    # ─── IBD ───
    "ecm": DeskConfig(
        name="ECM/IPO", module="ecm_env", factory="make_ecm_env",
        total_timesteps=200_000, hidden_dim=256,
    ),
    "dcm": DeskConfig(
        name="DCM/LevFin", module="dcm_env", factory="make_dcm_env",
        total_timesteps=300_000, hidden_dim=384, lr=1.5e-4,
        entropy_coeff=0.04, rollout_steps=1024, n_epochs=6,
        reward_scale=0.01,  # Normalize large reward magnitudes
    ),
    "ma_advisory": DeskConfig(
        name="M&A Advisory", module="ma_advisory_env", factory="make_ma_advisory_env",
        total_timesteps=200_000, hidden_dim=256,
    ),
    # ─── Meta-Environments ───
    "ficc_floor": DeskConfig(
        name="FICC Floor", module="ficc_floor_env", factory="make_ficc_floor_env",
        total_timesteps=500_000, hidden_dim=384, lr=1e-4,
        entropy_coeff=0.04, rollout_steps=2048, batch_size=256, n_epochs=6,
    ),
    "investment_bank": DeskConfig(
        name="Investment Bank", module="investment_bank_env",
        factory="make_investment_bank_env",
        total_timesteps=1_000_000, hidden_dim=512, lr=1e-4,
        entropy_coeff=0.05, rollout_steps=2048, batch_size=256, n_epochs=8,
    ),
}

FICC_DESKS = ["mbs", "treasuries", "distressed", "rates", "ig_credit",
              "munis", "repo", "fx", "commodities"]
EQUITY_DESKS = ["equities", "eq_derivatives", "prime_brokerage"]
IBD_DESKS = ["ecm", "dcm", "ma_advisory"]
META_ENVS = ["ficc_floor", "investment_bank"]
ALL_DESKS = FICC_DESKS + EQUITY_DESKS + IBD_DESKS
ALL_ENVS = ALL_DESKS + META_ENVS


# ─── Reuse core RL components from train.py ───────────────────────────────

class PolicyNetwork(nn.Module):
    """Actor-Critic network for PPO with layer norm."""

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
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, dim),
            ) for dim in action_dims
        ])
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
        with torch.no_grad():
            logits, value = self.forward(obs.unsqueeze(0))
        actions, log_probs = [], []
        for head_logits in logits:
            dist = Categorical(logits=head_logits.squeeze(0))
            action = dist.probs.argmax() if deterministic else dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        return actions, torch.stack(log_probs).sum().item(), value.item()

    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        total_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        total_entropy = torch.zeros(obs.shape[0], device=obs.device)
        for i, head_logits in enumerate(logits):
            dist = Categorical(logits=head_logits)
            total_log_prob += dist.log_prob(actions[:, i])
            total_entropy += dist.entropy()
        return total_log_prob, values.squeeze(-1), total_entropy


class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns(self, gamma=0.99, lam=0.95):
        n = len(self.rewards)
        returns, advantages = np.zeros(n), np.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = self.values[t + 1] if t + 1 < n else 0.0
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]
        return torch.tensor(returns, dtype=torch.float32), \
               torch.tensor(advantages, dtype=torch.float32)

    def to_tensors(self, device="cpu"):
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns, advantages = self.compute_returns()
        return obs, actions, log_probs, returns.to(device), advantages.to(device)

    def clear(self):
        for attr in ['obs', 'actions', 'rewards', 'dones', 'log_probs', 'values']:
            getattr(self, attr).clear()

    def __len__(self):
        return len(self.obs)


# ─── Generic Desk Trainer ─────────────────────────────────────────────────

class DeskTrainer:
    """PPO trainer for any desk environment."""

    def __init__(self, desk_key: str, seed: int = 42, device: str = "cpu",
                 override_timesteps: Optional[int] = None):
        self.desk_key = desk_key
        self.config = DESK_REGISTRY[desk_key]
        self.seed = seed
        self.device = device

        if override_timesteps:
            self.config.total_timesteps = override_timesteps

        # Create envs
        self.env = _make_env(self.config.module, self.config.factory, seed=seed)
        self.eval_env = _make_env(self.config.module, self.config.factory, seed=seed + 1000)

        obs_dim = self.env.observation_space.shape[0]
        action_dims = list(self.env.action_space.nvec)

        self.policy = PolicyNetwork(
            obs_dim, action_dims, self.config.hidden_dim
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()

        self._timestep = 0
        self.episode_rewards: list[float] = []
        self.eval_rewards: list[float] = []
        self._best_eval = -float('inf')

        print(f"[{self.config.name}] obs={obs_dim}, actions={action_dims}, "
              f"hidden={self.config.hidden_dim}, steps={self.config.total_timesteps:,}")

    def _entropy_coeff(self) -> float:
        anneal_steps = self.config.total_timesteps // 2
        if self._timestep >= anneal_steps:
            return 0.002
        frac = self._timestep / anneal_steps
        return self.config.entropy_coeff * (1 - frac) + 0.002 * frac

    def train(self, save_dir: Optional[str] = None) -> dict:
        obs, _ = self.env.reset()
        ep_reward, ep_len = 0.0, 0
        start = time.time()

        while self._timestep < self.config.total_timesteps:
            # Collect rollout
            for _ in range(self.config.rollout_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, lp, val = self.policy.get_action(obs_t)
                next_obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc

                self.buffer.add(obs, action, reward * self.config.reward_scale,
                                done, lp, val)
                ep_reward += reward
                ep_len += 1
                self._timestep += 1

                if done:
                    self.episode_rewards.append(ep_reward)
                    obs, _ = self.env.reset()
                    ep_reward, ep_len = 0.0, 0
                else:
                    obs = next_obs

            self._ppo_update()

            # Eval
            if self._timestep % self.config.eval_freq < self.config.rollout_steps:
                eval_r = self._evaluate()
                self.eval_rewards.append(eval_r)
                elapsed = time.time() - start
                fps = self._timestep / elapsed if elapsed > 0 else 0
                train_r = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                print(f"[{self.config.name:15s}] Step {self._timestep:>8d} | "
                      f"Train: {train_r:>8.1f} | Eval: {eval_r:>8.1f} | "
                      f"FPS: {fps:.0f}")

                # Save best
                if save_dir and eval_r > self._best_eval:
                    self._best_eval = eval_r
                    self.save(os.path.join(save_dir, f"{self.desk_key}_best.pt"))

        # Final save
        if save_dir:
            self.save(os.path.join(save_dir, f"{self.desk_key}_final.pt"))

        return {
            "desk": self.desk_key,
            "name": self.config.name,
            "timesteps": self._timestep,
            "train_rewards": self.episode_rewards,
            "eval_rewards": self.eval_rewards,
            "best_eval": self._best_eval,
            "time": time.time() - start,
        }

    def _ppo_update(self):
        obs, actions, old_lp, returns, advantages = self.buffer.to_tensors(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ent_coeff = self._entropy_coeff()

        # Learning rate annealing: linearly decay to 20% of initial lr
        frac_remaining = 1.0 - self._timestep / self.config.total_timesteps
        new_lr = self.config.lr * max(0.2, frac_remaining)
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                if end > len(obs):
                    break
                idx = indices[start:end]
                b_obs, b_act = obs[idx], actions[idx]
                b_old_lp, b_ret, b_adv = old_lp[idx], returns[idx], advantages[idx]

                new_lp, values, entropy = self.policy.evaluate_actions(b_obs, b_act)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, b_ret)
                loss = policy_loss + 0.5 * value_loss - ent_coeff * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.clear()

    def _evaluate(self) -> float:
        rewards = []
        for _ in range(self.config.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            total, done = 0.0, False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, _, _ = self.policy.get_action(obs_t, deterministic=True)
                obs, r, term, trunc, _ = self.eval_env.step(action)
                total += r
                done = term or trunc
            rewards.append(total)
        return float(np.mean(rewards))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "desk_key": self.desk_key,
            "config": {k: v for k, v in self.config.__dict__.items()},
            "episode_rewards": self.episode_rewards,
            "eval_rewards": self.eval_rewards,
            "best_eval": self._best_eval,
            "timestep": self._timestep,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._timestep = ckpt.get("timestep", 0)
        self._best_eval = ckpt.get("best_eval", -float('inf'))
        print(f"  Loaded {path} (step {self._timestep}, best_eval={self._best_eval:.1f})")


# ─── Benchmarking ─────────────────────────────────────────────────────────

def benchmark_desk(desk_key: str, n_episodes: int = 100, seed: int = 42,
                   model_path: Optional[str] = None) -> dict:
    """Benchmark random vs trained agent on a single desk."""
    cfg = DESK_REGISTRY[desk_key]
    env = _make_env(cfg.module, cfg.factory, seed=seed)

    # Random baseline
    random_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        random_rewards.append(total)

    result = {
        "desk": desk_key,
        "name": cfg.name,
        "random_mean": float(np.mean(random_rewards)),
        "random_std": float(np.std(random_rewards)),
    }

    # Trained agent (if model exists)
    if model_path and os.path.exists(model_path):
        obs_dim = env.observation_space.shape[0]
        action_dims = list(env.action_space.nvec)
        policy = PolicyNetwork(obs_dim, action_dims, cfg.hidden_dim)
        ckpt = torch.load(model_path, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()

        trained_rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total, done = 0.0, False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, _, _ = policy.get_action(obs_t, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                total += r
                done = term or trunc
            trained_rewards.append(total)

        result["trained_mean"] = float(np.mean(trained_rewards))
        result["trained_std"] = float(np.std(trained_rewards))
        result["improvement"] = result["trained_mean"] - result["random_mean"]

    return result


def benchmark_all(n_episodes: int = 50, seed: int = 42,
                  model_dir: Optional[str] = None) -> dict:
    """Benchmark all desks, print comparison table."""
    print(f"\n{'='*80}")
    print(f"  INVESTMENT BANK BENCHMARK — {len(ALL_ENVS)} Environments")
    print(f"{'='*80}\n")

    results = {}
    print(f"{'Desk':<20} {'Obs':>5} {'Acts':>12} {'Random':>10} {'Trained':>10} {'Delta':>8}")
    print(f"{'-'*65}")

    for desk_key in ALL_ENVS:
        cfg = DESK_REGISTRY[desk_key]
        model_path = os.path.join(model_dir, f"{desk_key}_best.pt") if model_dir else None

        try:
            result = benchmark_desk(desk_key, n_episodes, seed, model_path)
            results[desk_key] = result

            env = _make_env(cfg.module, cfg.factory, seed=seed)
            obs_dim = env.observation_space.shape[0]
            act_str = "x".join(str(d) for d in env.action_space.nvec)

            trained_str = f"{result.get('trained_mean', 0):>8.1f}" if 'trained_mean' in result else "     N/A"
            delta_str = f"{result.get('improvement', 0):>+7.1f}" if 'improvement' in result else "     N/A"

            if desk_key == FICC_DESKS[0]:
                print(f"  {'── FICC ──':}")
            elif desk_key == EQUITY_DESKS[0]:
                print(f"  {'── Equities ──':}")
            elif desk_key == IBD_DESKS[0]:
                print(f"  {'── IBD ──':}")
            elif desk_key == META_ENVS[0]:
                print(f"  {'── Meta ──':}")

            print(f"  {cfg.name:<18} {obs_dim:>5} {act_str:>12} "
                  f"{result['random_mean']:>+9.1f} {trained_str} {delta_str}")

        except Exception as e:
            print(f"  {cfg.name:<18} {'ERROR':>5} {str(e)[:30]}")
            results[desk_key] = {"error": str(e)}

    print(f"\n{'='*80}")

    # Summary stats
    random_means = [r["random_mean"] for r in results.values() if "random_mean" in r]
    print(f"\nRandom baseline: mean across desks = {np.mean(random_means):.1f}")
    if any("trained_mean" in r for r in results.values()):
        trained_means = [r["trained_mean"] for r in results.values() if "trained_mean" in r]
        improvements = [r["improvement"] for r in results.values() if "improvement" in r]
        print(f"Trained agents: mean = {np.mean(trained_means):.1f}, "
              f"avg improvement = {np.mean(improvements):+.1f}")

    return results


# ─── Curriculum Training ──────────────────────────────────────────────────

def train_curriculum(
    desks: Optional[list[str]] = None,
    timesteps_per_desk: Optional[int] = None,
    seed: int = 42,
    save_dir: str = "models/bank",
    device: str = "cpu",
) -> dict:
    """
    Curriculum training: train individual desks, then meta-environments.

    Stage 1: Train all individual desks in parallel (or sequentially)
    Stage 2: Train FICC Floor meta-environment
    Stage 3: Train Investment Bank meta-environment

    Args:
        desks: List of desk keys to train (default: all)
        timesteps_per_desk: Override timesteps for all desks
        seed: Random seed
        save_dir: Directory to save models
        device: Training device
    """
    if desks is None:
        desks = ALL_DESKS  # Individual desks first

    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    # Stage 1: Individual desks
    print(f"\n{'='*70}")
    print(f"  STAGE 1: Training {len(desks)} Individual Desks")
    print(f"{'='*70}\n")

    for i, desk_key in enumerate(desks):
        print(f"\n--- [{i+1}/{len(desks)}] {DESK_REGISTRY[desk_key].name} ---")
        trainer = DeskTrainer(desk_key, seed=seed, device=device,
                              override_timesteps=timesteps_per_desk)
        result = trainer.train(save_dir=save_dir)
        all_results[desk_key] = result
        print(f"  Final: train={np.mean(result['train_rewards'][-20:]):.1f}, "
              f"best_eval={result['best_eval']:.1f}, "
              f"time={result['time']:.0f}s")

    # Stage 2: FICC Floor (if FICC desks were trained)
    ficc_trained = [d for d in FICC_DESKS if d in desks]
    if len(ficc_trained) >= 3:
        print(f"\n{'='*70}")
        print(f"  STAGE 2: Training FICC Floor Meta-Environment")
        print(f"{'='*70}\n")
        trainer = DeskTrainer("ficc_floor", seed=seed, device=device,
                              override_timesteps=timesteps_per_desk)
        result = trainer.train(save_dir=save_dir)
        all_results["ficc_floor"] = result

    # Stage 3: Investment Bank (if enough desks were trained)
    if len(desks) >= 10:
        print(f"\n{'='*70}")
        print(f"  STAGE 3: Training Investment Bank Meta-Environment")
        print(f"{'='*70}\n")
        trainer = DeskTrainer("investment_bank", seed=seed, device=device,
                              override_timesteps=timesteps_per_desk)
        result = trainer.train(save_dir=save_dir)
        all_results["investment_bank"] = result

    # Final benchmark
    print(f"\n{'='*70}")
    print(f"  FINAL BENCHMARK")
    print(f"{'='*70}")
    benchmark_all(n_episodes=20, seed=seed, model_dir=save_dir)

    # Save summary
    summary = {
        desk: {
            "name": r.get("name", desk),
            "timesteps": r.get("timesteps", 0),
            "best_eval": r.get("best_eval", 0),
            "time": r.get("time", 0),
            "final_train": float(np.mean(r["train_rewards"][-20:])) if r.get("train_rewards") else 0,
        }
        for desk, r in all_results.items()
    }
    summary_path = os.path.join(save_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return all_results


# ─── Quick Smoke Test ─────────────────────────────────────────────────────

def smoke_test_all(seed: int = 42):
    """Verify all environments can be created, reset, and stepped."""
    print(f"\n{'='*60}")
    print(f"  Smoke Test: {len(ALL_ENVS)} Environments")
    print(f"{'='*60}\n")

    passed, failed = 0, 0
    total_obs_dim = 0

    for desk_key in ALL_ENVS:
        cfg = DESK_REGISTRY[desk_key]
        try:
            env = _make_env(cfg.module, cfg.factory, seed=seed)
            obs, info = env.reset()
            action = env.action_space.sample()
            next_obs, reward, term, trunc, info = env.step(action)

            obs_dim = obs.shape[0]
            act_str = "x".join(str(d) for d in env.action_space.nvec)
            total_obs_dim += obs_dim
            print(f"  OK  {cfg.name:<20} obs={obs_dim:>4}  act={act_str:<12}  r={reward:>+8.2f}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {cfg.name:<20} {e}")
            failed += 1

    print(f"\n  {passed} passed, {failed} failed, {total_obs_dim} total obs dims")
    return passed, failed


# ─── CLI Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Investment Bank Training Pipeline")
    sub = parser.add_subparsers(dest="command")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Smoke test all environments")
    p_smoke.add_argument("--seed", type=int, default=42)

    # train
    p_train = sub.add_parser("train", help="Train a single desk")
    p_train.add_argument("desk", choices=list(DESK_REGISTRY.keys()))
    p_train.add_argument("--timesteps", type=int, default=None)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--save-dir", default="models/bank")

    # curriculum
    p_curr = sub.add_parser("curriculum", help="Curriculum training across all desks")
    p_curr.add_argument("--desks", nargs="*", default=None,
                        choices=list(DESK_REGISTRY.keys()))
    p_curr.add_argument("--timesteps", type=int, default=None)
    p_curr.add_argument("--seed", type=int, default=42)
    p_curr.add_argument("--device", default="cpu")
    p_curr.add_argument("--save-dir", default="models/bank")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark all desks")
    p_bench.add_argument("--episodes", type=int, default=50)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--model-dir", default=None)

    args = parser.parse_args()

    if args.command == "smoke":
        smoke_test_all(seed=args.seed)
    elif args.command == "train":
        trainer = DeskTrainer(args.desk, seed=args.seed, device=args.device,
                              override_timesteps=args.timesteps)
        trainer.train(save_dir=args.save_dir)
    elif args.command == "curriculum":
        train_curriculum(
            desks=args.desks,
            timesteps_per_desk=args.timesteps,
            seed=args.seed,
            save_dir=args.save_dir,
            device=args.device,
        )
    elif args.command == "benchmark":
        benchmark_all(n_episodes=args.episodes, seed=args.seed, model_dir=args.model_dir)
    else:
        parser.print_help()
