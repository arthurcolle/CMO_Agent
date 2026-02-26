"""
Model Migration Tool: 129/135-dim -> 175-dim observation space.

Old models trained with obs_dim=129 or 135 are incompatible with the current
175-dim environment (68 market + 40 ecosystem + 67 deal). This tool:

1. Expands the first linear layer (shared.0.weight) from [hidden, old] to [hidden, 175]
   by copying existing columns and filling new ones with small random init.
2. Copies all other layers unchanged (they operate on hidden_dim, not obs_dim).
3. Saves migrated checkpoint with "_migrated_175" suffix.
4. Registers the new model in model_registry with parent_model link.
5. Optionally fine-tunes for N steps via PPOTrainer on the new env.

Usage:
    python -m cmo_agent.migrate_models                          # migrate all old models
    python -m cmo_agent.migrate_models --model gse_shelf_200k.pt
    python -m cmo_agent.migrate_models --fine-tune --steps 20000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .train import PolicyNetwork, PPOConfig, PPOTrainer
from .model_registry import (
    ModelMetadata,
    register_model,
    MODELS_DIR,
)
from .yield_book_env import make_yield_book_env

NEW_OBS_DIM = 175
KNOWN_OLD_DIMS = {129, 135}


def _detect_architecture(state_dict: dict) -> dict:
    """Probe a state_dict to recover obs_dim, hidden_dim, and action_dims."""
    obs_dim = 129
    hidden_dim = 256
    action_dims = [14, 10, 20, 20]

    if "shared.0.weight" in state_dict:
        obs_dim = state_dict["shared.0.weight"].shape[1]
        hidden_dim = state_dict["shared.0.weight"].shape[0]

    detected = []
    for i in range(10):
        key = f"policy_heads.{i}.2.weight"
        if key in state_dict:
            detected.append(state_dict[key].shape[0])
        else:
            break
    if detected:
        action_dims = detected

    return {"obs_dim": obs_dim, "hidden_dim": hidden_dim, "action_dims": action_dims}


def _expand_first_layer(
    old_weight: torch.Tensor,
    old_bias: torch.Tensor,
    new_obs_dim: int,
    init_scale: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand shared.0 weight from [hidden, old_obs_dim] to [hidden, new_obs_dim].

    Existing columns are preserved. New columns are initialized with small
    random values (N(0, init_scale)) so the network produces near-identical
    outputs on the old feature subset and near-zero contribution from the
    new ecosystem dimensions until fine-tuning teaches it to use them.
    """
    hidden_dim, old_obs_dim = old_weight.shape
    assert new_obs_dim > old_obs_dim, (
        f"new_obs_dim ({new_obs_dim}) must exceed old ({old_obs_dim})"
    )

    new_weight = torch.randn(hidden_dim, new_obs_dim) * init_scale
    new_weight[:, :old_obs_dim] = old_weight

    # Bias is shape [hidden_dim] and does not depend on obs_dim, copy as-is.
    new_bias = old_bias.clone()

    return new_weight, new_bias


def migrate_single_model(
    old_path: str,
    new_obs_dim: int = NEW_OBS_DIM,
    fine_tune: bool = False,
    fine_tune_steps: int = 20_000,
    init_scale: float = 0.01,
) -> str:
    """Migrate one checkpoint from old obs_dim to new_obs_dim.

    Args:
        old_path: Path to old .pt checkpoint.
        new_obs_dim: Target observation dimension (default 175).
        fine_tune: If True, run PPOTrainer for fine_tune_steps after migration.
        fine_tune_steps: Number of PPO timesteps for fine-tuning.
        init_scale: Std of random init for new input columns.

    Returns:
        Path to the saved migrated checkpoint.
    """
    if not os.path.isabs(old_path):
        old_path = os.path.join(MODELS_DIR, old_path)
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"Model not found: {old_path}")

    checkpoint = torch.load(old_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("policy_state_dict", checkpoint)

    arch = _detect_architecture(state_dict)
    old_obs_dim = arch["obs_dim"]
    hidden_dim = arch["hidden_dim"]
    action_dims = arch["action_dims"]

    if old_obs_dim == new_obs_dim:
        print(f"  [skip] {os.path.basename(old_path)} already has obs_dim={new_obs_dim}")
        return old_path

    if old_obs_dim > new_obs_dim:
        raise ValueError(
            f"Cannot shrink obs_dim from {old_obs_dim} to {new_obs_dim}. "
            "Only expansion is supported."
        )

    print(f"  Migrating {os.path.basename(old_path)}: "
          f"obs_dim {old_obs_dim} -> {new_obs_dim}, "
          f"hidden={hidden_dim}, action_dims={action_dims}")

    # --- Expand first layer ---
    new_w, new_b = _expand_first_layer(
        state_dict["shared.0.weight"],
        state_dict["shared.0.bias"],
        new_obs_dim,
        init_scale=init_scale,
    )

    # Build new state dict
    new_state_dict = {}
    for key, val in state_dict.items():
        if key == "shared.0.weight":
            new_state_dict[key] = new_w
        elif key == "shared.0.bias":
            new_state_dict[key] = new_b
        else:
            new_state_dict[key] = val.clone() if isinstance(val, torch.Tensor) else val

    # Verify it loads into a new-dim PolicyNetwork
    new_policy = PolicyNetwork(new_obs_dim, action_dims, hidden=hidden_dim)
    new_policy.load_state_dict(new_state_dict)

    # Quick sanity: forward pass with random input should not crash
    test_obs = torch.randn(1, new_obs_dim)
    logits, value = new_policy(test_obs)
    assert len(logits) == len(action_dims), "Policy head count mismatch after migration"

    # --- Build new checkpoint ---
    basename = os.path.basename(old_path).replace(".pt", "")
    new_name = f"{basename}_migrated_{new_obs_dim}.pt"
    new_path = os.path.join(MODELS_DIR, new_name)

    new_checkpoint = {
        "policy_state_dict": new_policy.state_dict(),
        "config": checkpoint.get("config", {}),
        "best_eval_reward": checkpoint.get("best_eval_reward", 0.0),
        "timestep": checkpoint.get("timestep", 0),
        "migration": {
            "source_file": os.path.basename(old_path),
            "old_obs_dim": old_obs_dim,
            "new_obs_dim": new_obs_dim,
            "init_scale": init_scale,
            "migrated_at": time.time(),
        },
    }
    # We intentionally omit optimizer_state_dict: the old optimizer state has
    # wrong shapes for the expanded parameter, so fine-tuning starts fresh.

    torch.save(new_checkpoint, new_path)
    print(f"  Saved migrated model -> {new_path}")

    # --- Register in model registry ---
    parent_id = basename
    model_id = new_name.replace(".pt", "")
    meta = ModelMetadata(
        model_id=model_id,
        file_path=new_name,
        obs_dim=new_obs_dim,
        action_dims=action_dims,
        hidden_dim=hidden_dim,
        best_eval_reward=float(checkpoint.get("best_eval_reward", 0.0) or 0.0),
        training_config=checkpoint.get("config", {}),
        compatible_envs=["AGENCY"],
        parent_model=parent_id,
        description=f"Migrated from {parent_id} (obs_dim {old_obs_dim}->{new_obs_dim})",
        timesteps=checkpoint.get("timestep", 0) or 0,
    )
    register_model(meta)
    print(f"  Registered as '{model_id}' (parent: {parent_id})")

    # --- Optional fine-tuning ---
    if fine_tune:
        print(f"  Fine-tuning for {fine_tune_steps:,} steps...")
        env = make_yield_book_env(mode="AGENCY")
        eval_env = make_yield_book_env(mode="AGENCY")

        config = PPOConfig(
            total_timesteps=fine_tune_steps,
            lr=1e-4,             # Lower LR for fine-tuning (preserve learned weights)
            hidden_dim=hidden_dim,
            save_dir=MODELS_DIR,
            eval_freq=min(5000, fine_tune_steps // 4),
            use_lr_schedule=True,
            warmup_steps=min(2000, fine_tune_steps // 10),
            normalize_rewards=True,
            use_ema=True,
        )
        trainer = PPOTrainer(env, config=config, eval_env=eval_env)

        # Inject the migrated weights into the trainer's policy
        trainer.policy.load_state_dict(new_policy.state_dict())
        trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=config.lr)

        if trainer.ema is not None:
            from .train import EMAPolicy
            trainer.ema = EMAPolicy(trainer.policy, decay=config.ema_decay)

        trainer.train()

        ft_name = f"{basename}_migrated_{new_obs_dim}_ft{fine_tune_steps // 1000}k.pt"
        ft_path = os.path.join(MODELS_DIR, ft_name)
        trainer.save(ft_path)
        print(f"  Fine-tuned model saved -> {ft_path}")
        new_path = ft_path

    return new_path


def migrate_all(
    new_obs_dim: int = NEW_OBS_DIM,
    fine_tune: bool = False,
    fine_tune_steps: int = 20_000,
) -> list[str]:
    """Scan models/ directory and migrate all old-dim models.

    Returns list of paths to newly created migrated checkpoints.
    """
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return []

    migrated = []
    skipped = 0

    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".pt"):
            continue
        # Skip models that are already migrations
        if "_migrated_" in fname:
            continue

        full_path = os.path.join(MODELS_DIR, fname)
        try:
            checkpoint = torch.load(full_path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"  [error] Cannot load {fname}: {e}")
            continue

        state_dict = checkpoint.get("policy_state_dict", checkpoint)
        if not isinstance(state_dict, dict) or "shared.0.weight" not in state_dict:
            skipped += 1
            continue

        old_obs_dim = state_dict["shared.0.weight"].shape[1]

        if old_obs_dim == new_obs_dim:
            skipped += 1
            continue

        if old_obs_dim >= new_obs_dim:
            print(f"  [skip] {fname}: obs_dim={old_obs_dim} >= target {new_obs_dim}")
            skipped += 1
            continue

        try:
            new_path = migrate_single_model(
                full_path,
                new_obs_dim=new_obs_dim,
                fine_tune=fine_tune,
                fine_tune_steps=fine_tune_steps,
            )
            migrated.append(new_path)
        except Exception as e:
            print(f"  [error] Failed to migrate {fname}: {e}")

    print(f"\nMigration complete: {len(migrated)} migrated, {skipped} skipped")
    return migrated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CMO Agent models from old obs_dim (129/135) to 175-dim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m cmo_agent.migrate_models\n"
            "  python -m cmo_agent.migrate_models --model gse_shelf_200k.pt\n"
            "  python -m cmo_agent.migrate_models --fine-tune --steps 20000\n"
            "  python -m cmo_agent.migrate_models --model ppo_agency_50k.pt --fine-tune\n"
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model file to migrate (basename or full path). "
             "If omitted, migrates all old models in models/ dir.",
    )
    parser.add_argument(
        "--fine-tune", action="store_true", default=False,
        help="Fine-tune migrated model on the new 175-dim environment.",
    )
    parser.add_argument(
        "--steps", type=int, default=20_000,
        help="Number of PPO timesteps for fine-tuning (default: 20000).",
    )
    parser.add_argument(
        "--obs-dim", type=int, default=NEW_OBS_DIM,
        help=f"Target observation dimension (default: {NEW_OBS_DIM}).",
    )
    parser.add_argument(
        "--init-scale", type=float, default=0.01,
        help="Std of random init for new input columns (default: 0.01).",
    )

    args = parser.parse_args()

    print(f"CMO Agent Model Migration Tool")
    print(f"Target obs_dim: {args.obs_dim}")
    print(f"Models dir: {MODELS_DIR}")
    print()

    if args.model:
        path = migrate_single_model(
            args.model,
            new_obs_dim=args.obs_dim,
            fine_tune=args.fine_tune,
            fine_tune_steps=args.steps,
            init_scale=args.init_scale,
        )
        print(f"\nDone. Migrated model: {path}")
    else:
        paths = migrate_all(
            new_obs_dim=args.obs_dim,
            fine_tune=args.fine_tune,
            fine_tune_steps=args.steps,
        )
        if paths:
            print("\nMigrated models:")
            for p in paths:
                print(f"  {p}")
        else:
            print("\nNo models needed migration.")


if __name__ == "__main__":
    main()
