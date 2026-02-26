"""
Model Registry for CMO Agent trained models.

Tracks metadata, compatibility, and provides discovery for all .pt model files.
JSON store at models/registry.json.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "registry.json")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@dataclass
class ModelMetadata:
    model_id: str
    file_path: str
    obs_dim: int = 175
    action_dims: list[int] = field(default_factory=lambda: [14, 10, 20, 20])
    hidden_dim: int = 256
    env_name: str = "YieldBookEnv"
    best_eval_reward: float = 0.0
    training_config: dict = field(default_factory=dict)
    compatible_envs: list[str] = field(default_factory=lambda: ["AGENCY"])
    parent_model: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    description: str = ""
    timesteps: int = 0


def _load_registry() -> dict[str, dict]:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict[str, dict]):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(meta: ModelMetadata) -> str:
    """Register a model in the registry. Returns model_id."""
    registry = _load_registry()
    registry[meta.model_id] = asdict(meta)
    _save_registry(registry)
    return meta.model_id


def list_models() -> list[dict]:
    """List all registered models."""
    registry = _load_registry()
    return list(registry.values())


def get_model(model_id: str) -> Optional[dict]:
    """Get metadata for a specific model."""
    registry = _load_registry()
    return registry.get(model_id)


def load_model(model_id: str):
    """Load a model by registry ID. Returns (policy, metadata)."""
    import torch
    from .train import PolicyNetwork

    registry = _load_registry()
    meta = registry.get(model_id)
    if meta is None:
        raise ValueError(f"Model '{model_id}' not in registry")

    file_path = meta["file_path"]
    if not os.path.isabs(file_path):
        file_path = os.path.join(MODELS_DIR, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    checkpoint = torch.load(file_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("policy_state_dict", checkpoint)

    policy = PolicyNetwork(
        obs_dim=meta["obs_dim"],
        action_dims=meta["action_dims"],
        hidden=meta["hidden_dim"],
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, meta


def check_compatibility(model_id: str, env_obs_dim: int, env_name: str = "AGENCY") -> dict:
    """Check if a model is compatible with a given environment."""
    registry = _load_registry()
    meta = registry.get(model_id)
    if meta is None:
        return {"compatible": False, "error": f"Model '{model_id}' not in registry"}

    issues = []
    if meta["obs_dim"] != env_obs_dim:
        issues.append(f"obs_dim mismatch: model={meta['obs_dim']}, env={env_obs_dim}")
    if env_name not in meta.get("compatible_envs", []):
        issues.append(f"env '{env_name}' not in compatible_envs: {meta['compatible_envs']}")

    return {
        "compatible": len(issues) == 0,
        "model_id": model_id,
        "issues": issues,
    }


def scan_unregistered() -> list[ModelMetadata]:
    """Auto-discover .pt files not yet in the registry. Returns list of newly registered models."""
    import torch

    registry = _load_registry()
    registered_files = {m["file_path"] for m in registry.values()}
    # Also check basenames
    registered_basenames = {os.path.basename(m["file_path"]) for m in registry.values()}

    new_models = []
    if not os.path.exists(MODELS_DIR):
        return new_models

    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".pt"):
            continue
        full_path = os.path.join(MODELS_DIR, fname)
        if fname in registered_basenames or full_path in registered_files:
            continue

        # Probe the checkpoint for metadata
        try:
            checkpoint = torch.load(full_path, weights_only=False, map_location="cpu")
        except Exception:
            continue

        state_dict = checkpoint.get("policy_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            continue

        # Detect architecture from weights
        obs_dim = 129  # default old
        hidden_dim = 256
        action_dims = [14, 10, 20, 20]

        if "shared.0.weight" in state_dict:
            obs_dim = state_dict["shared.0.weight"].shape[1]
            hidden_dim = state_dict["shared.0.weight"].shape[0]

        detected_dims = []
        for i in range(10):
            key = f"policy_heads.{i}.2.weight"
            if key in state_dict:
                detected_dims.append(state_dict[key].shape[0])
            else:
                break
        if detected_dims:
            action_dims = detected_dims

        best_eval = 0.0
        config = {}
        timesteps = 0
        if isinstance(checkpoint, dict):
            best_eval = checkpoint.get("best_eval_reward", 0.0)
            config = checkpoint.get("config", {})
            timesteps = checkpoint.get("timestep", 0)

        model_id = fname.replace(".pt", "")
        compatible = ["AGENCY"]
        if "cmbs" in fname.lower():
            compatible = ["CMBS"]
        elif "multi_client" in fname.lower():
            compatible = ["MULTI_CLIENT"]

        meta = ModelMetadata(
            model_id=model_id,
            file_path=fname,
            obs_dim=obs_dim,
            action_dims=action_dims,
            hidden_dim=hidden_dim,
            best_eval_reward=float(best_eval) if best_eval else 0.0,
            training_config=config if isinstance(config, dict) else {},
            compatible_envs=compatible,
            created_at=os.path.getmtime(full_path),
            timesteps=timesteps,
        )
        register_model(meta)
        new_models.append(meta)

    return new_models


def delete_model(model_id: str) -> bool:
    """Remove a model from the registry (does not delete the file)."""
    registry = _load_registry()
    if model_id in registry:
        del registry[model_id]
        _save_registry(registry)
        return True
    return False
