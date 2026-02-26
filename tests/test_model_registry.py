"""
Tests for cmo_agent.model_registry module.
"""
import json
import os
import time
import struct

import pytest

from cmo_agent.model_registry import (
    ModelMetadata,
    register_model,
    list_models,
    get_model,
    check_compatibility,
    scan_unregistered,
    delete_model,
    _load_registry,
    _save_registry,
    REGISTRY_PATH,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_registry(tmp_model_dir, monkeypatch):
    """Patch REGISTRY_PATH and MODELS_DIR to use temp directory."""
    reg_path = str(tmp_model_dir / "models" / "registry.json")
    mod_dir = str(tmp_model_dir / "models")

    monkeypatch.setattr("cmo_agent.model_registry.REGISTRY_PATH", reg_path)
    monkeypatch.setattr("cmo_agent.model_registry.MODELS_DIR", mod_dir)

    return mod_dir, reg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegisterAndList:
    """register_model + list_models round-trip."""

    def test_register_returns_id(self, patched_registry):
        meta = ModelMetadata(
            model_id="test_model_001",
            file_path="test_model_001.pt",
            obs_dim=175,
            description="Test model for unit tests",
        )
        model_id = register_model(meta)
        assert model_id == "test_model_001"

    def test_list_after_register(self, patched_registry):
        meta = ModelMetadata(
            model_id="list_test",
            file_path="list_test.pt",
            obs_dim=175,
        )
        register_model(meta)
        models = list_models()
        assert len(models) >= 1
        ids = [m["model_id"] for m in models]
        assert "list_test" in ids

    def test_register_multiple(self, patched_registry):
        for i in range(3):
            meta = ModelMetadata(
                model_id=f"multi_{i}",
                file_path=f"multi_{i}.pt",
                obs_dim=175,
            )
            register_model(meta)

        models = list_models()
        assert len(models) == 3


class TestGetModel:
    """get_model should retrieve metadata by ID."""

    def test_get_existing(self, patched_registry):
        meta = ModelMetadata(
            model_id="get_test",
            file_path="get_test.pt",
            obs_dim=175,
            hidden_dim=256,
            best_eval_reward=42.5,
            description="For get_model test",
        )
        register_model(meta)

        result = get_model("get_test")
        assert result is not None
        assert result["model_id"] == "get_test"
        assert result["obs_dim"] == 175
        assert result["hidden_dim"] == 256
        assert result["best_eval_reward"] == 42.5

    def test_get_nonexistent(self, patched_registry):
        result = get_model("does_not_exist")
        assert result is None


class TestCheckCompatibility:
    """check_compatibility should compare obs_dim and env names."""

    def test_compatible(self, patched_registry):
        meta = ModelMetadata(
            model_id="compat_test",
            file_path="compat_test.pt",
            obs_dim=175,
            compatible_envs=["AGENCY"],
        )
        register_model(meta)

        result = check_compatibility("compat_test", env_obs_dim=175, env_name="AGENCY")
        assert result["compatible"] is True
        assert result["issues"] == []

    def test_incompatible_obs_dim(self, patched_registry):
        meta = ModelMetadata(
            model_id="incompat_obs",
            file_path="incompat_obs.pt",
            obs_dim=175,
            compatible_envs=["AGENCY"],
        )
        register_model(meta)

        result = check_compatibility("incompat_obs", env_obs_dim=185, env_name="AGENCY")
        assert result["compatible"] is False
        assert any("obs_dim" in issue for issue in result["issues"])

    def test_incompatible_env(self, patched_registry):
        meta = ModelMetadata(
            model_id="incompat_env",
            file_path="incompat_env.pt",
            obs_dim=175,
            compatible_envs=["AGENCY"],
        )
        register_model(meta)

        result = check_compatibility("incompat_env", env_obs_dim=175, env_name="CMBS")
        assert result["compatible"] is False
        assert any("env" in issue for issue in result["issues"])

    def test_nonexistent_model(self, patched_registry):
        result = check_compatibility("nope", env_obs_dim=175)
        assert result["compatible"] is False
        assert "error" in result


class TestScanUnregistered:
    """scan_unregistered should discover .pt files not yet in registry."""

    def test_scan_with_mock_pt(self, patched_registry):
        """Create a minimal fake .pt file and scan it."""
        mod_dir, _ = patched_registry

        # Create a minimal file that torch.load will read.
        # Since scan_unregistered imports torch, we skip if torch unavailable.
        torch = pytest.importorskip("torch")

        fake_path = os.path.join(mod_dir, "fake_model.pt")
        # Save a minimal checkpoint dict
        checkpoint = {
            "policy_state_dict": {
                "shared.0.weight": torch.randn(256, 175),
                "shared.0.bias": torch.randn(256),
                "policy_heads.0.2.weight": torch.randn(15, 128),
                "policy_heads.1.2.weight": torch.randn(10, 128),
                "policy_heads.2.2.weight": torch.randn(20, 128),
                "policy_heads.3.2.weight": torch.randn(20, 128),
            },
            "best_eval_reward": 33.0,
            "timestep": 100000,
            "config": {"lr": 3e-4},
        }
        torch.save(checkpoint, fake_path)

        new_models = scan_unregistered()
        assert len(new_models) == 1
        assert new_models[0].model_id == "fake_model"
        assert new_models[0].obs_dim == 175
        assert new_models[0].hidden_dim == 256

    def test_scan_idempotent(self, patched_registry):
        """Second scan should find nothing new."""
        torch = pytest.importorskip("torch")
        mod_dir, _ = patched_registry

        fake_path = os.path.join(mod_dir, "idempotent.pt")
        torch.save({"policy_state_dict": {}}, fake_path)

        scan_unregistered()
        second_scan = scan_unregistered()
        assert len(second_scan) == 0


class TestDeleteModel:
    """delete_model should remove from registry."""

    def test_delete_existing(self, patched_registry):
        meta = ModelMetadata(
            model_id="delete_me",
            file_path="delete_me.pt",
            obs_dim=175,
        )
        register_model(meta)
        assert get_model("delete_me") is not None

        result = delete_model("delete_me")
        assert result is True
        assert get_model("delete_me") is None

    def test_delete_nonexistent(self, patched_registry):
        result = delete_model("never_existed")
        assert result is False

    def test_delete_does_not_remove_file(self, patched_registry):
        """delete_model only removes from registry, not the file."""
        mod_dir, _ = patched_registry

        fake_path = os.path.join(mod_dir, "keep_file.pt")
        with open(fake_path, "wb") as f:
            f.write(b"fake data")

        meta = ModelMetadata(
            model_id="keep_file",
            file_path="keep_file.pt",
            obs_dim=175,
        )
        register_model(meta)
        delete_model("keep_file")

        # File should still exist on disk
        assert os.path.exists(fake_path)
