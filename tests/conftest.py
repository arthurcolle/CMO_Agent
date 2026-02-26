"""
Shared test fixtures for the CMO Agent test suite.
"""
import os
import sys
import json
import tempfile

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmo_agent.yield_curve import YieldCurve, YieldCurvePoint, build_us_treasury_curve
from cmo_agent.spec_pool import (
    SpecPool, AgencyType, CollateralType, PoolCharacteristic,
    project_pool_cashflows,
)
from cmo_agent.prepayment import PrepaymentModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def curve():
    """Build a US Treasury yield curve (hardcoded fallback, no network)."""
    return build_us_treasury_curve(live=False)


@pytest.fixture
def pool():
    """Standard FNMA 5.5% 30-year pool, $100M face."""
    return SpecPool(
        pool_id="TEST_POOL_001",
        agency=AgencyType.FNMA,
        collateral_type=CollateralType.FN,
        coupon=5.5,
        wac=6.0,
        wam=357,
        wala=3,
        original_balance=100_000_000,
        current_balance=100_000_000,
        original_term=360,
        characteristic=PoolCharacteristic.TBA,
        avg_loan_size=300_000,
        avg_fico=740,
        avg_ltv=80,
    )


@pytest.fixture
def pool_cashflows(pool):
    """Projected cash flows for the standard pool at 150 PSA."""
    return project_pool_cashflows(pool, psa_speed=150)


@pytest.fixture
def env():
    """Create a YieldBookEnv in AGENCY mode with a fixed seed."""
    from cmo_agent.yield_book_env import YieldBookEnv
    return YieldBookEnv(
        max_steps=30,
        max_tranches=10,
        collateral_balance=100_000_000,
        seed=42,
        deal_mode="AGENCY",
    )


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Provide a temporary directory for model registry tests.

    Creates a dummy registry.json and returns the path to the temp dir.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_path = models_dir / "registry.json"
    registry_path.write_text("{}")
    return tmp_path
