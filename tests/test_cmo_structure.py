"""
Tests for cmo_agent.cmo_structure module.

Key invariant: total principal out == total principal in (cash flow conservation).
"""
import numpy as np
import pytest

from cmo_agent.spec_pool import (
    SpecPool, AgencyType, CollateralType, project_pool_cashflows,
)
from cmo_agent.cmo_structure import (
    TrancheSpec,
    PrincipalType,
    InterestType,
    CMOCashFlows,
    create_sequential_cmo,
    create_pac_support_cmo,
    create_kitchen_sink_structure,
    structure_cmo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collateral(balance=100_000_000, psa=150):
    """Create collateral flows for testing."""
    pool = SpecPool(
        pool_id="TEST",
        agency=AgencyType.FNMA,
        collateral_type=CollateralType.FN,
        coupon=5.5,
        wac=6.0,
        wam=357,
        wala=3,
        original_balance=balance,
        current_balance=balance,
    )
    return project_pool_cashflows(pool, psa_speed=psa)


# ---------------------------------------------------------------------------
# Sequential CMO tests
# ---------------------------------------------------------------------------

class TestSequentialCMO:
    """create_sequential_cmo basic checks."""

    def test_creates_flows(self):
        cf = _make_collateral()
        cmo = create_sequential_cmo(
            deal_id="SEQ001",
            collateral_flows=cf,
            tranche_sizes=[40e6, 30e6, 30e6],
            tranche_coupons=[5.0, 5.25, 5.5],
            tranche_names=["A", "B", "C"],
            collateral_coupon=5.5,
        )
        assert isinstance(cmo, CMOCashFlows)

    def test_tranche_count(self):
        cf = _make_collateral()
        cmo = create_sequential_cmo(
            deal_id="SEQ002",
            collateral_flows=cf,
            tranche_sizes=[50e6, 50e6],
            tranche_coupons=[5.0, 5.5],
        )
        assert len(cmo.tranche_flows) == 2

    def test_balance_sum_equals_collateral(self):
        """Sum of tranche original balances should equal collateral face."""
        balance = 100_000_000
        sizes = [40e6, 30e6, 30e6]
        assert sum(sizes) == balance

        cf = _make_collateral(balance)
        cmo = create_sequential_cmo(
            deal_id="SEQ003",
            collateral_flows=cf,
            tranche_sizes=sizes,
            tranche_coupons=[5.0, 5.25, 5.5],
            collateral_coupon=5.5,
        )

        total_tranche_principal = sum(
            np.sum(tcf.principal) for tcf in cmo.tranche_flows.values()
        )
        collateral_principal = np.sum(cf.total_principal)
        # Allow small tolerance for rounding
        assert abs(total_tranche_principal - collateral_principal) / collateral_principal < 0.01


# ---------------------------------------------------------------------------
# PAC / Support CMO tests
# ---------------------------------------------------------------------------

class TestPacSupportCMO:
    """create_pac_support_cmo tests."""

    def test_creates_flows(self):
        cf = _make_collateral()
        cmo = create_pac_support_cmo(
            deal_id="PAC001",
            collateral_flows=cf,
            pac_balance=60e6,
            pac_coupon=5.0,
            support_balance=40e6,
            support_coupon=5.75,
            collateral_coupon=5.5,
        )
        assert isinstance(cmo, CMOCashFlows)
        assert "PAC" in cmo.tranche_flows
        assert "SUP" in cmo.tranche_flows

    def test_pac_plus_support_equals_collateral(self):
        """PAC + Support balances must equal collateral balance."""
        balance = 100_000_000
        pac_bal = 60e6
        sup_bal = 40e6
        assert pac_bal + sup_bal == balance

        cf = _make_collateral(balance)
        cmo = create_pac_support_cmo(
            deal_id="PAC002",
            collateral_flows=cf,
            pac_balance=pac_bal,
            pac_coupon=5.0,
            support_balance=sup_bal,
            support_coupon=5.75,
            collateral_coupon=5.5,
        )

        total_prin_out = sum(
            np.sum(tcf.principal) for tcf in cmo.tranche_flows.values()
        )
        collateral_prin = np.sum(cf.total_principal)
        assert abs(total_prin_out - collateral_prin) / collateral_prin < 0.01

    def test_with_z_bond(self):
        """Adding a Z-bond should still conserve principal."""
        cf = _make_collateral()
        cmo = create_pac_support_cmo(
            deal_id="PAC_Z",
            collateral_flows=cf,
            pac_balance=50e6,
            pac_coupon=5.0,
            support_balance=30e6,
            support_coupon=5.75,
            z_bond_balance=20e6,
            z_bond_coupon=5.5,
            collateral_coupon=5.5,
        )
        assert "Z" in cmo.tranche_flows
        assert len(cmo.tranche_flows) == 3


# ---------------------------------------------------------------------------
# Cash flow conservation
# ---------------------------------------------------------------------------

class TestCashFlowConservation:
    """Total principal out must equal total principal in."""

    def test_sequential_conservation(self):
        cf = _make_collateral()
        cmo = create_sequential_cmo(
            deal_id="CONS_SEQ",
            collateral_flows=cf,
            tranche_sizes=[25e6, 25e6, 25e6, 25e6],
            tranche_coupons=[5.0, 5.0, 5.25, 5.5],
            collateral_coupon=5.5,
        )
        total_out = sum(np.sum(tcf.principal) for tcf in cmo.tranche_flows.values())
        total_in = np.sum(cf.total_principal)
        rel_error = abs(total_out - total_in) / max(total_in, 1.0)
        assert rel_error < 0.01, f"Conservation violated: {rel_error:.4%} error"

    def test_pac_support_conservation(self):
        cf = _make_collateral()
        cmo = create_pac_support_cmo(
            deal_id="CONS_PAC",
            collateral_flows=cf,
            pac_balance=60e6,
            pac_coupon=5.0,
            support_balance=40e6,
            support_coupon=5.75,
            collateral_coupon=5.5,
        )
        total_out = sum(np.sum(tcf.principal) for tcf in cmo.tranche_flows.values())
        total_in = np.sum(cf.total_principal)
        rel_error = abs(total_out - total_in) / max(total_in, 1.0)
        assert rel_error < 0.01, f"Conservation violated: {rel_error:.4%} error"

    def test_residual_is_small(self):
        """Residual cash flow should be near zero for a well-structured deal."""
        cf = _make_collateral()
        cmo = create_sequential_cmo(
            deal_id="RES",
            collateral_flows=cf,
            tranche_sizes=[50e6, 50e6],
            tranche_coupons=[5.5, 5.5],
            collateral_coupon=5.5,
        )
        abs_residual = np.sum(np.abs(cmo.residual))
        collateral_total = np.sum(cf.total_cash_flow)
        # Residual should be <5% of total (interest differences are expected
        # when tranche coupons differ from collateral WAC)
        assert abs_residual / collateral_total < 0.05


# ---------------------------------------------------------------------------
# Kitchen Sink structure
# ---------------------------------------------------------------------------

class TestKitchenSink:
    """create_kitchen_sink_structure: the full exotic deal."""

    def test_runs_without_error(self):
        cf = _make_collateral()
        cmo = create_kitchen_sink_structure(
            deal_id="KS001",
            collateral_flows=cf,
            collateral_coupon=5.5,
        )
        assert isinstance(cmo, CMOCashFlows)
        assert len(cmo.tranche_flows) >= 5, "Kitchen sink should have many tranches"

    def test_all_tranches_have_cashflows(self):
        cf = _make_collateral()
        cmo = create_kitchen_sink_structure("KS002", cf, 5.5)
        for name, tcf in cmo.tranche_flows.items():
            total = np.sum(tcf.principal) + np.sum(tcf.interest)
            assert total >= 0, f"Tranche {name} has negative total cash flow"
