"""
Tests for cmo_agent.deal_economics module.
"""
import numpy as np
import pytest

from cmo_agent.deal_economics import (
    compute_deal_pnl,
    compute_io_value_ticks,
    compute_investor_surplus_ticks,
    compute_convexity_redistribution_ticks,
    validate_deal_structure,
    market_clearing_spread,
    DealPnL,
    StructuralValidation,
    DemandCurve,
    DEMAND_CURVES,
)
from cmo_agent.cmo_structure import (
    TrancheSpec,
    PrincipalType,
    InterestType,
)
from cmo_agent.pricing import PricingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_tranches(collateral_balance=100e6, wac=6.0):
    """Create a simple PAC + Support + IO tranche list."""
    pac = TrancheSpec(
        name="PAC",
        principal_type=PrincipalType.PAC,
        interest_type=InterestType.FIXED,
        original_balance=collateral_balance * 0.60,
        coupon=5.0,
        priority=0,
    )
    sup = TrancheSpec(
        name="SUP",
        principal_type=PrincipalType.SUPPORT,
        interest_type=InterestType.FIXED,
        original_balance=collateral_balance * 0.40,
        coupon=5.75,
        priority=1,
    )
    io = TrancheSpec(
        name="IO",
        principal_type=PrincipalType.PASSTHROUGH,
        interest_type=InterestType.IO_ONLY,
        original_balance=0,
        notional_balance=collateral_balance,
        coupon=0.5,
        priority=99,
    )
    return [pac, sup, io]


def _dummy_pricing(tranches):
    """Create minimal PricingResult objects for testing."""
    results = {}
    for t in tranches:
        results[t.name] = PricingResult(
            name=t.name,
            price=100.0,
            yield_pct=5.0,
            wal_years=5.0,
            mod_duration=4.0,
            eff_duration=4.0,
            convexity=-2.0,
            spread_bps=0.0,
            oas_bps=0.0,
            z_spread_bps=0.0,
            dv01=0.04,
            window="1-120 months",
            first_pay=1,
            last_pay=120,
            total_principal=t.original_balance,
            total_interest=t.original_balance * 0.05 * 5,
        )
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeDealPnlExists:
    """compute_deal_pnl should exist and be callable."""

    def test_callable(self):
        assert callable(compute_deal_pnl)

    def test_returns_deal_pnl(self):
        tranches = _simple_tranches()
        pricing = _dummy_pricing(tranches)
        pnl = compute_deal_pnl(
            tranches=tranches,
            pricing_results=pricing,
            collateral_wac=6.0,
            collateral_balance=100e6,
            collateral_wal=5.0,
        )
        assert isinstance(pnl, DealPnL)
        assert isinstance(pnl.total_ticks, float)
        assert isinstance(pnl.arb_ticks, float)
        assert isinstance(pnl.io_ticks, float)

    def test_n_tranches(self):
        tranches = _simple_tranches()
        pricing = _dummy_pricing(tranches)
        pnl = compute_deal_pnl(
            tranches=tranches,
            pricing_results=pricing,
            collateral_wac=6.0,
            collateral_balance=100e6,
            collateral_wal=5.0,
        )
        assert pnl.n_tranches == 3


class TestValidateDealStructure:
    """validate_deal_structure should catch structural issues."""

    def test_valid_deal(self):
        tranches = _simple_tranches()
        v = validate_deal_structure(tranches, collateral_wac=6.0, collateral_balance=100e6)
        assert isinstance(v, StructuralValidation)
        # No unpaired inverse floaters
        assert v.n_inverse == 0
        assert len(v.warnings) == 0

    def test_unpaired_inverse_flagged(self):
        """An inverse floater without a matching floater should be flagged."""
        inv = TrancheSpec(
            name="INV",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.INVERSE_FLOATING,
            original_balance=10e6,
            coupon=0.0,
        )
        v = validate_deal_structure([inv], collateral_wac=6.0, collateral_balance=10e6)
        assert v.n_inverse == 1
        assert v.n_paired == 0
        assert len(v.warnings) > 0

    def test_io_excess_coupon(self):
        """Excess coupon should be WAC minus weighted avg bond coupon."""
        tranches = _simple_tranches(100e6, wac=6.0)
        v = validate_deal_structure(tranches, 6.0, 100e6)
        assert v.io_excess_coupon_pct >= 0


class TestIoValue:
    """compute_io_value_ticks tests."""

    def test_io_present(self):
        tranches = _simple_tranches()
        ticks = compute_io_value_ticks(tranches, 6.0, 100e6, 5.0)
        assert ticks > 0, "IO value should be positive when IO strip exists"

    def test_no_io_zero(self):
        """Without an IO strip, IO value should be 0."""
        pac = TrancheSpec(
            name="PAC",
            principal_type=PrincipalType.PAC,
            interest_type=InterestType.FIXED,
            original_balance=100e6,
            coupon=5.0,
        )
        ticks = compute_io_value_ticks([pac], 6.0, 100e6, 5.0)
        assert ticks == 0.0


class TestMarketClearingSpread:
    """market_clearing_spread tests."""

    def test_returns_float(self):
        t = TrancheSpec(
            name="T",
            principal_type=PrincipalType.PAC,
            interest_type=InterestType.FIXED,
            original_balance=50e6,
            coupon=5.0,
        )
        spread = market_clearing_spread(t, supply_of_type_M=50.0)
        assert isinstance(spread, float)

    def test_structurally_invalid_wide_spread(self):
        """Structurally invalid tranches should get a 150bp blowout."""
        t = TrancheSpec(
            name="INV",
            principal_type=PrincipalType.SEQUENTIAL,
            interest_type=InterestType.INVERSE_FLOATING,
            original_balance=10e6,
            coupon=0.0,
        )
        spread = market_clearing_spread(t, supply_of_type_M=10.0, structurally_valid=False)
        assert spread == 150.0


class TestConvexityRedistribution:
    """compute_convexity_redistribution_ticks tests."""

    def test_pac_support_positive(self):
        """A PAC + Support deal should have positive convexity alpha."""
        tranches = _simple_tranches()
        durations = {"PAC": 4.0, "SUP": 6.0, "IO": 3.0}
        ticks = compute_convexity_redistribution_ticks(tranches, durations, 100e6)
        assert ticks >= 0.0

    def test_empty_tranches(self):
        ticks = compute_convexity_redistribution_ticks([], {}, 100e6)
        assert ticks == 0.0
