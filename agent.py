"""
Autonomous CMO Pricing and Structuring Agent.
Uses Claude API to make intelligent decisions about buying spec pools,
structuring CMOs, pricing tranches, and generating trade recommendations.
"""
import json
import os
import sys
import traceback
from datetime import date, datetime
from dataclasses import dataclass, asdict
from typing import Optional, Any

import numpy as np
import anthropic

from .yield_curve import YieldCurve, YieldCurvePoint, build_us_treasury_curve, build_curve_from_dict
from .prepayment import (
    PrepaymentModel, PrepaymentModelConfig, estimate_psa_speed,
    mortgage_rate_from_treasury, psa_cpr, smm_from_cpr, cpr_from_smm,
)
from .spec_pool import (
    SpecPool, AgencyType, CollateralType, PoolCharacteristic,
    project_pool_cashflows, price_tba, spec_pool_payup,
)
from .cmo_structure import (
    TrancheSpec, PrincipalType, InterestType,
    create_sequential_cmo, create_pac_support_cmo,
    create_floater_inverse_structure, structure_cmo,
    create_vadm_z_structure, create_pac_jump_z_structure,
    create_pac_ii_structure, create_nas_structure,
    create_schedule_bond_structure, create_z_pac_structure,
    create_kitchen_sink_structure,
)
from .pricing import (
    price_tranche, price_deal, yield_from_price, price_from_yield,
    structuring_profit, pool_price_from_yield,
)
from .remic_loader import load_all_remic_data, analyze_remic_trends
from .fed_api import (
    get_latest_rates, get_sofr_latest, get_effr_latest,
    get_soma_summary, get_ambs_results_last, get_reverse_repo_last,
    get_market_data_snapshot, extract_sofr_rate, extract_effr_rate,
)
from .tba import analyze_dollar_roll, build_tba_price_grid, cheapest_to_deliver
from .risk import key_rate_durations, compute_full_risk_metrics, scenario_matrix
from .monte_carlo import compute_oas, MonteCarloConfig, HullWhiteParams
from .credit import (
    DoubleTriggerModel, price_crt_bond, structure_nonagency_cmo,
    stress_test_credit, CRTBondSpec, LoanCharacteristics, EconomicScenario,
)
from .analytics import compute_s_curve, relative_value_grid, burnout_analysis
from .prepayment import (
    StantonPrepaymentModel, StantonParams,
    compute_disguised_default_adjustment, DisguisedDefaultParams,
)
from .data_sources import (
    build_live_treasury_curve, fetch_treasury_curve_xml,
    fetch_current_mortgage_rates, fetch_mortgage_rate_history,
    fetch_fhfa_hpi_national, compute_refi_incentive,
    get_full_market_snapshot, check_all_sources,
    fred_fetch_series, fred_search,
)
from .backtest import (
    backtest_agent_vs_experts, backtest_to_json, load_policy,
    replay_agent_policy, _adapt_obs,
)
from .yield_book_env import YieldBookEnv, make_yield_book_env
from .compliance import (
    quick_check as cgc_quick_check,
    check_deal_compliance, check_portfolio,
    get_all_rules as cgc_get_all_rules,
    classify_cmo_tranche_subdivision,
    SecurityHolding, EntityProfile, CGCSubdivision,
)
from .ginnie_mae_participants import (
    validate_sponsor, get_directory_summary, resolve_firm,
)


# ─── Tool Definitions for Claude ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_yield_curve",
        "description": "Get the current US Treasury yield curve with all maturities. Returns yields for 1M through 30Y treasuries, plus interpolated rates at any maturity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "additional_maturities": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional additional maturities (in years) to interpolate"
                }
            },
        },
    },
    {
        "name": "estimate_prepayment_speed",
        "description": "Estimate prepayment speed (PSA) for a mortgage pool based on WAC, current rates, and loan characteristics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wac": {"type": "number", "description": "Weighted average coupon (e.g., 5.5 for 5.5%)"},
                "current_mortgage_rate": {"type": "number", "description": "Current prevailing mortgage rate (e.g., 5.9)"},
                "wala": {"type": "integer", "description": "Weighted average loan age in months", "default": 0},
                "n_months": {"type": "integer", "description": "Number of months to project", "default": 360},
            },
            "required": ["wac", "current_mortgage_rate"],
        },
    },
    {
        "name": "create_spec_pool",
        "description": "Create and analyze a specified mortgage pool. Returns pool characteristics and projected cash flows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pool_id": {"type": "string", "description": "Pool identifier"},
                "agency": {"type": "string", "enum": ["GNMA", "FNMA", "FHLMC"]},
                "coupon": {"type": "number", "description": "Pass-through coupon rate (e.g., 5.5)"},
                "wac": {"type": "number", "description": "Weighted average coupon"},
                "wam": {"type": "integer", "description": "Weighted average maturity in months"},
                "wala": {"type": "integer", "description": "Weighted average loan age in months", "default": 0},
                "original_balance": {"type": "number", "description": "Original pool balance in dollars"},
                "original_term": {"type": "integer", "description": "Original term (360 for 30yr, 180 for 15yr)", "default": 360},
                "characteristic": {"type": "string", "enum": ["TBA", "LLB", "HLTV", "NY", "INV", "LFICO", "GEO", "HWAC"], "default": "TBA"},
                "avg_loan_size": {"type": "number", "default": 300000},
                "avg_fico": {"type": "number", "default": 740},
                "avg_ltv": {"type": "number", "default": 80},
                "psa_speed": {"type": "number", "description": "PSA speed assumption", "default": 150},
            },
            "required": ["pool_id", "agency", "coupon", "wac", "wam", "original_balance"],
        },
    },
    {
        "name": "price_pool",
        "description": "Price a mortgage pool at a given yield or spread. Returns price, WAL, duration, and payup analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {"type": "number", "description": "Pool coupon rate"},
                "wac": {"type": "number", "description": "Weighted average coupon"},
                "wam": {"type": "integer", "description": "WAM in months"},
                "wala": {"type": "integer", "default": 0},
                "balance": {"type": "number", "description": "Pool balance"},
                "agency": {"type": "string", "enum": ["GNMA", "FNMA", "FHLMC"], "default": "GNMA"},
                "psa_speed": {"type": "number", "default": 150},
                "yield_pct": {"type": "number", "description": "Yield for pricing (if not provided, uses market)"},
                "characteristic": {"type": "string", "enum": ["TBA", "LLB", "HLTV", "NY", "INV", "LFICO"], "default": "TBA"},
            },
            "required": ["coupon", "wac", "wam", "balance"],
        },
    },
    {
        "name": "structure_sequential_cmo",
        "description": "Structure a sequential-pay CMO from collateral pools. Allocates principal sequentially to tranches A, B, C, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string"},
                "collateral_coupon": {"type": "number", "description": "Collateral coupon rate"},
                "collateral_wac": {"type": "number"},
                "collateral_wam": {"type": "integer"},
                "collateral_balance": {"type": "number"},
                "tranche_sizes": {"type": "array", "items": {"type": "number"}, "description": "Balance of each tranche"},
                "tranche_coupons": {"type": "array", "items": {"type": "number"}, "description": "Coupon of each tranche"},
                "tranche_names": {"type": "array", "items": {"type": "string"}, "description": "Names for tranches"},
                "psa_speed": {"type": "number", "default": 150},
            },
            "required": ["deal_id", "collateral_coupon", "collateral_wac", "collateral_wam",
                         "collateral_balance", "tranche_sizes", "tranche_coupons"],
        },
    },
    {
        "name": "structure_pac_cmo",
        "description": "Structure a PAC/Support CMO with optional Z-bond and IO tranches. PAC tranches have stable cash flows within PSA bands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string"},
                "collateral_coupon": {"type": "number"},
                "collateral_wac": {"type": "number"},
                "collateral_wam": {"type": "integer"},
                "collateral_balance": {"type": "number"},
                "pac_balance": {"type": "number", "description": "PAC tranche balance"},
                "pac_coupon": {"type": "number", "description": "PAC tranche coupon"},
                "support_balance": {"type": "number", "description": "Support tranche balance"},
                "support_coupon": {"type": "number"},
                "pac_lower_band": {"type": "number", "default": 100, "description": "Lower PSA band for PAC schedule"},
                "pac_upper_band": {"type": "number", "default": 300, "description": "Upper PSA band for PAC schedule"},
                "z_bond_balance": {"type": "number", "default": 0},
                "z_bond_coupon": {"type": "number", "default": 0},
                "io_notional": {"type": "number", "default": 0},
                "io_coupon": {"type": "number", "default": 0},
                "psa_speed": {"type": "number", "default": 150},
            },
            "required": ["deal_id", "collateral_coupon", "collateral_wac", "collateral_wam",
                         "collateral_balance", "pac_balance", "pac_coupon",
                         "support_balance", "support_coupon"],
        },
    },
    {
        "name": "price_cmo_deal",
        "description": "Price all tranches in a CMO deal. Returns yield, WAL, duration, OAS, and Z-spread for each tranche.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string"},
                "spreads": {"type": "object", "description": "Dict of tranche name -> spread in bps for pricing"},
            },
            "required": ["deal_id"],
        },
    },
    {
        "name": "calculate_structuring_profit",
        "description": "Calculate the profit from buying collateral and selling CMO tranches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collateral_cost_price": {"type": "number", "description": "Price paid for collateral (per 100)"},
                "collateral_balance": {"type": "number", "description": "Total face value of collateral"},
                "tranche_prices": {"type": "object", "description": "Dict of tranche name -> selling price (per 100)"},
                "tranche_balances": {"type": "object", "description": "Dict of tranche name -> face value"},
            },
            "required": ["collateral_cost_price", "collateral_balance", "tranche_prices", "tranche_balances"],
        },
    },
    {
        "name": "analyze_remic_history",
        "description": "Analyze historical REMIC issuance data from Ginnie Mae files. Shows trends in dealers, structure types, coupons, and volumes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_dir": {"type": "string", "description": "Path to REMIC data directory",
                            "default": "CMO REMIC"},
                "max_files": {"type": "integer", "description": "Maximum number of files to parse", "default": 5},
            },
        },
    },
    {
        "name": "scenario_analysis",
        "description": "Run scenario analysis on a CMO deal under different interest rate and prepayment scenarios.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string"},
                "rate_shifts_bps": {"type": "array", "items": {"type": "number"},
                                   "description": "Parallel rate shifts in bps (e.g., [-100, -50, 0, 50, 100])"},
                "psa_speeds": {"type": "array", "items": {"type": "number"},
                              "description": "PSA speeds to test (e.g., [75, 100, 150, 200, 300])"},
            },
            "required": ["deal_id"],
        },
    },
    {
        "name": "market_snapshot",
        "description": "Get a comprehensive market snapshot including Treasury yields, mortgage rates, MBS prices, and Fed expectations.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_fed_rates",
        "description": "Get live SOFR, EFFR, and other reference rates from the NY Fed Markets Data API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rate_type": {"type": "string", "enum": ["all", "sofr", "effr", "obfr"], "default": "all"},
                "n": {"type": "integer", "description": "Number of recent observations", "default": 5},
            },
        },
    },
    {
        "name": "get_soma_holdings",
        "description": "Get Federal Reserve SOMA (System Open Market Account) holdings data including MBS, Agency, and Treasury holdings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "holding_type": {"type": "string", "enum": ["summary", "agency", "mbs", "treasury"], "default": "summary"},
            },
        },
    },
    {
        "name": "get_ambs_operations",
        "description": "Get recent Agency MBS operations from the NY Fed (purchases, sales, dollar rolls, coupon swaps).",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of recent operations", "default": 5},
            },
        },
    },
    {
        "name": "analyze_dollar_roll",
        "description": "Analyze a dollar roll transaction (sell front month TBA, buy back month). Shows implied financing rate, drop, and roll advantage vs repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {"type": "number", "description": "MBS coupon rate (e.g., 5.5)"},
                "front_price": {"type": "number", "description": "Front month TBA price"},
                "back_price": {"type": "number", "description": "Back month TBA price"},
                "wam": {"type": "integer", "default": 357},
                "psa_speed": {"type": "number", "default": 150},
                "repo_rate": {"type": "number", "description": "Current repo rate for comparison", "default": 0.043},
            },
            "required": ["coupon", "front_price", "back_price"],
        },
    },
    {
        "name": "compute_key_rate_durations",
        "description": "Compute key rate durations (partial DV01s) for an MBS pool, showing sensitivity to shifts at each tenor on the curve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {"type": "number"},
                "wac": {"type": "number"},
                "wam": {"type": "integer"},
                "wala": {"type": "integer", "default": 0},
                "balance": {"type": "number"},
                "psa_speed": {"type": "number", "default": 150},
                "shift_bps": {"type": "number", "default": 25},
            },
            "required": ["coupon", "wac", "wam", "balance"],
        },
    },
    {
        "name": "compute_risk_metrics",
        "description": "Full risk analytics with prepayment repricing: effective duration/convexity, WAL sensitivity, price scenarios at +/-50 and +/-100 bps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {"type": "number"},
                "wac": {"type": "number"},
                "wam": {"type": "integer"},
                "wala": {"type": "integer", "default": 0},
                "balance": {"type": "number"},
                "psa_speed": {"type": "number", "default": 150},
                "agency": {"type": "string", "enum": ["GNMA", "FNMA", "FHLMC"], "default": "GNMA"},
            },
            "required": ["coupon", "wac", "wam", "balance"],
        },
    },
    {
        "name": "compute_scenario_matrix",
        "description": "Generate a price/WAL/duration scenario matrix across rate shifts and PSA speeds. Core of Yield Book scenario analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {"type": "number"},
                "wac": {"type": "number"},
                "wam": {"type": "integer"},
                "wala": {"type": "integer", "default": 0},
                "balance": {"type": "number"},
                "agency": {"type": "string", "default": "GNMA"},
                "rate_shifts": {"type": "array", "items": {"type": "integer"}, "description": "Rate shifts in bps"},
                "psa_speeds": {"type": "array", "items": {"type": "number"}, "description": "PSA speeds to test"},
            },
            "required": ["coupon", "wac", "wam", "balance"],
        },
    },
    {
        "name": "compute_monte_carlo_oas",
        "description": "Compute Option-Adjusted Spread using Monte Carlo simulation with Hull-White rate model and rate-dependent prepayments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_price": {"type": "number", "description": "Market price per 100 par"},
                "coupon": {"type": "number", "description": "Pass-through coupon (decimal, e.g., 0.055)"},
                "wac": {"type": "number", "description": "Weighted avg coupon (decimal, e.g., 0.06)"},
                "wam": {"type": "integer", "default": 360},
                "wala": {"type": "integer", "default": 0},
                "balance": {"type": "number", "default": 1000000},
                "n_paths": {"type": "integer", "default": 500},
            },
            "required": ["market_price", "coupon", "wac"],
        },
    },
    {
        "name": "price_crt_bond",
        "description": "Price a Credit Risk Transfer bond (STACR/CAS style) using expected loss framework with attachment/detachment points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bond_name": {"type": "string"},
                "attachment_pct": {"type": "number", "description": "Lower loss threshold (e.g., 0.5 for 0.5%)"},
                "detachment_pct": {"type": "number", "description": "Upper loss threshold (e.g., 2.0 for 2.0%)"},
                "coupon_spread_bps": {"type": "number", "description": "Spread over SOFR"},
                "expected_loss_pct": {"type": "number", "description": "Expected cumulative loss (e.g., 1.5 for 1.5%)"},
                "loss_vol": {"type": "number", "default": 0.5},
            },
            "required": ["bond_name", "attachment_pct", "detachment_pct", "coupon_spread_bps", "expected_loss_pct"],
        },
    },
    {
        "name": "structure_nonagency_deal",
        "description": "Structure a non-agency RMBS with senior-subordinated credit tranching. Shows AAA/AA/A/BBB/BB/equity waterfall with credit enhancement.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_name": {"type": "string"},
                "collateral_balance": {"type": "number"},
                "collateral_wac": {"type": "number"},
                "avg_fico": {"type": "number", "default": 720},
                "avg_ltv": {"type": "number", "default": 80},
                "senior_pct": {"type": "number", "description": "Senior tranche as % of deal (e.g., 0.80)", "default": 0.80},
                "scenario": {"type": "string", "enum": ["base", "mild_recession", "severe_recession"], "default": "base"},
            },
            "required": ["deal_name", "collateral_balance", "collateral_wac"],
        },
    },
    {
        "name": "credit_stress_test",
        "description": "Run credit stress tests across economic scenarios (base, mild recession, severe, great recession, depression).",
        "input_schema": {
            "type": "object",
            "properties": {
                "avg_fico": {"type": "number", "default": 720},
                "avg_ltv": {"type": "number", "default": 85},
                "collateral_wac": {"type": "number", "default": 6.0},
            },
        },
    },
    {
        "name": "compute_prepayment_s_curve",
        "description": "Compute the prepayment S-curve showing CPR vs. refinancing incentive. Decomposes into turnover and refi components.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wac": {"type": "number", "description": "WAC as decimal (e.g., 0.055 for 5.5%)"},
                "wala": {"type": "integer", "default": 24},
                "fico": {"type": "integer", "default": 740},
                "burnout_factor": {"type": "number", "description": "1.0 = fresh pool, 0.5 = heavily burned out", "default": 1.0},
            },
            "required": ["wac"],
        },
    },
    {
        "name": "relative_value_analysis",
        "description": "Compute relative value metrics across the coupon stack: price, WAL, OAS, duration, convexity for each coupon.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mortgage_rate": {"type": "number", "description": "Current mortgage rate (decimal, e.g., 0.059)"},
                "coupons": {"type": "array", "items": {"type": "number"}, "description": "Coupons to analyze (decimal)"},
            },
            "required": ["mortgage_rate"],
        },
    },
    {
        "name": "stanton_prepayment_analysis",
        "description": "Run the Stanton (1995) rational prepayment model with heterogeneous transaction costs, discrete decision frequency, and endogenous burnout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wac": {"type": "number", "description": "WAC as decimal (e.g., 0.055)"},
                "short_rate": {"type": "number", "description": "Current short rate (e.g., 0.04)"},
                "n_months": {"type": "integer", "default": 120},
                "loan_age": {"type": "integer", "default": 0},
                "use_monte_carlo": {"type": "boolean", "description": "Run MC with CIR rate paths", "default": False},
                "n_paths": {"type": "integer", "default": 200},
            },
            "required": ["wac", "short_rate"],
        },
    },
    {
        "name": "structure_advanced_cmo",
        "description": "Structure an advanced/exotic CMO using hold-down structures: VADM (prepayment-independent), Jump-Z (converts on support exhaustion), PAC-II (narrower bands), NAS (lockout), Schedule Bond (single-speed target), Z-PAC (Z-accrual + PAC), or Kitchen Sink (all types).",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string"},
                "structure_type": {
                    "type": "string",
                    "enum": ["vadm", "jump_z", "pac_ii", "nas", "schedule", "z_pac", "kitchen_sink"],
                    "description": "Type of exotic structure to create",
                },
                "collateral_coupon": {"type": "number"},
                "collateral_wac": {"type": "number"},
                "collateral_wam": {"type": "integer"},
                "collateral_balance": {"type": "number"},
                "psa_speed": {"type": "number", "default": 150},
                # VADM params
                "vadm_balance": {"type": "number", "description": "VADM tranche balance"},
                "vadm_coupon": {"type": "number"},
                "z_bond_balance": {"type": "number", "description": "Z-bond balance (VADM or Jump-Z)"},
                "z_bond_coupon": {"type": "number"},
                "seq_balances": {"type": "array", "items": {"type": "number"}, "description": "Sequential tranche balances (VADM)"},
                "seq_coupons": {"type": "array", "items": {"type": "number"}, "description": "Sequential tranche coupons (VADM)"},
                # PAC/Jump-Z params
                "pac_balance": {"type": "number"},
                "pac_coupon": {"type": "number"},
                "support_balance": {"type": "number"},
                "support_coupon": {"type": "number"},
                "pac_lower": {"type": "number", "default": 100},
                "pac_upper": {"type": "number", "default": 300},
                "sticky": {"type": "boolean", "default": True, "description": "Sticky Jump-Z stays current-pay once triggered"},
                # PAC-II params
                "pac_i_balance": {"type": "number"},
                "pac_i_coupon": {"type": "number"},
                "pac_ii_balance": {"type": "number"},
                "pac_ii_coupon": {"type": "number"},
                "pac_ii_lower": {"type": "number", "default": 150},
                "pac_ii_upper": {"type": "number", "default": 250},
                # NAS params
                "nas_balance": {"type": "number"},
                "nas_coupon": {"type": "number"},
                "lockout_months": {"type": "integer", "description": "NAS/Z-PAC lockout period in months"},
                "sequential_balance": {"type": "number"},
                "sequential_coupon": {"type": "number"},
                # Schedule bond params
                "schedule_balance": {"type": "number"},
                "schedule_coupon": {"type": "number"},
                "target_psa": {"type": "number", "description": "Target PSA speed for schedule bond"},
                # Z-PAC params
                "z_pac_balance": {"type": "number"},
                "z_pac_coupon": {"type": "number"},
            },
            "required": ["deal_id", "structure_type", "collateral_coupon", "collateral_wac",
                         "collateral_wam", "collateral_balance"],
        },
    },
    {
        "name": "disguised_default_analysis",
        "description": "Analyze the Hall & Maingi (2021) prepayment-default interaction. Shows how liquidity shocks manifest as prepayment (positive equity) or default (negative equity).",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_ltv": {"type": "number", "description": "Current LTV ratio"},
                "unemployment_change": {"type": "number", "description": "Change in unemployment from base (%)", "default": 0},
                "hpa_annual": {"type": "number", "description": "Annual HPA (%)", "default": 3.0},
                "loan_age_months": {"type": "integer", "default": 24},
            },
            "required": ["current_ltv"],
        },
    },
    # ─── Data Source Tools ────────────────────────────────────────────────
    {
        "name": "fetch_live_yield_curve",
        "description": "Fetch live US Treasury yield curve from FRED API or Treasury.gov. Falls back to hardcoded data if APIs are unavailable. Updates the agent's pricing curve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of_date": {"type": "string", "description": "Date (YYYY-MM-DD) for historical curve. Omit for latest."},
                "update_agent_curve": {"type": "boolean", "description": "Whether to update the agent's pricing curve with fetched data", "default": True},
            },
        },
    },
    {
        "name": "get_mortgage_rates",
        "description": "Fetch current and historical primary mortgage rates (30yr fixed, 15yr fixed, 5yr ARM) from FRED/Freddie Mac PMMS.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product": {"type": "string", "enum": ["30yr", "15yr", "5yr_arm", "all"], "default": "all"},
                "history_start": {"type": "string", "description": "Start date for historical data (YYYY-MM-DD). Omit for current only."},
            },
        },
    },
    {
        "name": "get_comprehensive_market_data",
        "description": "Get a comprehensive market snapshot: Treasury curve, mortgage rates, SOFR/EFFR, SOMA MBS holdings, housing data. Combines FRED + NY Fed + Treasury.gov data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_housing": {"type": "boolean", "description": "Include FHFA HPI and Case-Shiller data", "default": False},
                "include_spreads": {"type": "boolean", "description": "Include yield curve spread indicators (10Y-2Y, 10Y-3M)", "default": False},
            },
        },
    },
    {
        "name": "check_data_sources",
        "description": "Check connectivity and availability of all data sources (FRED, Treasury.gov, FHFA, NY Fed). Useful for diagnosing data issues.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "fred_series_lookup",
        "description": "Search FRED for economic data series or fetch specific series data. Covers 800,000+ time series: rates, housing, employment, inflation, MBS holdings, and more.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "fetch"], "description": "Search for series or fetch data"},
                "query": {"type": "string", "description": "Search query (for action=search) or series ID (for action=fetch)"},
                "start_date": {"type": "string", "description": "Start date for fetch (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Max results", "default": 20},
            },
            "required": ["action", "query"],
        },
    },
    {
        "name": "compute_refi_incentive",
        "description": "Compute refinancing incentive for a pool using live mortgage rate data. Shows incentive in bps, estimated PSA speed, and whether the pool is in/out of the money.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pool_wac": {"type": "number", "description": "Pool WAC (decimal, e.g. 0.065 for 6.5%)"},
                "current_rate": {"type": "number", "description": "Current 30yr mortgage rate (%). Omit to fetch live."},
            },
            "required": ["pool_wac"],
        },
    },
    # ─── RL-Powered Tools ────────────────────────────────────────────────
    {
        "name": "rl_suggest_structure",
        "description": "Use the trained RL agent to suggest an optimal CMO/REMIC structure for current market conditions. The agent was trained on 742 real Ginnie Mae REMIC deals and 200k+ PPO steps. Returns suggested tranche types, sizes, coupons, and estimated profit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to trained model (.pt file)", "default": "models/gse_shelf_200k.pt"},
                "collateral_balance": {"type": "number", "description": "Collateral face value", "default": 100000000},
                "n_suggestions": {"type": "integer", "description": "Number of deal suggestions to generate", "default": 3},
                "use_live_rates": {"type": "boolean", "description": "Fetch live FRED rates for the environment", "default": True},
            },
        },
    },
    {
        "name": "rl_backtest",
        "description": "Backtest the trained RL agent against 742 real Ginnie Mae REMIC deals structured by JPM, Goldman, BMO, Mizuho and other top dealers. Compares agent's structuring profit vs what dealers actually achieved.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to trained model", "default": "models/gse_shelf_200k.pt"},
                "n_deals": {"type": "integer", "description": "Number of deals to test (0 = all 742)", "default": 100},
            },
        },
    },
    {
        "name": "rl_live_deal",
        "description": "Generate a live deal recommendation: fetches current FRED rates, runs the trained RL agent in a calibrated environment, and returns a complete deal structure with risk metrics and estimated P&L.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to trained model", "default": "models/gse_shelf_200k.pt"},
                "collateral_balance": {"type": "number", "default": 100000000},
                "agency": {"type": "string", "enum": ["GNMA", "FNMA", "FHLMC"], "default": "GNMA"},
                "coupon": {"type": "number", "description": "Collateral coupon (e.g. 5.5). If omitted, uses current production coupon."},
            },
        },
    },
    # ─── CGC 53601 Compliance Tools ───────────────────────────────────────
    {
        "name": "cgc53601_check_security",
        "description": "Check a single security for CGC 53601 compliance. Returns applicable rules, maturity limits, rating requirements, concentration caps, and any violations. Handles SB 882 agency MBS exemption, SB 998 CP/MTN limits, SB 1489 settlement-date maturity and 45-day forward cap.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subdivision": {
                    "type": "string",
                    "description": "CGC 53601 subdivision: a (local agency bonds), b (US Treasury), c (CA state), d (other state), e (other local agency), f (federal agency/GSE), g (bankers acceptance), h (commercial paper), i (negotiable CD), j_repo, j_reverse, k (medium-term notes), l (mutual fund), m (trustee/indenture), n (secured obligations), o_agency (agency MBS - SB 882 exempt), o_non_agency (private-label ABS/MBS), p (JPA shares), q (supranational), r (public bank)",
                    "enum": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j_repo", "j_reverse", "k", "l", "m", "n", "o_agency", "o_non_agency", "p", "q", "r"],
                },
                "par_value": {"type": "number", "description": "Par/face value of the security"},
                "credit_rating": {"type": "string", "description": "Credit rating (e.g. AAA, AA, A, A-1, P-1)"},
                "maturity_days": {"type": "integer", "description": "Days from settlement to maturity"},
                "settlement_date": {"type": "string", "description": "Settlement date (YYYY-MM-DD)"},
                "maturity_date": {"type": "string", "description": "Maturity date (YYYY-MM-DD)"},
                "trade_date": {"type": "string", "description": "Trade date (YYYY-MM-DD) — for forward settlement check"},
                "entity_assets": {"type": "number", "description": "Entity's total investable assets (for SB 998 CP 40% threshold)", "default": 0},
                "instrument_subtype": {"type": "string", "description": "If prohibited type: inverse_floater, range_note, mortgage_derived_io_strip, zero_interest_accrual"},
            },
            "required": ["subdivision"],
        },
    },
    {
        "name": "cgc53601_check_deal",
        "description": "Check a proposed CMO deal (multiple tranches) for CGC 53601 compliance against an entity's existing portfolio. Validates prohibited instruments, maturity limits, concentration caps, single-issuer limits, and SB 882/998/1489 rules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_tranches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "par_value": {"type": "number"},
                            "market_value": {"type": "number"},
                            "agency": {"type": "string", "description": "GNMA, FNMA, FHLMC, or private issuer name"},
                            "principal_type": {"type": "string", "description": "SEQ, PAC, SUP, Z, IO, PO, etc."},
                            "interest_type": {"type": "string", "description": "FIX, FLT, INV, IO, PO, Z"},
                            "credit_rating": {"type": "string"},
                            "settlement_date": {"type": "string"},
                            "maturity_date": {"type": "string"},
                            "trade_date": {"type": "string"},
                            "issuer": {"type": "string"},
                        },
                    },
                    "description": "List of tranches in the proposed deal",
                },
                "entity_name": {"type": "string", "default": "Local Agency"},
                "entity_assets": {"type": "number", "description": "Entity's total investable assets", "default": 0},
                "has_extended_maturity_authority": {"type": "boolean", "default": False},
                "existing_portfolio_mv": {"type": "number", "description": "Market value of entity's existing portfolio (for concentration calc)", "default": 0},
            },
            "required": ["deal_tranches"],
        },
    },
    {
        "name": "cgc53601_rules",
        "description": "Get all CGC 53601 investment rules: authorized security types, maturity limits, concentration caps, rating requirements, prohibited instruments, and recent amendments (SB 882, SB 998, SB 1489).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cgc53601_classify_tranche",
        "description": "Classify a CMO tranche under CGC 53601 subdivisions and flag any prohibited instrument types. Determines if a tranche qualifies as agency MBS (SB 882 exempt) or non-agency, and checks for inverse floaters, IO strips, range notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agency": {"type": "string", "description": "Issuing agency: GNMA, FNMA, FHLMC, or private"},
                "principal_type": {"type": "string", "description": "SEQ, PAC, PAC2, TAC, SUP, PT, Z, IO, PO, NAS, VADM, etc."},
                "interest_type": {"type": "string", "description": "FIX, FLT, INV (inverse floater), IO, PO, Z"},
            },
            "required": ["agency", "principal_type", "interest_type"],
        },
    },
    # ─── Ginnie Mae Participant Tools ─────────────────────────────────────
    {
        "name": "ginnie_mae_validate_sponsor",
        "description": "Validate whether a dealer/firm is an approved Ginnie Mae REMIC sponsor or co-sponsor. Supports fuzzy matching (e.g. 'JPM', 'Goldman', 'BMO'). Returns approval status, roles (SF/MF/Reverse/Co-Sponsor), and contact info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dealer": {"type": "string", "description": "Firm name, alias, or abbreviation (e.g. 'JPM', 'Goldman Sachs', 'Mizuho')"},
                "program": {
                    "type": "string",
                    "enum": ["sf", "mf", "reverse", "all"],
                    "description": "Program to check: sf (single-family), mf (multifamily), reverse (HECM), all",
                    "default": "all",
                },
            },
            "required": ["dealer"],
        },
    },
    {
        "name": "ginnie_mae_participant_directory",
        "description": "Get the full Ginnie Mae approved multiclass participants directory: all sponsors (SF/MF/Reverse), co-sponsors, trustees, trust counsel, accountants, and REMIC info agents with contact details.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


class CMOAgent:
    """Autonomous CMO structuring and pricing agent."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.curve = build_us_treasury_curve()
        self.prepay_model = PrepaymentModel()
        self.deals: dict[str, dict] = {}  # Store structured deals
        self.pools: dict[str, dict] = {}  # Store created pools
        self.conversation_history: list[dict] = []

        # Current market data
        self.treasury_10y = 4.221
        self.current_mortgage_rate = mortgage_rate_from_treasury(self.treasury_10y)
        self.fed_rate = 3.625  # Mid-point of 3.50-3.75 target (82.3% prob per data)

    def _system_prompt(self) -> str:
        return f"""You are an expert autonomous CMO (Collateralized Mortgage Obligation) structuring and pricing agent.
You work at a major broker-dealer's structured products desk. Your role is to:

1. **Buy Spec Pools**: Analyze and purchase specified mortgage pools (GNMA, FNMA, FHLMC) based on prepayment characteristics
2. **Structure CMOs**: Create REMIC structures including sequential, PAC, TAC, support, Z-bond, IO/PO, floater/inverse floater, plus advanced hold-down structures: VADM, Jump-Z/Sticky Jump-Z, PAC-II, NAS (lockout), Schedule Bond, Z-PAC, Reverse TAC, Super PO
3. **Price Tranches**: Calculate yield, WAL, duration, OAS, Z-spread for each tranche
4. **Risk Analytics**: Key rate durations, effective duration/convexity with prepayment repricing, scenario matrices
5. **Monte Carlo OAS**: Hull-White rate model with rate-dependent prepayment projections
6. **Credit Analysis**: Double-trigger default model, CRT bond pricing, non-agency structuring, stress testing
7. **TBA Market**: Dollar roll analysis, cheapest-to-deliver, good delivery
8. **Prepayment Models**: Stanton (1995) rational model with heterogeneous transaction costs + Hall & Maingi (2021) disguised default channel
9. **Generate Trade Recommendations**: Identify profitable structuring opportunities

Current Market Data (as of {date.today().isoformat()}):
- US 10Y Treasury: {self.treasury_10y}%
- Current Mortgage Rate (est): {self.current_mortgage_rate:.3f}%
- Fed Funds Target: {self.fed_rate}% (82.3% probability of 3.50-3.75% at March meeting)
- US 2Y: 3.506%, US 5Y: 3.769%, US 30Y: 4.867%

Key Guidelines:
- Always consider prepayment risk when pricing MBS and CMO tranches
- Use the Stanton model for sophisticated prepayment analysis (endogenous burnout via beta-distributed transaction costs)
- PAC tranches should be sized to provide stable cash flows within PSA bands
- Support tranches absorb prepayment volatility and should be priced wider
- Z-bonds accrue interest and extend duration - use for duration-seeking investors
- IO tranches lose value as rates fall (more prepayments reduce notional)
- Consider the yield curve shape when structuring floater/inverse floater pairs
- For credit analysis, remember the Hall & Maingi dual-trigger insight: rising house prices suppress defaults by converting them to prepayments
- Typical structuring profit target: 10-50 bps on collateral face value
- Always verify that tranche sizes sum to collateral balance (no arbitrage)

Available tools let you build and price complete CMO deals from raw spec pools.
You have 42 tools for end-to-end structured products analysis.
Live data connectors: fetch_live_yield_curve (FRED/Treasury.gov), get_mortgage_rates (PMMS), get_comprehensive_market_data (full snapshot), fred_series_lookup (800K+ FRED series), compute_refi_incentive (live rate vs WAC), check_data_sources (connectivity check).
Advanced hold-down structures available: VADM (prepayment-independent via Z accretion), Jump-Z (converts on support exhaustion), PAC-II (narrow-band support carve-out), NAS (lockout period), Schedule Bond (single-speed target), Z-PAC (Z-accrual + PAC stability), Kitchen Sink (full deal with all exotic types).
RL-Powered tools: rl_suggest_structure (trained PPO agent suggests optimal deal structure), rl_backtest (compare agent vs 742 real dealer deals from JPM/Goldman/BMO/Mizuho), rl_live_deal (live FRED rates + calibrated RL agent = real-time deal recommendation).
The RL agent was trained on 200k+ PPO steps with imitation learning from 742 real Ginnie Mae REMIC deals. It knows GSE shelf structuring: PAC, Support, IO, Z-bond, Floater/Inverse Floater, and full kitchen-sink deals.
CGC 53601 Compliance tools: cgc53601_rules (full rule reference), cgc53601_check_security (single security check), cgc53601_check_deal (full deal compliance), cgc53601_classify_tranche (classify CMO tranche under CGC subdivisions). Implements SB 882 (agency MBS exempt), SB 998 (CP 40% for >$100M entities, 10% single-issuer CP+MTN), SB 1489 (settlement-date maturity, 45-day forward settlement cap). Checks prohibited instruments (inverse floaters, range notes, mortgage-derived IO strips, zero-interest-accrual). Use these when structuring deals for California local government investment programs.
Ginnie Mae Participant tools: ginnie_mae_validate_sponsor (check if a dealer is approved for SF/MF/Reverse REMIC programs), ginnie_mae_participant_directory (full directory of 20 SF sponsors, 19 MF sponsors, 9 Reverse sponsors, 10 co-sponsors, trustees, counsel, accountants). Use when structuring GNMA REMIC deals to validate dealers.
Be quantitative and precise in your analysis."""

    # Tool dispatch registry — maps tool_name -> method
    _tool_registry: dict[str, str] = {
        "get_yield_curve": "_tool_get_yield_curve",
        "estimate_prepayment_speed": "_tool_estimate_prepayment",
        "create_spec_pool": "_tool_create_spec_pool",
        "price_pool": "_tool_price_pool",
        "structure_sequential_cmo": "_tool_structure_sequential",
        "structure_pac_cmo": "_tool_structure_pac",
        "price_cmo_deal": "_tool_price_deal",
        "calculate_structuring_profit": "_tool_calc_profit",
        "analyze_remic_history": "_tool_analyze_remic",
        "scenario_analysis": "_tool_scenario_analysis",
        "market_snapshot": "_tool_market_snapshot",
        "get_fed_rates": "_tool_get_fed_rates",
        "get_soma_holdings": "_tool_get_soma_holdings",
        "get_ambs_operations": "_tool_get_ambs_operations",
        "analyze_dollar_roll": "_tool_analyze_dollar_roll",
        "compute_key_rate_durations": "_tool_key_rate_durations",
        "compute_risk_metrics": "_tool_risk_metrics",
        "compute_scenario_matrix": "_tool_scenario_matrix",
        "compute_monte_carlo_oas": "_tool_monte_carlo_oas",
        "price_crt_bond": "_tool_price_crt_bond",
        "structure_nonagency_deal": "_tool_structure_nonagency",
        "credit_stress_test": "_tool_credit_stress_test",
        "compute_prepayment_s_curve": "_tool_s_curve",
        "relative_value_analysis": "_tool_relative_value",
        "stanton_prepayment_analysis": "_tool_stanton_prepayment",
        "structure_advanced_cmo": "_tool_structure_advanced",
        "disguised_default_analysis": "_tool_disguised_default",
        "fetch_live_yield_curve": "_tool_fetch_live_curve",
        "get_mortgage_rates": "_tool_get_mortgage_rates",
        "get_comprehensive_market_data": "_tool_comprehensive_market",
        "check_data_sources": "_tool_check_data_sources",
        "fred_series_lookup": "_tool_fred_lookup",
        "compute_refi_incentive": "_tool_refi_incentive",
        "rl_suggest_structure": "_tool_rl_suggest",
        "rl_backtest": "_tool_rl_backtest",
        "rl_live_deal": "_tool_rl_live_deal",
        "cgc53601_check_security": "_tool_cgc_check_security",
        "cgc53601_check_deal": "_tool_cgc_check_deal",
        "cgc53601_rules": "_tool_cgc_rules",
        "cgc53601_classify_tranche": "_tool_cgc_classify",
        "ginnie_mae_validate_sponsor": "_tool_gnma_validate_sponsor",
        "ginnie_mae_participant_directory": "_tool_gnma_directory",
    }

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            method_name = self._tool_registry.get(tool_name)
            if method_name is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            return getattr(self, method_name)(tool_input)
        except Exception as e:
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

    # ─── Tool Implementations ────────────────────────────────────────────────

    def _tool_get_yield_curve(self, inp: dict) -> str:
        result = {
            "as_of_date": str(self.curve.as_of_date),
            "treasury_yields": {
                "1M": self.curve.get_yield(1/12),
                "3M": self.curve.get_yield(3/12),
                "6M": self.curve.get_yield(6/12),
                "1Y": self.curve.get_yield(1.0),
                "2Y": self.curve.get_yield(2.0),
                "3Y": self.curve.get_yield(3.0),
                "5Y": self.curve.get_yield(5.0),
                "7Y": self.curve.get_yield(7.0),
                "10Y": self.curve.get_yield(10.0),
                "20Y": self.curve.get_yield(20.0),
                "30Y": self.curve.get_yield(30.0),
            },
            "curve_shape": {
                "2s10s_spread_bps": round((self.curve.get_yield(10) - self.curve.get_yield(2)) * 100, 1),
                "2s30s_spread_bps": round((self.curve.get_yield(30) - self.curve.get_yield(2)) * 100, 1),
                "5s30s_spread_bps": round((self.curve.get_yield(30) - self.curve.get_yield(5)) * 100, 1),
            },
            "forward_rates": {
                "1y1y": round(self.curve.forward_rate(1, 2) * 100, 3),
                "2y3y": round(self.curve.forward_rate(2, 5) * 100, 3),
                "5y5y": round(self.curve.forward_rate(5, 10) * 100, 3),
            },
            "estimated_mortgage_rate": round(self.current_mortgage_rate, 3),
        }

        additional = inp.get("additional_maturities", [])
        if additional:
            result["additional"] = {
                f"{m}Y": round(self.curve.get_yield(m), 3) for m in additional
            }

        return json.dumps(result, indent=2)

    def _tool_estimate_prepayment(self, inp: dict) -> str:
        wac = inp["wac"]
        rate = inp["current_mortgage_rate"]
        wala = inp.get("wala", 0)
        n_months = inp.get("n_months", 360)

        psa_est = estimate_psa_speed(wac, rate / 100.0, wala)
        smms = self.prepay_model.project_schedule(
            n_months=min(n_months, 360),
            wac=wac / 100.0,
            current_mortgage_rate=rate / 100.0,
            loan_age=wala,
        )

        cprs = np.array([cpr_from_smm(s) for s in smms])

        result = {
            "estimated_psa_speed": round(psa_est, 0),
            "wac": wac,
            "current_rate": rate,
            "refinancing_incentive_bps": round((wac - rate) * 100, 0),
            "avg_cpr_year1": round(float(np.mean(cprs[:12])) * 100, 2),
            "avg_cpr_year2": round(float(np.mean(cprs[12:24])) * 100, 2) if len(cprs) > 12 else 0,
            "avg_cpr_year5": round(float(np.mean(cprs[48:60])) * 100, 2) if len(cprs) > 48 else 0,
            "peak_cpr": round(float(np.max(cprs)) * 100, 2),
            "steady_state_cpr": round(float(np.mean(cprs[30:60])) * 100, 2) if len(cprs) > 30 else 0,
            "cpr_schedule_quarterly": [
                round(float(np.mean(cprs[i:i+3])) * 100, 2)
                for i in range(0, min(len(cprs), 40), 3)
            ],
        }
        return json.dumps(result, indent=2)

    def _tool_create_spec_pool(self, inp: dict) -> str:
        char_map = {
            "TBA": PoolCharacteristic.TBA,
            "LLB": PoolCharacteristic.LOW_LOAN_BALANCE,
            "HLTV": PoolCharacteristic.HIGH_LTV,
            "NY": PoolCharacteristic.NEW_YORK,
            "INV": PoolCharacteristic.INVESTOR,
            "LFICO": PoolCharacteristic.LOW_FICO,
            "GEO": PoolCharacteristic.GEO_CONCENTRATED,
            "HWAC": PoolCharacteristic.HIGH_WAC,
        }

        pool = SpecPool(
            pool_id=inp["pool_id"],
            agency=AgencyType[inp["agency"]],
            collateral_type=CollateralType.G2 if inp["agency"] == "GNMA" else CollateralType.FN,
            coupon=inp["coupon"],
            wac=inp["wac"],
            wam=inp["wam"],
            wala=inp.get("wala", 0),
            original_balance=inp["original_balance"],
            current_balance=inp["original_balance"],
            original_term=inp.get("original_term", 360),
            characteristic=char_map.get(inp.get("characteristic", "TBA"), PoolCharacteristic.TBA),
            avg_loan_size=inp.get("avg_loan_size", 300000),
            avg_fico=inp.get("avg_fico", 740),
            avg_ltv=inp.get("avg_ltv", 80),
        )

        psa_speed = inp.get("psa_speed", 150)
        cf = project_pool_cashflows(pool, psa_speed=psa_speed)
        payup = spec_pool_payup(pool)
        tba_price = price_tba(pool.coupon, self.treasury_10y, pool.agency)

        self.pools[pool.pool_id] = {"pool": pool, "cashflows": cf, "psa_speed": psa_speed}

        result = {
            "pool_id": pool.pool_id,
            "agency": pool.agency.value,
            "coupon": pool.coupon,
            "wac": pool.wac,
            "wam": pool.wam,
            "wala": pool.wala,
            "original_balance": pool.original_balance,
            "characteristic": pool.characteristic.value,
            "avg_loan_size": pool.avg_loan_size,
            "avg_fico": pool.avg_fico,
            "avg_ltv": pool.avg_ltv,
            "num_loans": pool.num_loans,
            "psa_speed": psa_speed,
            "cashflow_summary": {
                "wal_years": round(cf.wal, 2),
                "total_interest": round(cf.total_interest, 2),
                "total_principal": round(cf.total_principal_paid, 2),
                "duration_months": cf.duration_months,
            },
            "tba_price": round(tba_price, 4),
            "payup_32nds": round(payup, 1),
            "spec_pool_price": round(tba_price + payup / 32.0, 4),
        }
        return json.dumps(result, indent=2)

    def _tool_price_pool(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id="PRICING",
            agency=AgencyType[inp.get("agency", "GNMA")],
            collateral_type=CollateralType.G2,
            coupon=inp["coupon"],
            wac=inp["wac"],
            wam=inp["wam"],
            wala=inp.get("wala", 0),
            original_balance=inp["balance"],
            current_balance=inp["balance"],
        )

        psa = inp.get("psa_speed", 150)
        cf = project_pool_cashflows(pool, psa_speed=psa)

        if "yield_pct" in inp:
            price = pool_price_from_yield(cf, inp["yield_pct"])
        else:
            tba = price_tba(pool.coupon, self.treasury_10y, pool.agency)
            price = tba

        y = yield_from_price(
            type('obj', (), {
                'total_cash_flow': cf.total_cash_flow,
                'months': cf.months,
                'beginning_balance': cf.beginning_balance,
                'principal': cf.total_principal,
                'interest': cf.interest,
                'ending_balance': cf.ending_balance,
                'accrued_interest': np.zeros(len(cf.months)),
                'name': 'POOL',
                'wal': cf.wal,
            })(), price
        )

        result = {
            "price": round(price, 4),
            "yield": round(y, 3) if y else 0,
            "wal_years": round(cf.wal, 2),
            "duration_months": cf.duration_months,
            "psa_speed": psa,
            "total_principal": round(cf.total_principal_paid, 2),
            "total_interest": round(cf.total_interest, 2),
            "dollar_price": round(price / 100 * inp["balance"], 2),
        }
        return json.dumps(result, indent=2)

    def _tool_structure_sequential(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id=f"{inp['deal_id']}_COLL",
            agency=AgencyType.GNMA,
            collateral_type=CollateralType.G2,
            coupon=inp["collateral_coupon"],
            wac=inp["collateral_wac"],
            wam=inp["collateral_wam"],
            original_balance=inp["collateral_balance"],
            current_balance=inp["collateral_balance"],
        )

        psa = inp.get("psa_speed", 150)
        cf = project_pool_cashflows(pool, psa_speed=psa)

        names = inp.get("tranche_names") or [f"A{i+1}" for i in range(len(inp["tranche_sizes"]))]
        deal_cf = create_sequential_cmo(
            deal_id=inp["deal_id"],
            collateral_flows=cf,
            tranche_sizes=inp["tranche_sizes"],
            tranche_coupons=inp["tranche_coupons"],
            tranche_names=names,
            collateral_coupon=inp["collateral_coupon"],
        )

        self.deals[inp["deal_id"]] = {
            "deal_cf": deal_cf,
            "collateral_cf": cf,
            "pool": pool,
            "psa_speed": psa,
        }

        summary = deal_cf.summary()
        for name, tcf in deal_cf.tranche_flows.items():
            summary["tranches"][name]["first_pay"] = tcf.first_pay_month
            summary["tranches"][name]["last_pay"] = tcf.last_pay_month

        summary["collateral"] = {
            "balance": inp["collateral_balance"],
            "coupon": inp["collateral_coupon"],
            "wac": inp["collateral_wac"],
            "wam": inp["collateral_wam"],
            "psa_speed": psa,
            "wal_years": round(cf.wal, 2),
        }

        return json.dumps(summary, indent=2)

    def _tool_structure_pac(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id=f"{inp['deal_id']}_COLL",
            agency=AgencyType.GNMA,
            collateral_type=CollateralType.G2,
            coupon=inp["collateral_coupon"],
            wac=inp["collateral_wac"],
            wam=inp["collateral_wam"],
            original_balance=inp["collateral_balance"],
            current_balance=inp["collateral_balance"],
        )

        psa = inp.get("psa_speed", 150)
        cf = project_pool_cashflows(pool, psa_speed=psa)

        deal_cf = create_pac_support_cmo(
            deal_id=inp["deal_id"],
            collateral_flows=cf,
            pac_balance=inp["pac_balance"],
            pac_coupon=inp["pac_coupon"],
            support_balance=inp["support_balance"],
            support_coupon=inp["support_coupon"],
            pac_lower=inp.get("pac_lower_band", 100),
            pac_upper=inp.get("pac_upper_band", 300),
            collateral_coupon=inp["collateral_coupon"],
            z_bond_balance=inp.get("z_bond_balance", 0),
            z_bond_coupon=inp.get("z_bond_coupon", 0),
            io_notional=inp.get("io_notional", 0),
            io_coupon=inp.get("io_coupon", 0),
        )

        self.deals[inp["deal_id"]] = {
            "deal_cf": deal_cf,
            "collateral_cf": cf,
            "pool": pool,
            "psa_speed": psa,
        }

        summary = deal_cf.summary()
        for name, tcf in deal_cf.tranche_flows.items():
            summary["tranches"][name]["first_pay"] = tcf.first_pay_month
            summary["tranches"][name]["last_pay"] = tcf.last_pay_month

        return json.dumps(summary, indent=2)

    def _tool_price_deal(self, inp: dict) -> str:
        deal_id = inp["deal_id"]
        if deal_id not in self.deals:
            return json.dumps({"error": f"Deal {deal_id} not found. Structure a deal first."})

        deal_data = self.deals[deal_id]
        deal_cf = deal_data["deal_cf"]
        spreads = inp.get("spreads", {})

        results = price_deal(deal_cf, self.curve, spreads)

        output = {"deal_id": deal_id, "tranches": {}}
        for name, pr in results.items():
            output["tranches"][name] = pr.to_dict()

        return json.dumps(output, indent=2)

    def _tool_calc_profit(self, inp: dict) -> str:
        cost_price = inp["collateral_cost_price"]
        balance = inp["collateral_balance"]
        collateral_cost = cost_price / 100.0 * balance

        tranche_proceeds = {}
        for name, price in inp["tranche_prices"].items():
            tranche_bal = inp["tranche_balances"].get(name, 0)
            tranche_proceeds[name] = price / 100.0 * tranche_bal

        result = structuring_profit(collateral_cost, tranche_proceeds, balance)
        return json.dumps(result, indent=2)

    def _tool_analyze_remic(self, inp: dict) -> str:
        data_dir = inp.get("data_dir", "CMO REMIC")
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_path, data_dir)

        if not os.path.exists(full_path):
            return json.dumps({"error": f"Data directory not found: {full_path}"})

        max_files = inp.get("max_files", 5)
        files = sorted(os.listdir(full_path))
        xlsx_files = [f for f in files if f.endswith('.xlsx')][-max_files:]

        all_deals = []
        for f in xlsx_files:
            try:
                from .remic_loader import parse_remic_xlsx
                deals = parse_remic_xlsx(os.path.join(full_path, f))
                all_deals.extend(deals)
            except Exception as e:
                continue

        trends = analyze_remic_trends(all_deals)
        trends["files_parsed"] = len(xlsx_files)
        trends["sample_deals"] = [d.to_dict() for d in all_deals[:5]]

        return json.dumps(trends, indent=2, default=str)

    def _tool_scenario_analysis(self, inp: dict) -> str:
        deal_id = inp["deal_id"]
        if deal_id not in self.deals:
            return json.dumps({"error": f"Deal {deal_id} not found."})

        deal_data = self.deals[deal_id]
        pool = deal_data["pool"]
        psa_base = deal_data["psa_speed"]

        rate_shifts = inp.get("rate_shifts_bps", [-100, -50, 0, 50, 100])
        psa_speeds = inp.get("psa_speeds", [75, 100, 150, 200, 300])

        results = {"deal_id": deal_id, "scenarios": []}

        for shift in rate_shifts:
            shifted_curve = self.curve.shift(shift)
            for psa in psa_speeds:
                cf = project_pool_cashflows(pool, psa_speed=psa)

                # Re-structure with same specs
                deal_cf_orig = deal_data["deal_cf"]
                tranche_specs = []
                for name, tcf in deal_cf_orig.tranche_flows.items():
                    tranche_specs.append(TrancheSpec(
                        name=name,
                        principal_type=PrincipalType.SEQUENTIAL,
                        interest_type=InterestType.FIXED,
                        original_balance=tcf.beginning_balance[0],
                        coupon=tcf.interest[0] / tcf.beginning_balance[0] * 12 * 100 if tcf.beginning_balance[0] > 0 else 0,
                        priority=len(tranche_specs),
                    ))

                deal_cf = structure_cmo(deal_id, cf, tranche_specs, pool.coupon)
                pricing = price_deal(deal_cf, shifted_curve)

                scenario = {
                    "rate_shift_bps": shift,
                    "psa_speed": psa,
                    "tranches": {},
                }
                for name, pr in pricing.items():
                    scenario["tranches"][name] = {
                        "price": round(pr.price, 3),
                        "yield": round(pr.yield_pct, 3),
                        "wal": round(pr.wal_years, 2),
                        "duration": round(pr.mod_duration, 3),
                    }
                results["scenarios"].append(scenario)

        return json.dumps(results, indent=2)

    def _tool_market_snapshot(self, inp: dict) -> str:
        result = {
            "date": str(date.today()),
            "us_treasury_yields": {
                "1M": 3.699, "3M": 3.690, "6M": 3.640,
                "1Y": 3.460, "2Y": 3.506, "3Y": 3.579,
                "5Y": 3.769, "7Y": 3.988, "10Y": 4.221,
                "20Y": 4.812, "30Y": 4.867,
            },
            "curve_shape": {
                "2s10s_bps": round((4.221 - 3.506) * 100, 1),
                "2s30s_bps": round((4.867 - 3.506) * 100, 1),
                "slope": "positively sloped (normal)",
            },
            "fed_funds": {
                "current_target": "3.50-3.75%",
                "march_meeting_prob_325_350": "17.7%",
                "march_meeting_prob_350_375": "82.3%",
            },
            "mortgage_rates": {
                "estimated_30yr_fixed": round(self.current_mortgage_rate, 3),
                "spread_to_10y_bps": 170,
            },
            "mbs_market": {
                "gnma_2_5_5_tba_price_est": round(price_tba(5.5, 4.221, AgencyType.GNMA), 3),
                "gnma_2_6_0_tba_price_est": round(price_tba(6.0, 4.221, AgencyType.GNMA), 3),
                "gnma_2_5_0_tba_price_est": round(price_tba(5.0, 4.221, AgencyType.GNMA), 3),
                "fnma_5_5_tba_price_est": round(price_tba(5.5, 4.221, AgencyType.FNMA), 3),
                "fnma_6_0_tba_price_est": round(price_tba(6.0, 4.221, AgencyType.FNMA), 3),
            },
            "equity_indices": {
                "dow": 50115.67,
                "sp500": 6932.30,
                "nasdaq": 23031.21,
                "vix": 17.76,
            },
            "dollar_index": 97.480,
        }
        return json.dumps(result, indent=2)

    def _tool_get_fed_rates(self, inp: dict) -> str:
        rate_type = inp.get("rate_type", "all")
        n = inp.get("n", 5)
        if rate_type == "all":
            data = get_latest_rates()
        elif rate_type == "sofr":
            data = get_sofr_latest(n)
        elif rate_type == "effr":
            data = get_effr_latest(n)
        else:
            data = get_latest_rates()

        # Extract key rates
        sofr = extract_sofr_rate(data)
        effr = extract_effr_rate(data)
        result = {"raw_data": data}
        if sofr is not None:
            result["sofr"] = sofr
        if effr is not None:
            result["effr"] = effr
        return json.dumps(result, indent=2, default=str)

    def _tool_get_soma_holdings(self, inp: dict) -> str:
        holding_type = inp.get("holding_type", "summary")
        if holding_type == "summary":
            data = get_soma_summary()
        elif holding_type == "mbs":
            from .fed_api import get_soma_mbs_holdings
            data = get_soma_mbs_holdings()
        elif holding_type == "agency":
            from .fed_api import get_soma_agency_holdings
            data = get_soma_agency_holdings()
        elif holding_type == "treasury":
            from .fed_api import get_soma_treasury_holdings
            data = get_soma_treasury_holdings()
        else:
            data = get_soma_summary()
        return json.dumps(data, indent=2, default=str)

    def _tool_get_ambs_operations(self, inp: dict) -> str:
        n = inp.get("n", 5)
        data = get_ambs_results_last(n)
        return json.dumps(data, indent=2, default=str)

    def _tool_analyze_dollar_roll(self, inp: dict) -> str:
        result = analyze_dollar_roll(
            coupon=inp["coupon"],
            front_price=inp["front_price"],
            back_price=inp["back_price"],
            wam=inp.get("wam", 357),
            psa_speed=inp.get("psa_speed", 150),
            repo_rate=inp.get("repo_rate", 0.043),
        )
        return json.dumps({
            "drop": result.drop,
            "drop_32nds": result.drop_32nds,
            "implied_financing_rate": result.implied_financing_rate,
            "coupon_income": result.coupon_income,
            "breakeven_speed": result.breakeven_speed,
            "roll_advantage_bps": result.roll_advantage_bps,
            "recommendation": result.recommendation,
        }, indent=2)

    def _tool_key_rate_durations(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id="KRD", agency=AgencyType[inp.get("agency", "GNMA")],
            collateral_type=CollateralType.G2,
            coupon=inp["coupon"], wac=inp["wac"], wam=inp["wam"],
            wala=inp.get("wala", 0),
            original_balance=inp["balance"], current_balance=inp["balance"],
        )
        result = key_rate_durations(
            pool, self.curve, psa_speed=inp.get("psa_speed", 150),
            shift_bps=inp.get("shift_bps", 25),
        )
        return json.dumps(result, indent=2)

    def _tool_risk_metrics(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id="RISK", agency=AgencyType[inp.get("agency", "GNMA")],
            collateral_type=CollateralType.G2,
            coupon=inp["coupon"], wac=inp["wac"], wam=inp["wam"],
            wala=inp.get("wala", 0),
            original_balance=inp["balance"], current_balance=inp["balance"],
        )
        metrics = compute_full_risk_metrics(
            pool, self.curve, psa_speed=inp.get("psa_speed", 150),
        )
        return json.dumps({
            "price": metrics.price,
            "eff_duration": metrics.eff_duration,
            "eff_convexity": metrics.eff_convexity,
            "mod_duration": metrics.mod_duration,
            "spread_duration": metrics.spread_duration,
            "dv01": metrics.dv01,
            "spread_dv01": metrics.spread_dv01,
            "wal_years": metrics.wal_years,
            "wal_up_100": metrics.wal_up,
            "wal_down_100": metrics.wal_down,
            "price_up_100": metrics.price_up_100,
            "price_down_100": metrics.price_down_100,
            "price_up_50": metrics.price_up_50,
            "price_down_50": metrics.price_down_50,
            "negative_convexity": bool(metrics.negative_convexity),
        }, indent=2)

    def _tool_scenario_matrix(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id="SCEN", agency=AgencyType[inp.get("agency", "GNMA")],
            collateral_type=CollateralType.G2,
            coupon=inp["coupon"], wac=inp["wac"], wam=inp["wam"],
            wala=inp.get("wala", 0),
            original_balance=inp["balance"], current_balance=inp["balance"],
        )
        result = scenario_matrix(
            pool, self.curve,
            rate_shifts=inp.get("rate_shifts"),
            psa_speeds=inp.get("psa_speeds"),
        )
        return json.dumps(result, indent=2)

    def _tool_monte_carlo_oas(self, inp: dict) -> str:
        config = MonteCarloConfig(
            n_paths=inp.get("n_paths", 500),
            n_months=inp.get("wam", 360),
        )
        result = compute_oas(
            market_price=inp["market_price"],
            par_balance=inp.get("balance", 1_000_000),
            wac=inp["wac"],
            coupon=inp["coupon"],
            wam=inp.get("wam", 360),
            wala=inp.get("wala", 0),
            curve=self.curve,
            config=config,
        )
        return json.dumps(result, indent=2)

    def _tool_price_crt_bond(self, inp: dict) -> str:
        bond = CRTBondSpec(
            name=inp["bond_name"],
            attachment_point=inp["attachment_pct"] / 100,
            detachment_point=inp["detachment_pct"] / 100,
            coupon_spread_bps=inp["coupon_spread_bps"],
        )
        result = price_crt_bond(
            bond,
            expected_cumulative_loss=inp["expected_loss_pct"] / 100,
            loss_vol=inp.get("loss_vol", 0.5),
        )
        return json.dumps(result, indent=2)

    def _tool_structure_nonagency(self, inp: dict) -> str:
        scenario_map = {
            "base": EconomicScenario(name="base"),
            "mild_recession": EconomicScenario(
                name="mild_recession", unemployment_rate=6.0,
                unemployment_change=2.0, hpa_annual=-2.0),
            "severe_recession": EconomicScenario(
                name="severe_recession", unemployment_rate=9.0,
                unemployment_change=5.0, hpa_annual=-10.0),
        }
        scenario = scenario_map.get(inp.get("scenario", "base"), EconomicScenario())
        result = structure_nonagency_cmo(
            deal_name=inp["deal_name"],
            collateral_balance=inp["collateral_balance"],
            collateral_wac=inp["collateral_wac"],
            collateral_avg_fico=inp.get("avg_fico", 720),
            collateral_avg_ltv=inp.get("avg_ltv", 80),
            target_senior_pct=inp.get("senior_pct", 0.80),
            scenario=scenario,
        )
        return json.dumps({
            "deal_name": result.deal_name,
            "collateral_balance": result.collateral_balance,
            "tranches": result.tranches,
            "credit_enhancement": result.credit_enhancement,
            "expected_losses": result.expected_losses,
            "waterfall": result.waterfall_summary,
        }, indent=2)

    def _tool_credit_stress_test(self, inp: dict) -> str:
        results = stress_test_credit(
            collateral_avg_fico=inp.get("avg_fico", 720),
            collateral_avg_ltv=inp.get("avg_ltv", 85),
            collateral_wac=inp.get("collateral_wac", 6.0),
        )
        return json.dumps(results, indent=2)

    def _tool_s_curve(self, inp: dict) -> str:
        points = compute_s_curve(
            wac=inp["wac"],
            wala=inp.get("wala", 24),
            fico=inp.get("fico", 740),
            burnout_factor=inp.get("burnout_factor", 1.0),
        )
        # Return a selection of points (every 3rd to keep response size manageable)
        selected = [
            {"incentive_bps": p.incentive_bps, "cpr": p.cpr, "psa": p.psa_speed,
             "turnover": p.turnover, "refi": p.refi}
            for p in points[::3]
        ]
        return json.dumps({"s_curve": selected, "wac": inp["wac"]}, indent=2)

    def _tool_relative_value(self, inp: dict) -> str:
        mortgage_rate = inp["mortgage_rate"]
        coupons = inp.get("coupons") or [
            r / 200 for r in range(int(mortgage_rate * 200) - 6, int(mortgage_rate * 200) + 4)
        ]
        if not coupons:
            coupons = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
        results = relative_value_grid(
            coupons=coupons,
            curve=self.curve,
            mortgage_rate=mortgage_rate,
        )
        return json.dumps(results, indent=2)

    def _tool_stanton_prepayment(self, inp: dict) -> str:
        model = StantonPrepaymentModel()
        wac = inp["wac"]
        r0 = inp["short_rate"]
        n_months = inp.get("n_months", 120)
        loan_age = inp.get("loan_age", 0)

        if inp.get("use_monte_carlo", False):
            result = model.monte_carlo_prepayment(
                wac=wac, r0=r0, n_months=n_months,
                n_paths=inp.get("n_paths", 200),
                loan_age=loan_age,
            )
            return json.dumps({
                "model": "Stanton (1995) MC",
                "avg_wal_years": result["avg_wal_years"],
                "wal_std": result["wal_std_years"],
                "wal_range": f"{result['wal_5th_pctile']:.2f} - {result['wal_95th_pctile']:.2f}",
                "n_paths": result["n_paths"],
                "params": result["params"],
            }, indent=2)
        else:
            result = model.project_schedule(
                n_months=n_months, wac=wac,
                initial_short_rate=r0, loan_age=loan_age,
            )
            return json.dumps({
                "model": "Stanton (1995)",
                "avg_cpr_year1": round(result["avg_cpr_year1"] * 100, 2),
                "avg_cpr_year3": round(result["avg_cpr_year3"] * 100, 2),
                "avg_cpr_year5": round(result["avg_cpr_year5"] * 100, 2),
                "terminal_pool_factor": round(result["terminal_pool_factor"], 4),
                "params": {
                    "rho": model.params.rho,
                    "lambda": model.params.lam,
                    "alpha": model.params.alpha,
                    "beta": model.params.beta_param,
                },
                "note": "Endogenous burnout via heterogeneous transaction costs",
            }, indent=2)

    def _tool_structure_advanced(self, inp: dict) -> str:
        pool = SpecPool(
            pool_id=f"{inp['deal_id']}_COLL",
            agency=AgencyType.GNMA,
            collateral_type=CollateralType.G2,
            coupon=inp["collateral_coupon"],
            wac=inp["collateral_wac"],
            wam=inp["collateral_wam"],
            original_balance=inp["collateral_balance"],
            current_balance=inp["collateral_balance"],
        )
        psa = inp.get("psa_speed", 150)
        cf = project_pool_cashflows(pool, psa_speed=psa)
        stype = inp["structure_type"]

        if stype == "vadm":
            deal_cf = create_vadm_z_structure(
                inp["deal_id"], cf,
                seq_balances=inp.get("seq_balances", [inp["collateral_balance"] * 0.4]),
                seq_coupons=inp.get("seq_coupons", [inp["collateral_coupon"] - 0.25]),
                vadm_balance=inp.get("vadm_balance", inp["collateral_balance"] * 0.15),
                vadm_coupon=inp.get("vadm_coupon", inp["collateral_coupon"] - 0.5),
                z_bond_balance=inp.get("z_bond_balance", inp["collateral_balance"] * 0.35),
                z_bond_coupon=inp.get("z_bond_coupon", inp["collateral_coupon"] + 0.5),
                collateral_coupon=inp["collateral_coupon"],
            )
        elif stype == "jump_z":
            deal_cf = create_pac_jump_z_structure(
                inp["deal_id"], cf,
                pac_balance=inp.get("pac_balance", inp["collateral_balance"] * 0.50),
                pac_coupon=inp.get("pac_coupon", inp["collateral_coupon"] - 0.25),
                support_balance=inp.get("support_balance", inp["collateral_balance"] * 0.25),
                support_coupon=inp.get("support_coupon", inp["collateral_coupon"]),
                jump_z_balance=inp.get("z_bond_balance", inp["collateral_balance"] * 0.25),
                jump_z_coupon=inp.get("z_bond_coupon", inp["collateral_coupon"] + 0.5),
                pac_lower=inp.get("pac_lower", 100),
                pac_upper=inp.get("pac_upper", 300),
                sticky=inp.get("sticky", True),
                collateral_coupon=inp["collateral_coupon"],
            )
        elif stype == "pac_ii":
            deal_cf = create_pac_ii_structure(
                inp["deal_id"], cf,
                pac_i_balance=inp.get("pac_i_balance", inp["collateral_balance"] * 0.40),
                pac_i_coupon=inp.get("pac_i_coupon", inp["collateral_coupon"] - 0.25),
                pac_ii_balance=inp.get("pac_ii_balance", inp["collateral_balance"] * 0.20),
                pac_ii_coupon=inp.get("pac_ii_coupon", inp["collateral_coupon"]),
                support_balance=inp.get("support_balance", inp["collateral_balance"] * 0.40),
                support_coupon=inp.get("support_coupon", inp["collateral_coupon"] + 0.25),
                pac_i_lower=inp.get("pac_lower", 100),
                pac_i_upper=inp.get("pac_upper", 300),
                pac_ii_lower=inp.get("pac_ii_lower", 150),
                pac_ii_upper=inp.get("pac_ii_upper", 250),
                collateral_coupon=inp["collateral_coupon"],
            )
        elif stype == "nas":
            deal_cf = create_nas_structure(
                inp["deal_id"], cf,
                nas_balance=inp.get("nas_balance", inp["collateral_balance"] * 0.30),
                nas_coupon=inp.get("nas_coupon", inp["collateral_coupon"]),
                nas_lockout_months=inp.get("lockout_months", 36),
                sequential_balance=inp.get("sequential_balance", inp["collateral_balance"] * 0.40),
                sequential_coupon=inp.get("sequential_coupon", inp["collateral_coupon"] - 0.25),
                support_balance=inp.get("support_balance", inp["collateral_balance"] * 0.30),
                support_coupon=inp.get("support_coupon", inp["collateral_coupon"] + 0.25),
                collateral_coupon=inp["collateral_coupon"],
            )
        elif stype == "schedule":
            deal_cf = create_schedule_bond_structure(
                inp["deal_id"], cf,
                schedule_balance=inp.get("schedule_balance", inp["collateral_balance"] * 0.60),
                schedule_coupon=inp.get("schedule_coupon", inp["collateral_coupon"] - 0.25),
                target_psa=inp.get("target_psa", 165),
                support_balance=inp.get("support_balance", inp["collateral_balance"] * 0.40),
                support_coupon=inp.get("support_coupon", inp["collateral_coupon"] + 0.25),
                collateral_coupon=inp["collateral_coupon"],
            )
        elif stype == "z_pac":
            deal_cf = create_z_pac_structure(
                inp["deal_id"], cf,
                seq_balance=inp.get("sequential_balance", inp["collateral_balance"] * 0.40),
                seq_coupon=inp.get("sequential_coupon", inp["collateral_coupon"] - 0.25),
                z_pac_balance=inp.get("z_pac_balance", inp["collateral_balance"] * 0.30),
                z_pac_coupon=inp.get("z_pac_coupon", inp["collateral_coupon"]),
                z_pac_lockout=inp.get("lockout_months", 48),
                support_balance=inp.get("support_balance", inp["collateral_balance"] * 0.30),
                support_coupon=inp.get("support_coupon", inp["collateral_coupon"] + 0.25),
                pac_lower=inp.get("pac_lower", 100),
                pac_upper=inp.get("pac_upper", 300),
                collateral_coupon=inp["collateral_coupon"],
            )
        elif stype == "kitchen_sink":
            deal_cf = create_kitchen_sink_structure(
                inp["deal_id"], cf,
                collateral_coupon=inp["collateral_coupon"],
            )
        else:
            return json.dumps({"error": f"Unknown structure type: {stype}"})

        self.deals[inp["deal_id"]] = {
            "deal_cf": deal_cf,
            "collateral_cf": cf,
            "pool": pool,
            "psa_speed": psa,
        }

        summary = deal_cf.summary()
        for name, tcf in deal_cf.tranche_flows.items():
            summary["tranches"][name]["first_pay"] = tcf.first_pay_month
            summary["tranches"][name]["last_pay"] = tcf.last_pay_month

        summary["structure_type"] = stype
        summary["collateral"] = {
            "balance": inp["collateral_balance"],
            "coupon": inp["collateral_coupon"],
            "wac": inp["collateral_wac"],
            "wam": inp["collateral_wam"],
            "psa_speed": psa,
            "wal_years": round(cf.wal, 2),
        }

        return json.dumps(summary, indent=2)

    def _tool_disguised_default(self, inp: dict) -> str:
        result = compute_disguised_default_adjustment(
            current_ltv=inp["current_ltv"],
            unemployment_change=inp.get("unemployment_change", 0),
            hpa_annual=inp.get("hpa_annual", 3.0),
            loan_age_months=inp.get("loan_age_months", 24),
        )
        return json.dumps({
            "model": "Hall & Maingi (2021)",
            "current_ltv": result["current_ltv"],
            "cpr_adjustment_pct": round(result["cpr_adjustment"] * 100, 2),
            "cdr_adjustment_pct": round(result["cdr_adjustment"] * 100, 2),
            "liquidity_shock_prob": round(result["liquidity_shock_prob"] * 100, 2),
            "channel": result["channel"],
            "note": "Dual trigger: liquidity shock + equity determines prepay vs default",
        }, indent=2)

    # ─── Data Source Tool Implementations ────────────────────────────────────

    def _tool_fetch_live_curve(self, inp: dict) -> str:
        as_of = inp.get("as_of_date")
        result = build_live_treasury_curve(as_of)

        if inp.get("update_agent_curve", True) and result.get("yields"):
            from .yield_curve import YieldCurvePoint
            points = [
                YieldCurvePoint(float(m), float(y))
                for m, y in result["yields"].items()
            ]
            self.curve = YieldCurve(as_of_date=date.today(), points=points)
            # Update dependent rates
            self.treasury_10y = result["yields"].get(10.0, self.treasury_10y)
            self.current_mortgage_rate = mortgage_rate_from_treasury(self.treasury_10y)

        formatted = {
            "source": result.get("source", "unknown"),
            "date": result.get("date"),
            "series_count": result.get("series_count", 0),
            "yields": {
                f"{m}Y": round(y, 3)
                for m, y in sorted(result.get("yields", {}).items())
            },
            "agent_curve_updated": inp.get("update_agent_curve", True),
        }
        return json.dumps(formatted, indent=2)

    def _tool_get_mortgage_rates(self, inp: dict) -> str:
        product = inp.get("product", "all")
        history_start = inp.get("history_start")

        result = {}
        if product == "all" or not history_start:
            result["current"] = fetch_current_mortgage_rates()

        if history_start:
            products = ["30yr", "15yr", "5yr_arm"] if product == "all" else [product]
            result["history"] = {}
            for p in products:
                data = fetch_mortgage_rate_history(p, start_date=history_start)
                result["history"][p] = {
                    "count": len(data),
                    "latest": data[0] if data else None,
                    "oldest": data[-1] if data else None,
                    "data": data[:50],  # Cap to avoid huge responses
                }

        return json.dumps(result, indent=2)

    def _tool_comprehensive_market(self, inp: dict) -> str:
        snapshot = get_full_market_snapshot()

        if inp.get("include_housing"):
            try:
                hpi = fetch_fhfa_hpi_national()
                snapshot["housing"] = {
                    "fhfa_hpi": hpi[:10] if hpi else [],
                    "count": len(hpi),
                }
            except Exception as e:
                snapshot["housing"] = {"error": str(e)}

            try:
                cs = fred_fetch_series("CSUSHPINSA", limit=12)
                snapshot["case_shiller"] = cs
            except Exception:
                pass

        if inp.get("include_spreads"):
            try:
                t10y2y = fred_fetch_series("T10Y2Y", limit=20)
                t10y3m = fred_fetch_series("T10Y3M", limit=20)
                snapshot["spreads"] = {
                    "10y_2y": t10y2y[:5],
                    "10y_3m": t10y3m[:5],
                }
            except Exception:
                pass

        return json.dumps(snapshot, indent=2, default=str)

    def _tool_check_data_sources(self, inp: dict) -> str:
        statuses = check_all_sources()
        result = []
        for s in statuses:
            result.append({
                "source": s.name,
                "available": s.available,
                "last_fetch": s.last_fetch,
                "record_count": s.record_count,
                "error": s.error,
            })
        return json.dumps({"data_sources": result}, indent=2)

    def _tool_fred_lookup(self, inp: dict) -> str:
        action = inp["action"]
        query = inp["query"]

        if action == "search":
            results = fred_search(query, limit=inp.get("limit", 20))
            return json.dumps({"search_results": results}, indent=2)
        elif action == "fetch":
            data = fred_fetch_series(
                query,
                start_date=inp.get("start_date"),
                limit=inp.get("limit", 100),
            )
            return json.dumps({
                "series_id": query,
                "count": len(data),
                "data": data[:100],
            }, indent=2)
        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    def _tool_refi_incentive(self, inp: dict) -> str:
        result = compute_refi_incentive(
            pool_wac=inp["pool_wac"],
            current_30yr_rate=inp.get("current_rate"),
        )
        return json.dumps(result, indent=2)

    # ─── RL-Powered Tool Implementations ─────────────────────────────────────

    def _tool_rl_suggest(self, inp: dict) -> str:
        """Use trained RL policy to suggest optimal CMO structures."""
        model_path = inp.get("model_path", "models/gse_shelf_200k.pt")
        balance = inp.get("collateral_balance", 100_000_000)
        n_suggestions = inp.get("n_suggestions", 3)

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(base_path, model_path)
        if not os.path.exists(full_model_path):
            return json.dumps({"error": f"Model not found: {full_model_path}"})

        env = make_yield_book_env(mode="AGENCY", collateral_balance=balance, seed=42)
        policy = load_policy(full_model_path)
        policy_obs_dim = policy.shared[0].in_features

        suggestions = []
        for i in range(n_suggestions):
            obs, info = env.reset(seed=42 + i * 100)
            total_reward = 0.0
            actions_taken = []
            done = False
            step = 0

            while not done:
                import torch
                adapted = _adapt_obs(obs, policy_obs_dim)
                obs_tensor = torch.tensor(adapted, dtype=torch.float32)
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                actions_taken.append(action)
                done = terminated or truncated
                step += 1

            # Extract deal details
            deal_info = {
                "suggestion": i + 1,
                "total_reward_ticks": round(total_reward, 2),
                "n_tranches": info.get("n_tranches", 0),
                "allocation_pct": info.get("allocation_pct", 0),
                "regime": info.get("regime", ""),
                "actions": [list(a) for a in actions_taken],
                "collateral_balance": balance,
            }

            # Decode actions into human-readable structure
            from .yield_book_env import ActionType
            _action_names = {
                0: "Sequential", 1: "PAC", 2: "TAC", 3: "Support",
                4: "Z-bond", 5: "IO strip", 6: "PO strip",
                7: "Floater", 8: "Inverse Floater",
                9: "Modify Size", 10: "Modify Coupon",
                11: "Remove", 12: "Execute", 13: "NOOP",
            }
            tranches = []
            for a in actions_taken:
                at = a[0]
                if at <= 8:
                    size_pct = 2.0 + a[2] * (48.0 / 19)
                    coupon_offset = -2.0 + a[3] * (3.0 / 19)
                    tranches.append({
                        "type": _action_names.get(at, f"Action_{at}"),
                        "size_pct": round(size_pct, 1),
                        "coupon_offset": round(coupon_offset, 2),
                    })
            deal_info["structure"] = tranches
            suggestions.append(deal_info)

        return json.dumps({
            "model": model_path,
            "n_suggestions": n_suggestions,
            "suggestions": suggestions,
        }, indent=2)

    def _tool_rl_backtest(self, inp: dict) -> str:
        """Run backtest of RL agent vs real dealers."""
        model_path = inp.get("model_path", "models/gse_shelf_200k.pt")
        n_deals = inp.get("n_deals", 100)

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(base_path, model_path)
        demo_file = os.path.join(base_path, "expert_demonstrations.json")

        if not os.path.exists(full_model_path):
            return json.dumps({"error": f"Model not found: {full_model_path}"})
        if not os.path.exists(demo_file):
            return json.dumps({"error": f"Demo file not found: {demo_file}"})

        summary = backtest_agent_vs_experts(
            model_path=full_model_path,
            demo_file=demo_file,
            n_demos=n_deals,
            verbose=False,
        )
        return json.dumps(backtest_to_json(summary), indent=2)

    def _tool_rl_live_deal(self, inp: dict) -> str:
        """Generate live deal recommendation with FRED rates + RL agent."""
        model_path = inp.get("model_path", "models/gse_shelf_200k.pt")
        balance = inp.get("collateral_balance", 100_000_000)

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(base_path, model_path)
        if not os.path.exists(full_model_path):
            return json.dumps({"error": f"Model not found: {full_model_path}"})

        # Fetch live market data
        live_data = {}
        try:
            curve_data = build_live_treasury_curve()
            live_data["treasury_curve"] = curve_data
            live_data["source"] = curve_data.get("source", "unknown")
        except Exception as e:
            live_data["curve_error"] = str(e)

        try:
            rates = fetch_current_mortgage_rates()
            live_data["mortgage_rates"] = rates
        except Exception as e:
            live_data["rates_error"] = str(e)

        # Run RL agent
        env = make_yield_book_env(mode="AGENCY", collateral_balance=balance, seed=42)
        policy = load_policy(full_model_path)
        policy_obs_dim = policy.shared[0].in_features

        obs, info = env.reset()
        total_reward = 0.0
        actions_taken = []
        done = False

        while not done:
            import torch
            adapted = _adapt_obs(obs, policy_obs_dim)
            obs_tensor = torch.tensor(adapted, dtype=torch.float32)
            action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions_taken.append(list(action))
            done = terminated or truncated

        # Build recommendation
        _action_names = {
            0: "Sequential", 1: "PAC", 2: "TAC", 3: "Support",
            4: "Z-bond", 5: "IO strip", 6: "PO strip",
            7: "Floater", 8: "Inverse Floater",
        }
        recommended_tranches = []
        for a in actions_taken:
            at = a[0]
            if at <= 8:
                size_pct = 2.0 + a[2] * (48.0 / 19)
                coupon_offset = -2.0 + a[3] * (3.0 / 19)
                recommended_tranches.append({
                    "type": _action_names.get(at, f"Action_{at}"),
                    "size_pct": round(size_pct, 1),
                    "estimated_size": round(balance * size_pct / 100, 0),
                    "coupon_offset_from_collateral": round(coupon_offset, 2),
                })

        result = {
            "recommendation": "LIVE DEAL STRUCTURE",
            "date": str(date.today()),
            "market_data": live_data,
            "collateral": {
                "balance": balance,
                "regime": info.get("regime", ""),
            },
            "structure": recommended_tranches,
            "estimated_profit_ticks": round(total_reward, 2),
            "n_tranches": len(recommended_tranches),
            "model_used": model_path,
            "actions_raw": actions_taken,
        }
        return json.dumps(result, indent=2)

    # ─── CGC 53601 Compliance Tool Handlers ─────────────────────────────────

    def _tool_cgc_check_security(self, inp: dict) -> str:
        """Check single security against CGC 53601."""
        result = cgc_quick_check(
            subdivision=inp.get("subdivision", ""),
            par_value=inp.get("par_value", 0),
            credit_rating=inp.get("credit_rating", ""),
            maturity_days=inp.get("maturity_days"),
            settlement_date=inp.get("settlement_date"),
            maturity_date=inp.get("maturity_date"),
            trade_date=inp.get("trade_date"),
            entity_assets=inp.get("entity_assets", 0),
            instrument_subtype=inp.get("instrument_subtype", ""),
        )
        return json.dumps(result, indent=2)

    def _tool_cgc_check_deal(self, inp: dict) -> str:
        """Check proposed CMO deal for CGC 53601 compliance."""
        tranches = inp.get("deal_tranches", [])
        entity = EntityProfile(
            name=inp.get("entity_name", "Local Agency"),
            total_investable_assets=inp.get("entity_assets", 0),
            has_extended_maturity_authority=inp.get("has_extended_maturity_authority", False),
        )

        # Build existing holdings stub for concentration context
        existing: list[SecurityHolding] = []
        existing_mv = inp.get("existing_portfolio_mv", 0)
        if existing_mv > 0:
            existing.append(SecurityHolding(
                name="_existing_portfolio",
                subdivision=CGCSubdivision.B_US_TREASURY,
                par_value=existing_mv,
                market_value=existing_mv,
            ))

        result = check_deal_compliance(tranches, entity, existing)
        return json.dumps(result.to_dict(), indent=2)

    def _tool_cgc_rules(self, inp: dict) -> str:
        """Return all CGC 53601 rules."""
        return json.dumps(cgc_get_all_rules(), indent=2)

    def _tool_cgc_classify(self, inp: dict) -> str:
        """Classify a CMO tranche under CGC 53601."""
        sub, prohibitions = classify_cmo_tranche_subdivision(
            inp.get("agency", ""),
            inp.get("principal_type", ""),
            inp.get("interest_type", ""),
        )
        from .compliance import SUBDIVISION_LABELS, PROHIBITED_LABELS, RULES
        rule = RULES[sub]
        return json.dumps({
            "subdivision": sub.value,
            "label": SUBDIVISION_LABELS[sub],
            "prohibited_flags": prohibitions,
            "prohibited_details": [PROHIBITED_LABELS.get(p, p) for p in prohibitions],
            "applicable_rules": {
                "max_maturity_days": rule.max_maturity_days,
                "max_portfolio_pct": f"{rule.max_portfolio_pct*100:.0f}%" if rule.max_portfolio_pct else "unlimited",
                "min_rating": rule.min_rating or "none",
                "notes": rule.notes,
            },
        }, indent=2)

    # ─── Ginnie Mae Participant Tool Handlers ────────────────────────────────

    def _tool_gnma_validate_sponsor(self, inp: dict) -> str:
        """Validate a dealer as approved Ginnie Mae REMIC sponsor."""
        result = validate_sponsor(
            dealer=inp.get("dealer", ""),
            program=inp.get("program", "all"),
        )
        return json.dumps(result, indent=2)

    def _tool_gnma_directory(self, inp: dict) -> str:
        """Return full Ginnie Mae participant directory."""
        return json.dumps(get_directory_summary(), indent=2)

    # ─── Agent Loop ──────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a message and get a response with tool use."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        full_response_text = []

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=self._system_prompt(),
                tools=TOOLS,
                messages=self.conversation_history,
            )

            # Collect text and tool use blocks
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    full_response_text.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "end_turn" or not tool_uses:
                # Done - add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })
                break

            # Process tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            tool_results = []
            for tool_use in tool_uses:
                result = self.execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            self.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

        return "\n".join(full_response_text)

    def run_autonomous(self, objective: str, max_iterations: int = 10) -> str:
        """
        Run the agent autonomously to achieve an objective.
        The agent will use tools iteratively until it reaches a conclusion.
        """
        prompt = f"""You are running autonomously to achieve the following objective:

{objective}

Work through this step by step:
1. First get a market snapshot to understand current conditions
2. Analyze the opportunity and identify the best approach
3. Create spec pools, structure a CMO deal, and price all tranches
4. Calculate the structuring profit
5. Provide a comprehensive recommendation with all key metrics

Execute all necessary tool calls to complete this analysis. Be thorough and quantitative."""

        return self.chat(prompt)


def main():
    """CLI entry point for the CMO agent."""
    print("=" * 70)
    print("  CMO Agent - Autonomous CMO Pricing & Structuring System")
    print("=" * 70)
    print()

    agent = CMOAgent()

    if len(sys.argv) > 1:
        # Run with provided objective
        objective = " ".join(sys.argv[1:])
        print(f"Objective: {objective}")
        print("-" * 70)
        result = agent.run_autonomous(objective)
        print(result)
        return

    # Interactive mode
    print("Commands:")
    print("  Type your question or request about CMO structuring")
    print("  'auto <objective>' - Run autonomously with an objective")
    print("  'market' - Get market snapshot")
    print("  'quit' - Exit")
    print()

    while True:
        try:
            user_input = input("CMO Agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "market":
            result = agent.execute_tool("market_snapshot", {})
            print(json.dumps(json.loads(result), indent=2))
            continue

        if user_input.lower().startswith("auto "):
            objective = user_input[5:]
            print(f"\nRunning autonomously: {objective}")
            print("-" * 70)
            result = agent.run_autonomous(objective)
            print(result)
            continue

        result = agent.chat(user_input)
        print(result)
        print()


if __name__ == "__main__":
    main()
