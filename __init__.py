"""CMO Agent - Autonomous CMO Pricing and Structuring System"""
__version__ = "0.6.0"

from .yield_curve import YieldCurve, YieldCurvePoint, build_us_treasury_curve
from .prepayment import (
    PrepaymentModel, PrepaymentModelConfig, estimate_psa_speed,
    StantonPrepaymentModel, StantonParams,
    compute_disguised_default_adjustment, DisguisedDefaultParams,
    SEASONAL_FACTORS, CIRT_SEASONING_CPR, CIRT_SEASONING_CDR,
    CAS_SEASONING_CPR, CAS_SEASONING_CDR,
)
from .spec_pool import SpecPool, AgencyType, CollateralType, project_pool_cashflows
from .cmo_structure import (
    TrancheSpec, PrincipalType, InterestType,
    structure_cmo, create_sequential_cmo, create_pac_support_cmo,
    create_vadm_z_structure, create_pac_jump_z_structure,
    create_pac_ii_structure, create_nas_structure,
    create_schedule_bond_structure, create_z_pac_structure,
    create_kitchen_sink_structure, create_floater_inverse_structure,
    CMODeal, CMOCashFlows,
    optimal_pac_fraction,
)
from .pricing import price_tranche, price_deal, structuring_profit, PricingResult
from .tba import TBAPriceGrid, TBAContract, analyze_dollar_roll, build_tba_price_grid
from .risk import key_rate_durations, compute_full_risk_metrics, scenario_matrix
from .monte_carlo import compute_oas, MonteCarloConfig, HullWhiteParams
from .credit import DoubleTriggerModel, price_crt_bond, structure_nonagency_cmo, STATE_CDR_MULTIPLIER
from .market_simulator import (
    MarketSimulator, MarketScenario,
    song_zhu_specialness, specialness_to_roll_income_ticks,
)
from .analytics import compute_s_curve, relative_value_grid, burnout_analysis
from .calibration import (
    FullCalibration, PrepaymentCalibration, CreditCalibration, RateModelCalibration,
    run_full_calibration, calibrate_rate_model_from_fred, apply_calibration,
)
from .data_sources import (
    build_live_treasury_curve, fetch_treasury_curve_xml,
    fetch_current_mortgage_rates, fetch_mortgage_rate_history,
    fetch_fhfa_hpi_national, compute_refi_incentive,
    get_full_market_snapshot, check_all_sources,
    fred_fetch_series, fred_search,
    LoanLevelDataset, LoanRecord,
)
from .deal_economics import (
    DemandCurve, DealPnL, StructuralValidation, DEMAND_CURVES,
    market_clearing_spread, validate_deal_structure,
    compute_io_value_ticks, compute_investor_surplus_ticks, compute_deal_pnl,
)
from .yield_book_env import YieldBookEnv, CMBSYieldBookEnv, make_yield_book_env
from .train import (
    PPOTrainer, PPOConfig, PolicyNetwork, ExpertDemoBuffer,
    RandomAgent, HeuristicAgent, GSEShelfAgent, FullDeskAgent,
    train_ppo, train_gse_shelf_agent, train_with_imitation,
    train_full_desk_agent, benchmark_agents,
)
from .client_demand import (
    ClientType, ClientDemand, DifficultyLevel,
    generate_client_demand, generate_client_scenario, compute_match_score,
)
from .multi_client_env import MultiClientYieldBookEnv, make_multi_client_env, MatchResult
from .train_multi_client import (
    MultiClientHeuristicAgent, CurriculumStage,
    train_multi_client_agent, benchmark_multi_client_agents,
)
from .data_pipeline import (
    parse_cas_loans, process_cas_data, CASLoan, CASProcessingResult,
    process_remic_deals, ExpertDemonstration,
    recalibrate_from_cas, run_full_pipeline,
)
from .backtest import (
    backtest_agent_vs_experts, BacktestSummary, DealBacktestResult,
    backtest_to_json, load_policy, replay_expert_demo, replay_agent_policy,
)
from .compliance import (
    CGCSubdivision, SecurityHolding, EntityProfile,
    ComplianceResult, Violation, SubdivisionRule,
    check_security, check_portfolio, check_deal_compliance,
    classify_cmo_tranche_subdivision,
    quick_check as cgc_quick_check, get_all_rules as cgc_get_all_rules,
    RULES as CGC_RULES, PROHIBITED_TYPES, PROHIBITED_LABELS,
)
from .ginnie_mae_participants import (
    Participant as GNMAParticipant, ParticipantDirectory,
    validate_sponsor as gnma_validate_sponsor,
    get_directory_summary as gnma_get_directory,
    resolve_firm as gnma_resolve_firm,
    SF_SPONSORS, MF_SPONSORS, REVERSE_SPONSORS, CO_SPONSORS,
    TRUSTEES, TRUST_COUNSEL, DIRECTORY as GNMA_DIRECTORY,
)

# ─── FICC Trading Floor Desks ────────────────────────────────────────────
from .treasuries_env import TreasuriesEnv, make_treasuries_env
from .distressed_credit_env import DistressedCreditEnv, make_distressed_credit_env
from .rates_env import RatesEnv, make_rates_env
from .ig_credit_env import IGCreditEnv, make_ig_credit_env
from .munis_env import MunisEnv, make_munis_env
from .repo_env import RepoEnv, make_repo_env
from .fx_env import FXEnv, make_fx_env
from .commodities_env import CommoditiesEnv, make_commodities_env
from .ficc_floor_env import FICCFloorEnv, make_ficc_floor_env

# ─── Equities Desks ─────────────────────────────────────────────────────
from .equities_env import EquitiesEnv, make_equities_env
from .eq_derivatives_env import EqDerivativesEnv, make_eq_derivatives_env
from .prime_brokerage_env import PrimeBrokerageEnv, make_prime_brokerage_env

# ─── Investment Banking Desks ──────────────────────────────────────────
from .ecm_env import ECMEnv, make_ecm_env
from .dcm_env import DCMEnv, make_dcm_env
from .ma_advisory_env import MAAdvisoryEnv, make_ma_advisory_env

# ─── Unified Investment Bank ──────────────────────────────────────────
from .investment_bank_env import InvestmentBankEnv, make_investment_bank_env

# ─── Multi-Desk Training Pipeline ────────────────────────────────────
from .train_bank import (
    DeskTrainer, DeskConfig, DESK_REGISTRY,
    benchmark_desk, benchmark_all, smoke_test_all,
    train_curriculum,
    FICC_DESKS, EQUITY_DESKS, IBD_DESKS, ALL_DESKS, ALL_ENVS,
)
