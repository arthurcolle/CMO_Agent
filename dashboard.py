"""
Streamlit Dashboard for CMO Agent — Interactive market data, deal structuring,
model management, compliance checking, and agent chat interface.

Launch:
    streamlit run cmo_agent/dashboard.py
"""
import os
import sys
import json
import traceback
from datetime import date, datetime
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CMO Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Conditional imports — graceful degradation when deps are missing
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Add project root to path so relative imports work when running standalone
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from cmo_agent.yield_curve import build_us_treasury_curve, YieldCurve
    HAS_YIELD_CURVE = True
except Exception:
    HAS_YIELD_CURVE = False

try:
    from cmo_agent.data_sources import (
        build_live_treasury_curve,
        fetch_current_mortgage_rates,
        get_full_market_snapshot,
    )
    HAS_DATA_SOURCES = True
except Exception:
    HAS_DATA_SOURCES = False

try:
    from cmo_agent.real_market_data import RealMarketDataProvider
    HAS_REAL_MARKET = True
except Exception:
    HAS_REAL_MARKET = False

try:
    from cmo_agent.spec_pool import SpecPool, AgencyType, CollateralType, project_pool_cashflows
    HAS_SPEC_POOL = True
except Exception:
    HAS_SPEC_POOL = False

try:
    from cmo_agent.cmo_structure import (
        create_sequential_cmo,
        create_pac_support_cmo,
        create_kitchen_sink_structure,
        CMOCashFlows,
    )
    HAS_CMO_STRUCTURE = True
except Exception:
    HAS_CMO_STRUCTURE = False

try:
    from cmo_agent.pricing import price_deal, PricingResult
    HAS_PRICING = True
except Exception:
    HAS_PRICING = False

try:
    from cmo_agent.model_registry import list_models, get_model, scan_unregistered
    HAS_MODEL_REGISTRY = True
except Exception:
    HAS_MODEL_REGISTRY = False

try:
    from cmo_agent.agent import CMOAgent, TOOLS
    HAS_AGENT = True
except Exception:
    HAS_AGENT = False

try:
    from cmo_agent.compliance import quick_check, get_all_rules, CGCSubdivision, SUBDIVISION_LABELS
    HAS_COMPLIANCE = True
except Exception:
    HAS_COMPLIANCE = False

try:
    from cmo_agent.backtest import backtest_agent_vs_experts
    HAS_BACKTEST = True
except Exception:
    HAS_BACKTEST = False

try:
    from cmo_agent.yield_book_env import make_yield_book_env
    HAS_ENV = True
except Exception:
    HAS_ENV = False

# Optional API mode
CMO_API_URL = os.environ.get("CMO_API_URL", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — Page navigation
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = [
    "Market Overview",
    "Deal Structuring",
    "Training Dashboard",
    "Model Zoo",
    "Agent Chat",
    "Compliance",
]

st.sidebar.title("CMO Agent")
st.sidebar.caption("Autonomous CMO Pricing & Structuring")
page = st.sidebar.radio("Navigate", PAGES, index=0)
st.sidebar.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_bps(v: float) -> str:
    return f"{v:.1f} bps"


def _fmt_pct(v: float) -> str:
    return f"{v:.3f}%"


def _fmt_usd(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


@st.cache_data(ttl=300)
def _fetch_yield_curve():
    """Fetch the best available yield curve. Returns (curve_obj, points_list, source_label)."""
    # Try live data sources first
    if HAS_DATA_SOURCES:
        try:
            curve_data = build_live_treasury_curve()
            if curve_data and "yields" in curve_data:
                yields = curve_data["yields"]
                points = []
                tenor_map = {
                    "1M": 1/12, "2M": 2/12, "3M": 0.25, "4M": 4/12, "6M": 0.5,
                    "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7,
                    "10Y": 10, "20Y": 20, "30Y": 30,
                }
                for label, mat in sorted(tenor_map.items(), key=lambda x: x[1]):
                    if label in yields and yields[label] is not None:
                        points.append({"maturity": mat, "label": label, "yield": yields[label]})
                if points:
                    curve_obj = None
                    if HAS_YIELD_CURVE:
                        try:
                            curve_obj = build_us_treasury_curve(live=True)
                        except Exception:
                            pass
                    return curve_obj, points, "Live (FRED/Treasury.gov)"
        except Exception:
            pass

    # Fallback to built-in curve
    if HAS_YIELD_CURVE:
        try:
            curve = build_us_treasury_curve(live=False)
            points = [
                {"maturity": p.maturity_years, "label": f"{p.maturity_years}Y", "yield": p.yield_pct}
                for p in curve.points
            ]
            return curve, points, "Built-in (static)"
        except Exception:
            pass

    return None, [], "Unavailable"


@st.cache_data(ttl=300)
def _fetch_mortgage_rates():
    if HAS_DATA_SOURCES:
        try:
            return fetch_current_mortgage_rates()
        except Exception:
            pass
    return {"30yr_fixed": 6.65, "15yr_fixed": 5.89, "5yr_arm": 6.10, "_source": "fallback"}


@st.cache_data(ttl=300)
def _fetch_market_snapshot():
    if HAS_DATA_SOURCES:
        try:
            return get_full_market_snapshot()
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 — Market Overview
# ═══════════════════════════════════════════════════════════════════════════════

def page_market_overview():
    st.header("Market Overview")
    st.caption(f"As of {datetime.now().strftime('%Y-%m-%d %H:%M')} ET")

    curve_obj, points, source = _fetch_yield_curve()
    mortgage_rates = _fetch_mortgage_rates()

    # ---------- Yield Curve Chart ----------
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("US Treasury Yield Curve")
        if points and HAS_PLOTLY:
            maturities = [p["maturity"] for p in points]
            yields = [p["yield"] for p in points]
            labels = [p["label"] for p in points]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=maturities, y=yields,
                mode="lines+markers",
                name="Treasury Curve",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
                text=labels,
                hovertemplate="<b>%{text}</b><br>Maturity: %{x:.2f}Y<br>Yield: %{y:.3f}%<extra></extra>",
            ))

            # Add Nelson-Siegel fit if we have a curve object with enough points
            if curve_obj is not None and hasattr(curve_obj, "_ns_params") and curve_obj._ns_params is not None:
                ns_x = [i * 0.1 for i in range(1, 301)]
                ns_y = [curve_obj.get_yield(m) for m in ns_x]
                fig.add_trace(go.Scatter(
                    x=ns_x, y=ns_y,
                    mode="lines",
                    name="Nelson-Siegel Fit",
                    line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                    hovertemplate="Maturity: %{x:.1f}Y<br>NS Yield: %{y:.3f}%<extra></extra>",
                ))

            fig.update_layout(
                xaxis_title="Maturity (Years)",
                yaxis_title="Yield (%)",
                height=450,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif points:
            # Fallback table when plotly not available
            st.table([{"Tenor": p["label"], "Yield (%)": f"{p['yield']:.3f}"} for p in points])
        else:
            st.warning("Yield curve data unavailable.")

        st.caption(f"Source: {source}")

    with col2:
        st.subheader("Mortgage Rates")
        rate_labels = {
            "30yr_fixed": "30Y Fixed",
            "15yr_fixed": "15Y Fixed",
            "5yr_arm": "5/1 ARM",
        }
        for key, label in rate_labels.items():
            val = mortgage_rates.get(key)
            if val is not None and isinstance(val, (int, float)):
                st.metric(label, f"{val:.2f}%")

        src = mortgage_rates.get("_source", "FRED")
        st.caption(f"Source: {src}")

    # ---------- Curve Data Table ----------
    if points:
        st.subheader("Curve Data")
        if HAS_PANDAS:
            df = pd.DataFrame(points)
            df.columns = ["Maturity (Y)", "Tenor", "Yield (%)"]
            df["Yield (%)"] = df["Yield (%)"].round(3)

            # Add forward rates if curve object available
            if curve_obj is not None:
                fwd = []
                for i in range(len(points)):
                    t = points[i]["maturity"]
                    try:
                        f = curve_obj.forward_rate(max(t - 0.5, 0.01), t + 0.5)
                        fwd.append(round(f, 3))
                    except Exception:
                        fwd.append(None)
                df["Fwd Rate (%)"] = fwd
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.json(points)

    # ---------- Market Snapshot ----------
    snapshot = _fetch_market_snapshot()
    if snapshot:
        st.subheader("Market Snapshot")
        cols = st.columns(4)
        fed_rates = snapshot.get("fed_rates", {})
        if fed_rates:
            with cols[0]:
                st.metric("SOFR", _fmt_pct(fed_rates.get("sofr", 0)) if fed_rates.get("sofr") else "N/A")
            with cols[1]:
                st.metric("EFFR", _fmt_pct(fed_rates.get("effr", 0)) if fed_rates.get("effr") else "N/A")
        soma = snapshot.get("soma_holdings", {})
        if soma:
            with cols[2]:
                mbs_par = soma.get("mbs_par", soma.get("total", 0))
                if mbs_par:
                    st.metric("SOMA MBS", _fmt_usd(mbs_par))
            with cols[3]:
                tsy_par = soma.get("treasury_par", 0)
                if tsy_par:
                    st.metric("SOMA TSY", _fmt_usd(tsy_par))


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 — Deal Structuring
# ═══════════════════════════════════════════════════════════════════════════════

def page_deal_structuring():
    st.header("Deal Structuring")

    if not HAS_SPEC_POOL or not HAS_CMO_STRUCTURE:
        st.error("Deal structuring modules not available. Check cmo_agent installation.")
        return

    # ---------- Sidebar inputs ----------
    st.sidebar.markdown("### Collateral Parameters")
    balance = st.sidebar.slider(
        "Collateral Balance ($M)", 10, 500, 100, step=10,
        help="Original face amount of the collateral pool",
    )
    balance_dollars = balance * 1_000_000

    coupon = st.sidebar.slider("Pass-Through Coupon (%)", 4.0, 8.0, 5.5, step=0.25)
    wac = st.sidebar.slider("WAC (%)", 4.5, 9.0, coupon + 0.5, step=0.25)
    wam = st.sidebar.slider("WAM (months)", 60, 360, 340, step=10)
    psa_speed = st.sidebar.slider("PSA Speed", 50, 500, 150, step=25)

    agency_choice = st.sidebar.selectbox("Agency", ["FNMA", "FHLMC", "GNMA"])
    agency_map = {"FNMA": AgencyType.FNMA, "FHLMC": AgencyType.FHLMC, "GNMA": AgencyType.GNMA}
    agency = agency_map[agency_choice]

    collateral_map = {"FNMA": CollateralType.FN, "FHLMC": CollateralType.FH, "GNMA": CollateralType.G2}
    collateral_type = collateral_map[agency_choice]

    structure_type = st.sidebar.selectbox(
        "Structure Type",
        ["Sequential", "PAC / Support", "Kitchen Sink"],
    )

    # ---------- Structure the deal ----------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Collateral Summary")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Balance | {_fmt_usd(balance_dollars)} |
| Agency | {agency_choice} |
| Coupon | {coupon:.2f}% |
| WAC | {wac:.2f}% |
| WAM | {wam} months |
| PSA | {psa_speed} |
| Structure | {structure_type} |
""")

    do_structure = st.button("Structure Deal", type="primary", use_container_width=True)

    if do_structure:
        with st.spinner("Structuring deal..."):
            try:
                pool = SpecPool(
                    pool_id="DASH-001",
                    agency=agency,
                    collateral_type=collateral_type,
                    coupon=coupon,
                    wac=wac,
                    wam=wam,
                    wala=0,
                    original_balance=balance_dollars,
                    current_balance=balance_dollars,
                    original_term=360 if wam > 180 else 180,
                )

                pool_cf = project_pool_cashflows(pool, psa_speed=psa_speed)

                if structure_type == "Sequential":
                    n_tranches = 4
                    sizes = [balance_dollars * w for w in [0.35, 0.30, 0.20, 0.15]]
                    coupons = [coupon - 0.5, coupon - 0.25, coupon, coupon + 0.25]
                    cmo = create_sequential_cmo(
                        deal_id="DASH-SEQ-001",
                        collateral_flows=pool_cf,
                        tranche_sizes=sizes,
                        tranche_coupons=coupons,
                        tranche_names=["A1", "A2", "A3", "A4"],
                        collateral_coupon=coupon,
                    )
                elif structure_type == "PAC / Support":
                    pac_bal = balance_dollars * 0.65
                    sup_bal = balance_dollars * 0.35
                    cmo = create_pac_support_cmo(
                        deal_id="DASH-PAC-001",
                        collateral_flows=pool_cf,
                        pac_balance=pac_bal,
                        pac_coupon=coupon - 0.25,
                        support_balance=sup_bal,
                        support_coupon=coupon + 0.50,
                        pac_lower=100.0,
                        pac_upper=300.0,
                        collateral_coupon=coupon,
                    )
                else:  # Kitchen Sink
                    cmo = create_kitchen_sink_structure(
                        deal_id="DASH-KS-001",
                        collateral_flows=pool_cf,
                        collateral_coupon=coupon,
                        total_balance=balance_dollars,
                    )

                st.session_state["last_cmo"] = cmo
                st.session_state["last_pool_cf"] = pool_cf
                st.success("Deal structured successfully.")

            except Exception as e:
                st.error(f"Structuring failed: {e}")
                st.code(traceback.format_exc())

    # ---------- Display results ----------
    cmo = st.session_state.get("last_cmo")
    if cmo is not None:
        st.subheader("Tranche Waterfall")
        summary = cmo.summary()
        tranche_data = summary.get("tranches", {})

        if tranche_data and HAS_PLOTLY:
            names = list(tranche_data.keys())
            principals = [tranche_data[n].get("total_principal", 0) for n in names]
            wals = [tranche_data[n].get("wal", 0) for n in names]
            windows = [tranche_data[n].get("window", "") for n in names]

            # Bar chart of tranche sizes
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=names, y=[p / 1e6 for p in principals],
                marker_color=px.colors.qualitative.Set2[:len(names)],
                text=[f"${p/1e6:.1f}M" for p in principals],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Principal: $%{y:.1f}M<extra></extra>",
            ))
            fig_bar.update_layout(
                title="Tranche Sizes ($M)",
                yaxis_title="Principal ($M)",
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # WAL comparison
            fig_wal = go.Figure()
            fig_wal.add_trace(go.Bar(
                x=names, y=wals,
                marker_color=px.colors.qualitative.Pastel[:len(names)],
                text=[f"{w:.1f}Y" for w in wals],
                textposition="auto",
            ))
            fig_wal.update_layout(
                title="Weighted Average Life (Years)",
                yaxis_title="WAL (Years)",
                height=350,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_wal, use_container_width=True)

        # Tranche detail table
        st.subheader("Tranche Details")
        if tranche_data and HAS_PANDAS:
            rows = []
            for name, info in tranche_data.items():
                rows.append({
                    "Tranche": name,
                    "Principal ($M)": round(info.get("total_principal", 0) / 1e6, 2),
                    "Interest ($M)": round(info.get("total_interest", 0) / 1e6, 2),
                    "WAL (Y)": info.get("wal", 0),
                    "Window": info.get("window", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif tranche_data:
            st.json(tranche_data)

        residual = summary.get("residual", 0)
        st.metric("Residual Cash Flow (should be ~$0)", f"${residual:,.2f}")

        # ---------- Pricing ----------
        st.subheader("Tranche Pricing")
        if HAS_PRICING and HAS_YIELD_CURVE:
            try:
                curve_obj, _, _ = _fetch_yield_curve()
                if curve_obj is None:
                    curve_obj = build_us_treasury_curve(live=False)
                pricing_results = price_deal(cmo, curve_obj)
                if pricing_results and HAS_PANDAS:
                    price_rows = []
                    for name, pr in pricing_results.items():
                        d = pr.to_dict() if hasattr(pr, "to_dict") else pr.__dict__
                        price_rows.append({
                            "Tranche": d.get("name", name),
                            "Price": d.get("price", 0),
                            "Yield (%)": d.get("yield", 0),
                            "WAL": d.get("wal", 0),
                            "Mod Dur": d.get("mod_duration", 0),
                            "Eff Dur": d.get("eff_duration", 0),
                            "OAS (bps)": d.get("oas_bps", 0),
                            "Z-Sprd (bps)": d.get("z_spread_bps", 0),
                            "DV01": d.get("dv01", 0),
                        })
                    df_price = pd.DataFrame(price_rows)
                    st.dataframe(df_price, use_container_width=True, hide_index=True)
                elif pricing_results:
                    for name, pr in pricing_results.items():
                        st.write(f"**{name}**: {pr}")
            except Exception as e:
                st.warning(f"Pricing unavailable: {e}")
        else:
            st.info("Pricing module or yield curve not available.")

        # ---------- RL Suggest ----------
        if HAS_ENV and HAS_MODEL_REGISTRY:
            st.subheader("RL-Suggested Structure")
            models = list_models()
            if models:
                model_names = [m.get("model_id", "unknown") for m in models]
                selected_model = st.selectbox("Select RL Model", model_names, key="rl_model_deal")
                if st.button("RL Suggest"):
                    st.info(
                        f"RL suggestion would load model '{selected_model}' and run inference "
                        f"on the current market/collateral state. This requires a GPU-enabled "
                        f"environment with PyTorch installed."
                    )
            else:
                st.info("No trained models found. Train a model first or scan for unregistered models.")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 — Training Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def page_training_dashboard():
    st.header("Training Dashboard")

    if not HAS_MODEL_REGISTRY:
        st.error("Model registry module not available.")
        return

    # ---------- Scan for models ----------
    col_scan, col_info = st.columns([1, 3])
    with col_scan:
        if st.button("Scan Models"):
            with st.spinner("Scanning for unregistered models..."):
                try:
                    new_models = scan_unregistered()
                    if new_models:
                        st.success(f"Discovered {len(new_models)} new model(s).")
                        for m in new_models:
                            st.write(f"  - {m.model_id}")
                    else:
                        st.info("No new models found.")
                except Exception as e:
                    st.warning(f"Scan error: {e}")

    models = list_models()
    if not models:
        st.info("No registered models. Place .pt files in the models/ directory and click 'Scan Models'.")
        return

    # ---------- Model comparison table ----------
    st.subheader("Model Comparison")
    if HAS_PANDAS:
        rows = []
        for m in models:
            rows.append({
                "Model ID": m.get("model_id", ""),
                "Env": m.get("env_name", ""),
                "Obs Dim": m.get("obs_dim", 0),
                "Action Dims": str(m.get("action_dims", [])),
                "Best Reward": round(m.get("best_eval_reward", 0), 2),
                "Timesteps": f"{m.get('timesteps', 0):,}",
                "Description": m.get("description", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        for m in models:
            st.write(f"**{m.get('model_id')}** — reward: {m.get('best_eval_reward', 0):.2f}")

    # ---------- Reward curves ----------
    st.subheader("Training Reward Curves")

    # Check for checkpoint data with reward logs
    models_dir = os.path.join(_project_root, "models")
    reward_files = []
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith("_rewards.json") or fname.endswith("_log.json"):
                reward_files.append(os.path.join(models_dir, fname))

    if reward_files and HAS_PLOTLY:
        fig = go.Figure()
        for rpath in reward_files:
            try:
                with open(rpath) as f:
                    data = json.load(f)
                rewards = data if isinstance(data, list) else data.get("rewards", data.get("eval_rewards", []))
                if rewards:
                    label = os.path.basename(rpath).replace("_rewards.json", "").replace("_log.json", "")
                    fig.add_trace(go.Scatter(
                        y=rewards,
                        mode="lines",
                        name=label,
                    ))
            except Exception:
                continue

        if fig.data:
            fig.update_layout(
                title="Training Reward Over Time",
                xaxis_title="Evaluation Episode",
                yaxis_title="Reward",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Reward log files found but no plottable data.")
    elif not reward_files:
        st.info("No reward log files (*_rewards.json or *_log.json) found in models/ directory.")

    # ---------- Training config for each model ----------
    st.subheader("Training Configurations")
    for m in models:
        config = m.get("training_config", {})
        if config:
            with st.expander(f"{m.get('model_id', 'unknown')} config"):
                st.json(config)

    # ---------- Backtest section ----------
    st.subheader("Backtesting")
    if HAS_BACKTEST:
        model_options = [m.get("model_id", "") for m in models]
        bt_model = st.selectbox("Model to Backtest", model_options, key="bt_model")
        bt_n = st.number_input("Number of demos (0=all)", 0, 742, 20, key="bt_n")
        if st.button("Run Backtest"):
            model_meta = get_model(bt_model)
            if model_meta:
                model_path = model_meta.get("file_path", "")
                if not os.path.isabs(model_path):
                    model_path = os.path.join(models_dir, model_path)
                with st.spinner(f"Backtesting {bt_model} against expert demos..."):
                    try:
                        result = backtest_agent_vs_experts(model_path, n_demos=bt_n)
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Agent Mean Reward", f"{result.agent_mean_reward:.2f}")
                        col_b.metric("Expert Mean Reward", f"{result.expert_mean_reward:.2f}")
                        col_c.metric("Win Rate", f"{result.win_rate:.1%}")
                        st.metric("Mean Alpha", f"{result.mean_alpha:.2f}")
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
            else:
                st.error(f"Model '{bt_model}' not found in registry.")
    else:
        st.info("Backtest module not available.")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 — Model Zoo
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_zoo():
    st.header("Model Zoo")

    if not HAS_MODEL_REGISTRY:
        st.error("Model registry module not available.")
        return

    models = list_models()
    if not models:
        st.info("No registered models. Use the Training Dashboard page to scan for models.")
        return

    st.subheader(f"Registered Models ({len(models)})")

    if HAS_PANDAS:
        rows = []
        for m in models:
            rows.append({
                "Model ID": m.get("model_id", ""),
                "Environment": m.get("env_name", ""),
                "Obs Dim": m.get("obs_dim", 0),
                "Action Dims": str(m.get("action_dims", [])),
                "Hidden Dim": m.get("hidden_dim", 256),
                "Best Eval Reward": round(m.get("best_eval_reward", 0), 2),
                "Timesteps": f"{m.get('timesteps', 0):,}",
                "Compatible Envs": ", ".join(m.get("compatible_envs", [])),
                "Parent": m.get("parent_model", ""),
                "File": os.path.basename(m.get("file_path", "")),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- Detailed model view ----------
    st.subheader("Model Details")
    model_ids = [m.get("model_id", "") for m in models]
    selected = st.selectbox("Select Model", model_ids, key="zoo_model_select")
    meta = get_model(selected)

    if meta:
        col1, col2, col3 = st.columns(3)
        col1.metric("Obs Dim", meta.get("obs_dim", 0))
        col2.metric("Best Reward", f"{meta.get('best_eval_reward', 0):.2f}")
        col3.metric("Timesteps", f"{meta.get('timesteps', 0):,}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Hidden Dim", meta.get("hidden_dim", 256))
        col5.metric("Action Dims", str(meta.get("action_dims", [])))
        col6.metric("Environment", meta.get("env_name", ""))

        description = meta.get("description", "")
        if description:
            st.markdown(f"**Description:** {description}")

        parent = meta.get("parent_model", "")
        if parent:
            st.markdown(f"**Parent Model:** {parent}")

        created = meta.get("created_at", 0)
        if created:
            try:
                ts = datetime.fromtimestamp(created)
                st.caption(f"Created: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                pass

        file_path = meta.get("file_path", "")
        if file_path:
            full_path = file_path if os.path.isabs(file_path) else os.path.join(_project_root, "models", file_path)
            exists = os.path.exists(full_path)
            st.markdown(f"**File:** `{file_path}` {'(exists)' if exists else '(MISSING)'}")
            if exists:
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                st.caption(f"File size: {size_mb:.1f} MB")

        config = meta.get("training_config", {})
        if config:
            with st.expander("Full Training Configuration"):
                st.json(config)

        with st.expander("Raw Metadata"):
            st.json(meta)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5 — Agent Chat
# ═══════════════════════════════════════════════════════════════════════════════

def page_agent_chat():
    st.header("CMO Agent Chat")

    if not HAS_AGENT:
        st.error(
            "Agent module not available. Ensure ANTHROPIC_API_KEY is set and "
            "the anthropic package is installed."
        )
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "ANTHROPIC_API_KEY not set. The agent requires an Anthropic API key. "
            "Set it via environment variable or enter below."
        )
        api_key = st.text_input("Anthropic API Key", type="password", key="api_key_input")
        if not api_key:
            return

    # Initialize agent in session state
    if "agent" not in st.session_state:
        try:
            st.session_state["agent"] = CMOAgent(api_key=api_key)
            st.session_state["chat_history"] = []
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return

    agent = st.session_state["agent"]
    chat_history = st.session_state.get("chat_history", [])

    # Available tools sidebar
    st.sidebar.markdown("### Agent Tools")
    st.sidebar.caption(f"{len(TOOLS)} tools available")
    with st.sidebar.expander("View All Tools"):
        for tool in TOOLS:
            st.sidebar.markdown(f"- **{tool['name']}**")

    # Display chat history
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)
        # Show tool calls if any
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            with st.expander(f"Tool calls ({len(tool_calls)})"):
                for tc in tool_calls:
                    st.markdown(f"**{tc.get('name', 'unknown')}**")
                    if tc.get("input"):
                        st.json(tc["input"])
                    if tc.get("output"):
                        st.code(tc["output"][:2000])

    # Chat input
    user_input = st.chat_input("Ask the CMO Agent anything...")
    if user_input:
        # Add user message
        chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                try:
                    response = agent.chat(user_input)

                    # Extract tool calls from conversation history
                    tool_calls = []
                    for entry in agent.conversation_history:
                        if entry.get("role") == "assistant" and isinstance(entry.get("content"), list):
                            for block in entry["content"]:
                                if hasattr(block, "type") and block.type == "tool_use":
                                    tool_calls.append({
                                        "name": block.name,
                                        "input": block.input if hasattr(block, "input") else {},
                                    })

                    st.markdown(response)

                    # Show tool calls in expandable section
                    if tool_calls:
                        recent_tools = tool_calls[-10:]  # Last 10 tool calls
                        with st.expander(f"Tool calls used ({len(recent_tools)})"):
                            for tc in recent_tools:
                                st.markdown(f"**{tc['name']}**")
                                st.json(tc["input"])

                    chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "tool_calls": tool_calls[-10:] if tool_calls else [],
                    })

                except Exception as e:
                    error_msg = f"Agent error: {e}"
                    st.error(error_msg)
                    chat_history.append({"role": "assistant", "content": error_msg})

        st.session_state["chat_history"] = chat_history

    # Quick-action buttons
    st.markdown("---")
    st.caption("Quick Actions")
    quick_cols = st.columns(4)
    prompts = [
        ("Yield Curve", "Show me the current yield curve and key rates."),
        ("Structure Deal", "Structure a $100M FNMA 5.5% sequential CMO with 4 tranches."),
        ("Risk Report", "Run a full risk analysis on a PAC/Support structure at 150 PSA."),
        ("Market View", "Give me today's market snapshot including SOFR, mortgage rates, and MBS technicals."),
    ]
    for i, (label, prompt) in enumerate(prompts):
        with quick_cols[i]:
            if st.button(label, key=f"quick_{i}"):
                st.session_state["_pending_prompt"] = prompt
                st.rerun()

    # Handle pending prompt from quick-action buttons
    pending = st.session_state.pop("_pending_prompt", None)
    if pending:
        chat_history.append({"role": "user", "content": pending})
        try:
            response = agent.chat(pending)
            chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            chat_history.append({"role": "assistant", "content": f"Error: {e}"})
        st.session_state["chat_history"] = chat_history
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 6 — Compliance
# ═══════════════════════════════════════════════════════════════════════════════

def page_compliance():
    st.header("CGC 53601 Compliance Checker")

    if not HAS_COMPLIANCE:
        st.error("Compliance module not available. Check cmo_agent installation.")
        return

    tab_check, tab_rules = st.tabs(["Security Check", "Rules Reference"])

    # ---------- Security Check ----------
    with tab_check:
        st.subheader("Check Security Compliance")

        col1, col2 = st.columns(2)

        with col1:
            # Build subdivision options
            sub_options = {}
            for sub in CGCSubdivision:
                label = SUBDIVISION_LABELS.get(sub, sub.value)
                sub_options[label] = sub.value

            selected_label = st.selectbox("Investment Type (Subdivision)", list(sub_options.keys()))
            subdivision = sub_options[selected_label]

            par_value = st.number_input(
                "Par Value ($)", min_value=0, max_value=500_000_000,
                value=1_000_000, step=100_000,
            )
            credit_rating = st.selectbox(
                "Credit Rating",
                ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                 "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
                 "A-1+", "A-1", "A-2", "P-1", "P-2", "NR"],
                index=0,
            )

        with col2:
            maturity_years = st.number_input("Maturity (years)", 0.0, 50.0, 5.0, step=0.5)
            maturity_days = int(maturity_years * 365.25)

            entity_assets = st.number_input(
                "Entity Assets ($M, for CP limit)", 0, 10000, 500, step=50,
            )
            entity_assets_dollars = entity_assets * 1_000_000

            trade_date = st.date_input("Trade Date", value=date.today())
            settlement_offset = st.number_input("Settlement (T+N days)", 0, 90, 2)
            settlement_date = date.fromordinal(trade_date.toordinal() + settlement_offset)
            st.caption(f"Settlement: {settlement_date}")

        if st.button("Run Compliance Check", type="primary"):
            with st.spinner("Checking compliance..."):
                try:
                    result = quick_check(
                        subdivision=subdivision,
                        par_value=par_value,
                        credit_rating=credit_rating,
                        maturity_days=maturity_days,
                        settlement_date=str(settlement_date),
                        trade_date=str(trade_date),
                        entity_assets=entity_assets_dollars,
                    )

                    # Display result
                    passed = result.get("passed", result.get("compliant", False))
                    if passed:
                        st.success("COMPLIANT - Security meets all CGC 53601 requirements.")
                    else:
                        st.error("NON-COMPLIANT - One or more violations detected.")

                    violations = result.get("violations", [])
                    if violations:
                        st.subheader("Violations")
                        for v in violations:
                            st.warning(v)

                    # Show rule details
                    details = {k: v for k, v in result.items() if k not in ("violations", "passed", "compliant")}
                    if details:
                        with st.expander("Rule Details"):
                            st.json(details)

                except Exception as e:
                    st.error(f"Compliance check failed: {e}")
                    st.code(traceback.format_exc())

    # ---------- Rules Reference ----------
    with tab_rules:
        st.subheader("CGC 53601 Rules Reference")
        try:
            all_rules = get_all_rules()

            prohibited = all_rules.pop("_prohibited_instruments", [])

            if HAS_PANDAS:
                rows = []
                for label, rule in all_rules.items():
                    rows.append({
                        "Investment Type": label,
                        "Subdivision": rule.get("subdivision", ""),
                        "Max Maturity": f"{rule.get('max_maturity_years', 'N/A')}Y" if rule.get("max_maturity_years") else "None",
                        "Max Portfolio %": rule.get("max_portfolio_pct", "unlimited"),
                        "Min Rating": rule.get("min_rating", "none"),
                        "Issuer Limit": rule.get("single_issuer_pct", "none"),
                        "Notes": (rule.get("notes", "") or "")[:80],
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.json(all_rules)

            if prohibited:
                st.subheader("Prohibited Instruments")
                for p in prohibited:
                    st.markdown(f"- {p}")

        except Exception as e:
            st.error(f"Failed to load rules: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE_MAP = {
    "Market Overview": page_market_overview,
    "Deal Structuring": page_deal_structuring,
    "Training Dashboard": page_training_dashboard,
    "Model Zoo": page_model_zoo,
    "Agent Chat": page_agent_chat,
    "Compliance": page_compliance,
}

# Status indicators in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
status_items = [
    ("Yield Curve", HAS_YIELD_CURVE),
    ("Data Sources", HAS_DATA_SOURCES),
    ("Real Market Data", HAS_REAL_MARKET),
    ("Spec Pool", HAS_SPEC_POOL),
    ("CMO Structure", HAS_CMO_STRUCTURE),
    ("Pricing", HAS_PRICING),
    ("Model Registry", HAS_MODEL_REGISTRY),
    ("Agent (Claude)", HAS_AGENT),
    ("Compliance", HAS_COMPLIANCE),
    ("Backtest", HAS_BACKTEST),
    ("RL Environment", HAS_ENV),
    ("Plotly", HAS_PLOTLY),
    ("Pandas", HAS_PANDAS),
]
for name, available in status_items:
    indicator = "[OK]" if available else "[--]"
    st.sidebar.text(f"  {indicator} {name}")

if CMO_API_URL:
    st.sidebar.markdown(f"**API:** `{CMO_API_URL}`")

# Render selected page
_PAGE_MAP[page]()
