"""
app.py
------
AI-Enabled Micro-Channel Battery Thermal Management for EVs (KJS-CES-02)
Streamlit Application for Maximum Battery Surface Temperature (T_max) Prediction.

Uses a trained Random Forest Regressor (200 trees) trained on MHD nanofluid thermal dataset.
Features:
    Ha   - Hartmann Number
    phi  - Nanoparticle Volume Fraction
    u_in - Inlet Flow Velocity (m/s)
Target:
    T_max - Maximum Battery Surface Temperature (°C)
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Battery T_max Predictor | KJS-CES-02",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("battery_temperature_model.pkl")
RESULTS_DIR = Path("results")
DATA_PATH = Path("training_data.csv")

# Exact features and order from train.py
FEATURES = ["Ha", "phi", "u_in"]
TARGET = "T_max"

# Exact risk thresholds and classification logic from predict.py
THRESHOLDS = [
    (35.0, "Safe"),
    (45.0, "Moderate"),
    (55.0, "High"),
]
CRITICAL_LABEL = "Critical"

RISK_METADATA = {
    "Safe": {
        "color": "#10b981",
        "bg": "rgba(16, 185, 129, 0.12)",
        "border": "#10b981",
        "icon": "🟢",
        "desc": "Optimal thermal conditions. Cell degradation is minimized, and cooling capacity is sufficient.",
    },
    "Moderate": {
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.12)",
        "border": "#f59e0b",
        "icon": "🟡",
        "desc": "Elevated thermal load within operational limits. Proactive cooling increase recommended.",
    },
    "High": {
        "color": "#f97316",
        "bg": "rgba(249, 115, 22, 0.12)",
        "border": "#f97316",
        "icon": "🟠",
        "desc": "Thermal throttling recommended. Sustained operation here accelerates battery capacity loss.",
    },
    "Critical": {
        "color": "#ef4444",
        "bg": "rgba(239, 68, 68, 0.15)",
        "border": "#ef4444",
        "icon": "🔴",
        "desc": "Severe thermal runaway hazard! Immediate peak micro-channel pump activation or load cutoff required.",
    },
}

# Real dataset statistics from training_data.csv (5,000 samples)
# Ha:   min=0.009719, max=59.996717, mean=30.000013, median=30.003007
# phi:  min=0.010002, max=0.050000, mean=0.030000, median=0.029998
# u_in: min=0.050004, max=0.299996, mean=0.175000, median=0.175001
HA_MIN, HA_MAX, HA_MEAN = 0.0, 60.0, 30.0
PHI_MIN, PHI_MAX, PHI_MEAN = 0.01, 0.05, 0.03
UIN_MIN, UIN_MAX, UIN_MEAN = 0.05, 0.30, 0.175

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font and subtle styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        padding: 1.5rem 1.8rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    .metric-card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        animation: fadeIn 0.4s ease-in-out;
        margin-bottom: 1.5rem;
    }
    
    .metric-val {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.6rem;
    }
    
    .stat-box {
        padding: 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        text-align: center;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.2rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model & Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load the Random Forest model bundle saved by train.py."""
    if not MODEL_PATH.exists():
        st.error(f"Model file `{MODEL_PATH}` not found! Run `python train.py` first.")
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def classify_risk(t_max: float) -> str:
    """Exact thermal condition banding from predict.py."""
    for limit, label in THRESHOLDS:
        if t_max < limit:
            return label
    return CRITICAL_LABEL


def run_prediction(model, features, ha_val: float, phi_val: float, uin_val: float):
    """Run model prediction on exact feature DataFrame."""
    X = pd.DataFrame([[ha_val, phi_val, uin_val]], columns=features)
    t_pred = float(model.predict(X)[0])
    condition = classify_risk(t_pred)
    return t_pred, condition


# -----------------------------------------------------------------------------
# Application Setup & Header
# -----------------------------------------------------------------------------
model, features = load_trained_model()

st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">⚡ AI-Enabled Micro-Channel Battery Thermal Management</h1>
        <p class="hero-subtitle">
            Predict maximum battery surface temperature (<strong>T_max</strong>) to enable proactive cooling 
            in electric vehicles (Project: <strong>KJS-CES-02</strong>).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Control Panel & Physical Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Thermal Input Parameters")
    st.caption("Adjust operating conditions using the verified simulation bounds from `training_data.csv`.")

    # Preset quick-selectors
    st.markdown("#### ⚡ Quick Presets")
    preset_cols = st.columns(3)
    if preset_cols[0].button("Baseline", use_container_width=True):
        st.session_state["ha_input"] = HA_MEAN
        st.session_state["phi_input"] = PHI_MEAN
        st.session_state["uin_input"] = UIN_MEAN
    if preset_cols[1].button("High Flow", use_container_width=True):
        st.session_state["ha_input"] = 10.0
        st.session_state["phi_input"] = 0.05
        st.session_state["uin_input"] = 0.28
    if preset_cols[2].button("Low Flow", use_container_width=True):
        st.session_state["ha_input"] = 55.0
        st.session_state["phi_input"] = 0.01
        st.session_state["uin_input"] = 0.08

    st.divider()

    # Input 1: Hartmann Number (Ha)
    st.markdown("**1. Hartmann Number ($Ha$)**")
    default_ha = st.session_state.get("ha_input", HA_MEAN)
    ha_slider = st.slider(
        "Ha slider",
        min_value=float(HA_MIN),
        max_value=float(HA_MAX),
        value=float(default_ha),
        step=1.0,
        label_visibility="collapsed",
        key="ha_slider",
        help="Ratio of electromagnetic force to viscous force in the micro-channel flow.",
    )
    ha_num = st.number_input(
        "Exact Ha",
        min_value=float(HA_MIN),
        max_value=float(HA_MAX),
        value=float(ha_slider),
        step=0.5,
        label_visibility="collapsed",
        key="ha_num",
    )
    ha_val = float(ha_num)
    st.caption("ℹ️ *Hartmann Number (0 - 60): Measures applied magnetic field strength suppressing flow turbulence.*")

    st.markdown("---")

    # Input 2: Nanoparticle Volume Fraction (phi)
    st.markdown(r"**2. Nanoparticle Fraction ($\phi$)**")
    default_phi = st.session_state.get("phi_input", PHI_MEAN)
    phi_slider = st.slider(
        "phi slider",
        min_value=float(PHI_MIN),
        max_value=float(PHI_MAX),
        value=float(default_phi),
        step=0.002,
        format="%.3f",
        label_visibility="collapsed",
        key="phi_slider",
        help="Volumetric fraction of nanoparticles suspended in the dielectric coolant (1% to 5%).",
    )
    phi_num = st.number_input(
        "Exact phi",
        min_value=float(PHI_MIN),
        max_value=float(PHI_MAX),
        value=float(phi_slider),
        step=0.001,
        format="%.3f",
        label_visibility="collapsed",
        key="phi_num",
    )
    phi_val = float(phi_num)
    st.caption("ℹ️ *Nanoparticle Volume Fraction (0.01 - 0.05): Higher values boost coolant thermal conductivity.*")

    st.markdown("---")

    # Input 3: Inlet Flow Velocity (u_in)
    st.markdown("**3. Inlet Flow Velocity ($u_{in}$) [m/s]**")
    default_uin = st.session_state.get("uin_input", UIN_MEAN)
    uin_slider = st.slider(
        "u_in slider",
        min_value=float(UIN_MIN),
        max_value=float(UIN_MAX),
        value=float(default_uin),
        step=0.01,
        format="%.3f",
        label_visibility="collapsed",
        key="uin_slider",
        help="Coolant fluid velocity entering the micro-channels in meters per second.",
    )
    uin_num = st.number_input(
        "Exact u_in",
        min_value=float(UIN_MIN),
        max_value=float(UIN_MAX),
        value=float(uin_slider),
        step=0.005,
        format="%.3f",
        label_visibility="collapsed",
        key="uin_num",
    )
    uin_val = float(uin_num)
    st.caption("ℹ️ *Inlet Flow Velocity (0.05 - 0.30 m/s): Strongest cooling lever; higher velocity speeds convective heat transfer.*")

    st.divider()
    predict_btn = st.button("🚀 Predict T_max", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Main Page: Prediction Execution & Results Card
# -----------------------------------------------------------------------------
t_pred, condition = run_prediction(model, features, ha_val, phi_val, uin_val)
meta = RISK_METADATA[condition]

# Highlight prediction section
st.markdown("### 🎯 Real-Time Thermal Prediction")

card_html = f"""
<div class="metric-card" style="background: {meta['bg']}; border-color: {meta['border']};">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div style="font-size: 0.95rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
                Predicted Maximum Battery Surface Temperature
            </div>
            <div class="metric-val" style="color: {meta['color']};">
                {t_pred:.2f} <span style="font-size: 1.8rem; font-weight: 600;">°C</span>
            </div>
            <div class="risk-badge" style="background: {meta['color']}22; color: {meta['color']}; border: 1px solid {meta['color']};">
                {meta['icon']} Risk Condition: {condition}
            </div>
        </div>
        <div style="max-width: 480px; text-align: left; padding: 0.8rem 1rem; background: rgba(0,0,0,0.25); border-radius: 10px; border-left: 4px solid {meta['color']};">
            <div style="font-weight: 600; color: #f8fafc; font-size: 0.92rem; margin-bottom: 0.25rem;">Thermal Condition Assessment:</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45;">
                {meta['desc']}
            </div>
        </div>
    </div>
</div>
"""
st.markdown(card_html, unsafe_allow_html=True)

# Quick Operational Summary Metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-label">Hartmann Number (Ha)</div>
            <div class="stat-value">{ha_val:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-label">Nanoparticle Frac (phi)</div>
            <div class="stat-value">{phi_val:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-label">Inlet Velocity (u_in)</div>
            <div class="stat-value">{uin_val:.3f} m/s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m4:
    delta_from_mean = t_pred - 43.46
    sign = "+" if delta_from_mean >= 0 else ""
    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-label">Delta from Mean (43.46°C)</div>
            <div class="stat-value" style="color: {'#ef4444' if delta_from_mean > 5 else '#10b981' if delta_from_mean < -3 else '#f59e0b'};">
                {sign}{delta_from_mean:.2f}°C
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_perf, tab_importance, tab_sensitivity, tab_eda, tab_physics = st.tabs(
    [
        "📊 Model Performance",
        "⚖️ Feature Importance",
        "📈 Sensitivity Analysis",
        "🖼️ EDA & Thermal Plots",
        "🔬 Thermal Physics Context",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: Model Performance
# -----------------------------------------------------------------------------
with tab_perf:
    st.markdown("### 🧪 Model Evaluation & Benchmark Comparison")
    st.caption("Trained on 5,000 simulation samples with an 80/20 train-test split (`random_state=42`).")

    # Load actual comparison CSV
    comp_file = RESULTS_DIR / "model_comparison.csv"
    sweep_file = RESULTS_DIR / "rf_config_sweep.csv"
    sample_file = RESULTS_DIR / "sample_predictions.csv"

    # KPI summary cards for final chosen Random Forest
    rf_mae = 0.7853
    rf_rmse = 0.9865
    rf_r2 = 0.9768

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Model Architecture", "Random Forest", "200 Trees")
    kpi2.metric("Coefficient of Determination (R²)", f"{rf_r2:.4f}", "+0.77% vs Linear")
    kpi3.metric("Mean Absolute Error (MAE)", f"{rf_mae:.4f} °C", "-0.111 °C error")
    kpi4.metric("Root Mean Squared Error (RMSE)", f"{rf_rmse:.4f} °C", "Penalizes outliers")

    st.markdown("#### 1. Model Benchmark: Random Forest vs Linear Regression")
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        st.dataframe(
            df_comp.style.format({
                "MAE": "{:.4f} °C",
                "MSE": "{:.4f} °C²",
                "RMSE": "{:.4f} °C",
                "R2": "{:.4f}",
            }).highlight_max(subset=["R2"], color="rgba(16, 185, 129, 0.25)")
              .highlight_min(subset=["MAE", "RMSE"], color="rgba(16, 185, 129, 0.25)"),
            use_container_width=True,
        )
    else:
        st.info("Results file `results/model_comparison.csv` not found.")

    st.markdown("#### 2. Random Forest Tree Count Sweep (`n_estimators`)")
    if sweep_file.exists():
        df_sweep = pd.read_csv(sweep_file)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(
                df_sweep.style.format({
                    "MAE": "{:.4f}",
                    "MSE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "R2": "{:.4f}",
                }),
                use_container_width=True,
            )
        with c2:
            st.line_chart(df_sweep.set_index("n_estimators")[["MAE", "RMSE"]])
        st.caption("Note: Performance plateaus around 200 trees (MAE ~ 0.785°C, R² = 0.9768).")

    st.markdown("#### 3. Test Set Sample Predictions (Actual vs Predicted)")
    if sample_file.exists():
        df_sample = pd.read_csv(sample_file)
        st.dataframe(
            df_sample.style.format({
                "Actual_T_max": "{:.3f} °C",
                "Predicted_T_max": "{:.3f} °C",
                "Absolute_Error": "{:.3f} °C",
            }).bar(subset=["Absolute_Error"], color="rgba(239, 68, 68, 0.4)"),
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: Feature Importance
# -----------------------------------------------------------------------------
with tab_importance:
    st.markdown("### 🔍 Feature Importance & Physical Impact")
    st.caption("Computed via Gini impurity reduction from the final 200-tree Random Forest Regressor.")

    fi_file = RESULTS_DIR / "feature_importance.csv"
    if fi_file.exists():
        df_fi = pd.read_csv(fi_file)
        if "importance" in df_fi.columns:
            # Handle nameless first column if saved with index
            feat_col = df_fi.columns[0] if df_fi.columns[0] != "importance" else "Feature"
            df_fi = df_fi.rename(columns={feat_col: "Feature"})
        
        c_fi1, c_fi2 = st.columns([1, 1])
        with c_fi1:
            st.dataframe(
                df_fi.style.format({"importance": "{:.4%}"})
                     .bar(subset=["importance"], color="rgba(56, 189, 248, 0.5)"),
                use_container_width=True,
            )
        with c_fi2:
            st.bar_chart(df_fi.set_index("Feature")["importance"])

    st.markdown("#### 💡 Physical Engineering Takeaways")
    st.markdown(
        r"""
        - **Inlet Velocity ($u_{in}$) dominates with ~68.70% importance:**
          Advective coolant transport is the primary heat extraction mechanism. Increasing velocity directly shrinks thermal boundary layers and enhances heat removal from micro-channels.
        - **Nanoparticle Volume Fraction ($\phi$) accounts for ~29.22% importance:**
          Dispersing nanoparticles elevates the nanofluid's thermal conductivity, drastically improving heat exchange capacity between the battery wall and fluid.
        - **Hartmann Number ($Ha$) contributes ~2.09% importance:**
          MHD forces dampen transverse vortices and fluid turbulence, subtly modifying the velocity profile and heat dissipation rates.
        """
    )

# -----------------------------------------------------------------------------
# TAB 3: Sensitivity Analysis
# -----------------------------------------------------------------------------
with tab_sensitivity:
    st.markdown("### 📈 Single-Variable Sensitivity Explorer")
    st.caption("Simulate how $T_{max}$ responds when varying one thermal parameter while keeping the other two fixed at current sidebar settings.")

    var_choice = st.radio(
        "Select variable to sweep:",
        ["Inlet Velocity (u_in)", "Nanoparticle Fraction (phi)", "Hartmann Number (Ha)"],
        horizontal=True,
    )

    if var_choice == "Inlet Velocity (u_in)":
        sweep_range = np.linspace(UIN_MIN, UIN_MAX, 50)
        sweep_data = [
            {"u_in": u, "Predicted_T_max": run_prediction(model, features, ha_val, phi_val, u)[0]}
            for u in sweep_range
        ]
        sweep_df = pd.DataFrame(sweep_data).set_index("u_in")
        st.line_chart(sweep_df)
        st.caption("Coolant velocity sweep: Demonstrates sharp non-linear cooling as inlet velocity increases.")

    elif var_choice == "Nanoparticle Fraction (phi)":
        sweep_range = np.linspace(PHI_MIN, PHI_MAX, 50)
        sweep_data = [
            {"phi": p, "Predicted_T_max": run_prediction(model, features, ha_val, p, uin_val)[0]}
            for p in sweep_range
        ]
        sweep_df = pd.DataFrame(sweep_data).set_index("phi")
        st.line_chart(sweep_df)
        st.caption("Nanoparticle fraction sweep: Shows thermal conductivity enhancement curve.")

    else:
        sweep_range = np.linspace(HA_MIN, HA_MAX, 60)
        sweep_data = [
            {"Ha": h, "Predicted_T_max": run_prediction(model, features, h, phi_val, uin_val)[0]}
            for h in sweep_range
        ]
        sweep_df = pd.DataFrame(sweep_data).set_index("Ha")
        st.line_chart(sweep_df)
        st.caption("Hartmann number sweep: Illustrates the magnetic field damping influence.")

# -----------------------------------------------------------------------------
# TAB 4: EDA & Thermal Plots
# -----------------------------------------------------------------------------
with tab_eda:
    st.markdown("### 🖼️ Exploratory Data Analysis & Diagnostic Graphs")
    st.caption("Actual generated figures from `analysis.py` stored in the `results/` folder.")

    plot_options = {
        "Target Distribution (T_max)": "target_distribution.png",
        "Actual vs Predicted T_max": "actual_vs_predicted.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Feature Importance Plot": "feature_importance.png",
        "Inlet Velocity vs T_max": "velocity_vs_tmax.png",
        "Nanoparticle Fraction vs T_max": "phi_vs_tmax.png",
        "Hartmann Number vs T_max": "ha_vs_tmax.png",
    }

    selected_plot_label = st.selectbox("Select figure to inspect:", list(plot_options.keys()))
    plot_filename = plot_options[selected_plot_label]
    img_path = RESULTS_DIR / plot_filename

    if img_path.exists():
        st.image(str(img_path), caption=f"results/{plot_filename}", use_container_width=True)
    else:
        st.warning(f"Image `{plot_filename}` was not found under `results/`.")

# -----------------------------------------------------------------------------
# TAB 5: Thermal Physics & System Context
# -----------------------------------------------------------------------------
with tab_physics:
    st.markdown("### 🔬 Battery Thermal Management Architecture (KJS-CES-02)")
    
    st.markdown(
        r"""
        #### Context & Motivation
        Electric vehicle lithium-ion battery packs exhibit optimal electrochemical efficiency and cycle longevity 
        within **25°C to 40°C**. Operating beyond **45°C** accelerates electrolyte decomposition and solid electrolyte interphase (SEI) 
        breakdown, while temperatures above **55°C** pose thermal runaway hazards.
        
        #### Micro-Channel Liquid Cooling with MHD Nanofluids
        - **Micro-channels**: Built into cold plates beneath or between battery cells, providing compact, high-surface-area heat extraction.
        - **Nanofluids**: Suspending metal or metal-oxide nanoparticles in the base dielectric coolant elevates effective thermal conductivity ($k_{eff}$) beyond conventional ethylene-glycol / water mixes.
        - **Magnetohydrodynamics (MHD)**: External magnetic fields govern fluid boundary layer growth via Lorentz forces ($Ha = B L \sqrt{\sigma/\mu}$), suppressing unwanted flow instabilities without moving parts.

        #### Classification Threshold Reference
        """
    )

    st.table(
        pd.DataFrame([
            {"Threshold Limit": "< 35.0 °C", "Risk Level": "Safe", "Operating Action": "Optimal condition; standard coolant circulation."},
            {"Threshold Limit": "35.0 - 45.0 °C", "Risk Level": "Moderate", "Operating Action": "Elevated thermal load; moderate pump speed boost."},
            {"Threshold Limit": "45.0 - 55.0 °C", "Risk Level": "High", "Operating Action": "High thermal stress; initiate thermal throttling & high flow."},
            {"Threshold Limit": "≥ 55.0 °C", "Risk Level": "Critical", "Operating Action": "Severe runaway risk! Maximum pump power / power cutoff."},
        ])
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 0.5rem 0;">
        ⚡ <strong>KJS-CES-02</strong> — AI-Enabled Micro-Channel Battery Thermal Management System for EVs 
        | Model: Random Forest Regressor (200 Estimators)
    </div>
    """,
    unsafe_allow_html=True,
)
