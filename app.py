"""
app.py
------
Flask Web Application for EV Battery Maximum Temperature Prediction (KJS-CES-02).
AI-Enabled Micro-Channel Battery Thermal Management System.

Features:
- Single startup model load (Random Forest Regressor, 200 trees)
- REST API for predictions and what-if sensitivity sweeps
- Static dataset benchmarks & diagnostic plot serving
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure dependencies are available; if invoked with an unconfigured Python (e.g. 3.14),
# automatically re-launch using the installed Python 3.13 environment where all ML packages reside.
try:
    import joblib
    import flask
    import sklearn
    import pandas as pd
    import numpy as np
except ModuleNotFoundError:
    py313_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python313\python.exe")
    if os.path.exists(py313_path) and os.path.abspath(sys.executable) != os.path.abspath(py313_path):
        print(f"[*] Re-launching with Python 3.13 runtime ({py313_path})...")
        sys.exit(subprocess.call([py313_path] + sys.argv))
    import shutil
    py_cmd = shutil.which("py")
    if py_cmd:
        print("[*] Re-launching with 'py -3.13'...")
        sys.exit(subprocess.call([py_cmd, "-3.13"] + sys.argv))
    raise

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "battery_temperature_model.pkl"
DATA_PATH = BASE_DIR / "training_data.csv"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"

# Exact features and order from train.py
FEATURES = ["Ha", "phi", "u_in"]
TARGET = "T_max"

# Exact risk classification bands and labels from predict.py
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

# Training dataset domain bounds (from training_data.csv, 5,000 samples)
LIMITS = {
    "Ha": {"min": 0.0, "max": 60.0, "mean": 30.0, "step": 1.0, "default": 30.0},
    "phi": {"min": 0.01, "max": 0.05, "mean": 0.03, "step": 0.001, "default": 0.030},
    "u_in": {"min": 0.05, "max": 0.30, "mean": 0.175, "step": 0.005, "default": 0.175},
}
T_MAX_MEAN = 43.46

# -----------------------------------------------------------------------------
# Startup Loading (Model & Static Data Cached Once)
# -----------------------------------------------------------------------------
MODEL = None
STATIC_CACHE = {}


def load_model_once():
    """Load model once at application boot."""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model '{MODEL_PATH}' was not found. Run train.py first.")
        bundle = joblib.load(MODEL_PATH)
        MODEL = bundle["model"]
    return MODEL


def load_static_cache():
    """Read static CSV tables and compute dataset statistics once at startup."""
    global STATIC_CACHE
    if STATIC_CACHE:
        return STATIC_CACHE

    cache = {
        "model_comparison": [],
        "rf_config_sweep": [],
        "feature_importance": [],
        "sample_predictions": [],
        "correlations": {},
        "limits": LIMITS,
        "mean_tmax": T_MAX_MEAN,
    }

    # Model comparison
    comp_file = RESULTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        try:
            df_comp = pd.read_csv(comp_file)
            cache["model_comparison"] = df_comp.to_dict(orient="records")
        except Exception:
            pass

    # RF sweep
    sweep_file = RESULTS_DIR / "rf_config_sweep.csv"
    if sweep_file.exists():
        try:
            df_sweep = pd.read_csv(sweep_file)
            cache["rf_config_sweep"] = df_sweep.to_dict(orient="records")
        except Exception:
            pass

    # Feature importance
    fi_file = RESULTS_DIR / "feature_importance.csv"
    if fi_file.exists():
        try:
            df_fi = pd.read_csv(fi_file)
            feat_col = df_fi.columns[0]
            cache["feature_importance"] = [
                {"feature": str(row[feat_col]), "importance": float(row["importance"])}
                for _, row in df_fi.iterrows()
            ]
        except Exception:
            pass

    # Sample predictions
    sp_file = RESULTS_DIR / "sample_predictions.csv"
    if sp_file.exists():
        try:
            df_sp = pd.read_csv(sp_file)
            cache["sample_predictions"] = df_sp.to_dict(orient="records")
        except Exception:
            pass

    # Dataset correlation matrix
    if DATA_PATH.exists():
        try:
            df_data = pd.read_csv(DATA_PATH)
            corr_cols = [c for c in ["Ha", "phi", "u_in", "T_max", "Nu", "Re", "Pr", "h_conv"] if c in df_data.columns]
            corr_matrix = df_data[corr_cols].corr().round(4)
            cache["correlations"] = {
                "columns": corr_cols,
                "values": corr_matrix.to_dict(orient="split")["data"],
            }
        except Exception:
            pass

    STATIC_CACHE = cache
    return STATIC_CACHE


# Initial boot load
load_model_once()
load_static_cache()


# -----------------------------------------------------------------------------
# Business Logic Helpers
# -----------------------------------------------------------------------------
def classify_tmax(t_max: float) -> str:
    """Classify predicted temperature using exact bands from predict.py."""
    for limit, label in THRESHOLDS:
        if t_max < limit:
            return label
    return CRITICAL_LABEL


def validate_input(data: dict):
    """
    Validates input parameters Ha, phi, u_in.
    Returns (cleaned_dict, error_string).
    """
    if not isinstance(data, dict):
        return None, "Request payload must be a JSON object."

    cleaned = {}
    keys_map = {
        "Ha": ["ha", "Ha"],
        "phi": ["phi", "Phi"],
        "u_in": ["u_in", "uin", "uIn", "velocity"],
    }

    for standard_name, aliases in keys_map.items():
        found = False
        val = None
        for alias in aliases:
            if alias in data:
                val = data[alias]
                found = True
                break

        if not found or val is None or val == "":
            return None, f"Missing required parameter '{standard_name}'."

        try:
            float_val = float(val)
        except (ValueError, TypeError):
            return None, f"Parameter '{standard_name}' must be a valid numeric value."

        # Domain boundary check
        param_min = LIMITS[standard_name]["min"]
        param_max = LIMITS[standard_name]["max"]
        if float_val < param_min - 1e-4 or float_val > param_max + 1e-4:
            return None, (
                f"Parameter '{standard_name}' ({float_val}) is outside the training data range "
                f"[{param_min}, {param_max}]."
            )

        cleaned[standard_name] = float_val

    return cleaned, None


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serves the main application page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts maximum battery temperature (T_max) from Ha, phi, u_in.
    Returns predicted value, risk classification, and thermal assessment.
    """
    try:
        req_data = request.get_json(silent=True)
        if req_data is None:
            return jsonify({"status": "error", "error": "Invalid or missing JSON payload."}), 400

        cleaned, err = validate_input(req_data)
        if err:
            return jsonify({"status": "error", "error": err}), 400

        # Construct DataFrame in exact feature order from train.py
        X = pd.DataFrame(
            [[cleaned["Ha"], cleaned["phi"], cleaned["u_in"]]],
            columns=FEATURES,
        )

        model = load_model_once()
        t_max = float(model.predict(X)[0])
        condition = classify_tmax(t_max)
        meta = RISK_METADATA[condition]
        delta_mean = t_max - T_MAX_MEAN

        return jsonify({
            "status": "success",
            "t_max": round(t_max, 2),
            "condition": condition,
            "color": meta["color"],
            "bg": meta["bg"],
            "border": meta["border"],
            "icon": meta["icon"],
            "explanation": meta["desc"],
            "delta_mean": round(delta_mean, 2),
            "inputs": {
                "Ha": cleaned["Ha"],
                "phi": cleaned["phi"],
                "u_in": cleaned["u_in"],
            },
        })

    except Exception as ex:
        return jsonify({"status": "error", "error": f"Internal prediction error: {str(ex)}"}), 500


@app.route("/whatif", methods=["POST"])
def whatif():
    """
    Computes a sensitivity curve for u_in across [0.05, 0.30] at given Ha and phi.
    """
    try:
        req_data = request.get_json(silent=True)
        if req_data is None:
            return jsonify({"status": "error", "error": "Invalid or missing JSON payload."}), 400

        # Extract Ha and phi
        ha_val = req_data.get("ha", req_data.get("Ha"))
        phi_val = req_data.get("phi", req_data.get("Phi"))

        if ha_val is None or phi_val is None:
            return jsonify({"status": "error", "error": "Must provide 'ha' and 'phi' for what-if simulation."}), 400

        try:
            ha_float = float(ha_val)
            phi_float = float(phi_val)
        except (ValueError, TypeError):
            return jsonify({"status": "error", "error": "Parameters 'ha' and 'phi' must be numbers."}), 400

        # Validate range
        if not (LIMITS["Ha"]["min"] <= ha_float <= LIMITS["Ha"]["max"]):
            return jsonify({"status": "error", "error": f"Ha must be within [{LIMITS['Ha']['min']}, {LIMITS['Ha']['max']}]."}), 400
        if not (LIMITS["phi"]["min"] <= phi_float <= LIMITS["phi"]["max"]):
            return jsonify({"status": "error", "error": f"phi must be within [{LIMITS['phi']['min']}, {LIMITS['phi']['max']}]."}), 400

        # Sweep 26 points across u_in range
        u_in_points = np.linspace(LIMITS["u_in"]["min"], LIMITS["u_in"]["max"], 26)
        sweep_rows = [[ha_float, phi_float, u] for u in u_in_points]
        X_sweep = pd.DataFrame(sweep_rows, columns=FEATURES)

        model = load_model_once()
        preds = model.predict(X_sweep)

        results = [
            {"u_in": round(float(u), 3), "t_max": round(float(p), 2), "condition": classify_tmax(float(p))}
            for u, p in zip(u_in_points, preds)
        ]

        return jsonify({
            "status": "success",
            "ha": ha_float,
            "phi": phi_float,
            "curve": results,
        })

    except Exception as ex:
        return jsonify({"status": "error", "error": f"Internal what-if calculation error: {str(ex)}"}), 500


@app.route("/api/static-data", methods=["GET"])
def api_static_data():
    """Returns pre-loaded static performance metrics and metadata."""
    cache = load_static_cache()
    return jsonify({"status": "success", "data": cache})


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Runs the Flask server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)
