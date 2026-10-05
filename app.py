"""
app.py
------
Simple Flask web app for EV Battery Maximum Temperature Prediction.
Loads the saved Random Forest model and serves a single-page frontend.
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder="static")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "battery_temperature_model.pkl")

# Load model once at startup
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

# Thermal condition thresholds
THRESHOLDS = [
    (35.0, "Safe"),
    (45.0, "Moderate"),
    (55.0, "High"),
]
CRITICAL_LABEL = "Critical"


def classify(t_max):
    for limit, label in THRESHOLDS:
        if t_max < limit:
            return label
    return CRITICAL_LABEL


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        ha = float(data["ha"])
        phi = float(data["phi"])
        u_in = float(data["u_in"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input. Provide ha, phi, and u_in as numbers."}), 400

    X = pd.DataFrame([[ha, phi, u_in]], columns=features)
    t_max = model.predict(X)[0]
    condition = classify(t_max)

    return jsonify({
        "t_max": round(float(t_max), 2),
        "condition": condition,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
