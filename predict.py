"""
predict.py
----------
Interactive prediction tool for EV Battery Maximum Temperature.

Loads the trained Random Forest model and predicts T_max from user-provided
values of Ha, phi, and u_in, then classifies the result into a simple
thermal condition band.
"""

import sys
import joblib
import pandas as pd

MODEL_PATH = "battery_temperature_model.pkl"

# Thermal condition thresholds (°C) — adjust to match your dataset's range
# if your actual T_max distribution differs.
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


def load_model(path=MODEL_PATH):
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        print(f"Error: model file '{path}' not found. Run train.py first.")
        sys.exit(1)
    return bundle["model"], bundle["features"]


def predict_one(model, features, Ha, phi, u_in):
    X = pd.DataFrame([[Ha, phi, u_in]], columns=features)
    t_max = model.predict(X)[0]
    condition = classify(t_max)
    return t_max, condition


def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a numeric value.")


def interactive_loop(model, features):
    print("=" * 60)
    print(" EV Battery Maximum Temperature Prediction")
    print(" Model: Random Forest Regression")
    print(" Inputs: Hartmann Number (Ha), Nanoparticle Volume Fraction (phi),")
    print("         Inlet Flow Velocity (u_in)")
    print("=" * 60)

    while True:
        print("\nEnter operating conditions (or type 'q' to quit):")
        raw = input("Ha = ")
        if raw.strip().lower() == "q":
            break
        Ha = float(raw)
        phi = get_float("phi = ")
        u_in = get_float("u_in = ")

        t_max, condition = predict_one(model, features, Ha, phi, u_in)

        print("\n--- Prediction Result ---")
        print(f"  Predicted Maximum Battery Temperature = {t_max:.2f} °C")
        print(f"  Thermal Condition                     = {condition}")
        print("--------------------------")


def demo(model, features):
    """Run a few preset cases to show that predictions change with inputs."""
    cases = [
        {"Ha": 30, "phi": 0.03, "u_in": 0.20},
        {"Ha": 55, "phi": 0.01, "u_in": 0.10},
        {"Ha": 10, "phi": 0.05, "u_in": 0.45},
    ]
    print("\n=== Demonstration: Predictions for Several Input Combinations ===")
    for i, case in enumerate(cases, start=1):
        t_max, condition = predict_one(model, features, **case)
        print(f"\nCase {i}: Ha={case['Ha']}, phi={case['phi']}, u_in={case['u_in']}")
        print(f"  Predicted T_max = {t_max:.2f} °C  ->  Condition: {condition}")


if __name__ == "__main__":
    model, features = load_model()

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo(model, features)
    else:
        demo(model, features)
        interactive_loop(model, features)
