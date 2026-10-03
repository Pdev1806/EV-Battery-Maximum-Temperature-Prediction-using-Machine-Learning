"""
train.py
--------
EV Battery Maximum Temperature Prediction (KJS-CES-02)

Trains a Random Forest Regression model to predict T_max (maximum battery
surface temperature) from three thermal-management input parameters:
    Ha    - Hartmann Number
    phi   - Nanoparticle Volume Fraction
    u_in  - Inlet Flow Velocity

Also:
    - Sweeps a few n_estimators configurations and reports MAE/RMSE/R^2 for each
    - Trains a Linear Regression baseline for comparison
    - Saves the final chosen Random Forest model to battery_temperature_model.pkl
    - Writes a small actual-vs-predicted sample table to results/sample_predictions.csv
    - Writes the config sweep + baseline comparison to results/model_comparison.csv
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "training_data.csv"
MODEL_PATH = "battery_temperature_model.pkl"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = ["Ha", "phi", "u_in"]
TARGET = "T_max"

RANDOM_STATE = 42
FINAL_N_ESTIMATORS = 200  # chosen configuration used for the saved model


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("=" * 60)
    print("Dataset split")
    print(f"  Training rows: {len(X_train)}")
    print(f"  Testing rows : {len(X_test)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Random Forest configuration sweep (n_estimators)
    # ------------------------------------------------------------------
    configs = [50, 100, 200, 300]
    sweep_rows = []
    best_model = None
    best_r2 = -np.inf

    print("\nRandom Forest configuration sweep:")
    for n in configs:
        rf = RandomForestRegressor(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        metrics = evaluate(y_test, preds)
        sweep_rows.append({"n_estimators": n, **metrics})
        print(f"  n_estimators={n:<4d} MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}")

        if n == FINAL_N_ESTIMATORS:
            best_model = rf
            best_r2 = metrics["R2"]
            final_metrics = metrics
            final_preds = preds

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS_DIR / "rf_config_sweep.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Linear Regression baseline
    # ------------------------------------------------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_metrics = evaluate(y_test, lr_preds)

    print("\nBaseline comparison:")
    print(f"  Linear Regression : MAE={lr_metrics['MAE']:.4f}  "
          f"RMSE={lr_metrics['RMSE']:.4f}  R2={lr_metrics['R2']:.4f}")
    print(f"  Random Forest ({FINAL_N_ESTIMATORS} trees): "
          f"MAE={final_metrics['MAE']:.4f}  RMSE={final_metrics['RMSE']:.4f}  "
          f"R2={final_metrics['R2']:.4f}")

    comparison_df = pd.DataFrame([
        {"Model": "Linear Regression", **lr_metrics},
        {"Model": f"Random Forest ({FINAL_N_ESTIMATORS} trees)", **final_metrics},
    ])
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Feature importance (from the final Random Forest)
    # ------------------------------------------------------------------
    importances = pd.Series(best_model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(RESULTS_DIR / "feature_importance.csv", header=["importance"])
    print("\nFeature importance:")
    for feat, val in importances.items():
        print(f"  {feat:6s}: {val:.4f}")

    # ------------------------------------------------------------------
    # 4. Sample actual-vs-predicted table
    # ------------------------------------------------------------------
    sample = pd.DataFrame({
        "Actual_T_max": y_test.values[:10],
        "Predicted_T_max": final_preds[:10],
    })
    sample["Absolute_Error"] = (sample["Actual_T_max"] - sample["Predicted_T_max"]).abs()
    sample.to_csv(RESULTS_DIR / "sample_predictions.csv", index=False)
    print("\nSample predictions:")
    print(sample.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Save final model
    # ------------------------------------------------------------------
    joblib.dump({"model": best_model, "features": FEATURES}, MODEL_PATH)
    print(f"\nFinal model saved to: {MODEL_PATH}")
    print("=" * 60)
    print("FINAL MODEL METRICS (Random Forest, "
          f"{FINAL_N_ESTIMATORS} trees, 80/20 split, random_state={RANDOM_STATE})")
    for k, v in final_metrics.items():
        print(f"  {k:4s} = {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
