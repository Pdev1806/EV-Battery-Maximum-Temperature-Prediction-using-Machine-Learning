"""
train_compare_models.py
------------------------
Compares multiple regression algorithms for EV Battery Maximum Temperature
Prediction (KJS-CES-02), using the same features and split as train.py:

    Features: Ha, phi, u_in
    Target:   T_max

Algorithms compared:
    - Linear Regression          (baseline)
    - Decision Tree Regressor
    - K-Nearest Neighbors (KNN)
    - Support Vector Regression (SVR)
    - Random Forest Regressor    (your existing model)
    - Gradient Boosting Regressor
    - XGBoost Regressor          (if installed, else skipped)

Also runs 5-fold cross-validation for every model (not just a single
80/20 split) and saves everything to results/algorithm_comparison.csv
and results/cv_scores.csv.

Run: pip install xgboost   (optional, script still works without it)
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[!] xgboost not installed — skipping XGBoost. Install with: pip install xgboost")

DATA_PATH = "training_data.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = ["Ha", "phi", "u_in"]
TARGET = "T_max"
RANDOM_STATE = 42


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def build_models():
    """
    Models that need feature scaling (KNN, SVR) are wrapped in a pipeline
    with StandardScaler. Tree-based models (Decision Tree, Random Forest,
    Gradient Boosting, XGBoost) and Linear Regression don't need scaling,
    but wrapping them does no harm and keeps the code uniform.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        "SVR (RBF kernel)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1)),
        "Random Forest (200 trees)": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbosity=0
        )
    return models


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = build_models()
    results = []
    cv_results = []

    print("=" * 70)
    print(f"{'Model':<28s}{'MAE':>10s}{'RMSE':>10s}{'R2':>10s}{'Train(s)':>10s}")
    print("-" * 70)

    for name, model in models.items():
        # Single train/test split (same as train.py, for direct comparison)
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        preds = model.predict(X_test)
        mae, rmse, r2 = evaluate(y_test, preds)
        results.append({
            "Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "Train_Time_s": train_time,
        })
        print(f"{name:<28s}{mae:>10.4f}{rmse:>10.4f}{r2:>10.4f}{train_time:>10.3f}")

        # 5-fold cross-validation (more robust than a single split)
        cv = cross_validate(
            model, X, y, cv=5,
            scoring=("neg_mean_absolute_error", "r2"),
        )
        cv_mae = -cv["test_neg_mean_absolute_error"]
        cv_r2 = cv["test_r2"]
        cv_results.append({
            "Model": name,
            "CV_MAE_mean": cv_mae.mean(), "CV_MAE_std": cv_mae.std(),
            "CV_R2_mean": cv_r2.mean(), "CV_R2_std": cv_r2.std(),
        })

    print("=" * 70)

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    cv_df = pd.DataFrame(cv_results).sort_values("CV_R2_mean", ascending=False)

    results_df.to_csv(RESULTS_DIR / "algorithm_comparison.csv", index=False)
    cv_df.to_csv(RESULTS_DIR / "cv_scores.csv", index=False)

    print("\nRanked by test-set R2 (single 80/20 split):")
    print(results_df.to_string(index=False))

    print("\nRanked by 5-fold cross-validation R2 (mean +/- std) — more robust:")
    print(cv_df.to_string(index=False))

    best = results_df.iloc[0]["Model"]
    print(f"\nBest single-split model: {best}")
    print("Saved: results/algorithm_comparison.csv, results/cv_scores.csv")


if __name__ == "__main__":
    main()