"""
analysis.py
-----------
Exploratory Data Analysis for the EV Battery Maximum Temperature dataset.

Produces:
    1. Console EDA summary (shape, dtypes, missing/duplicate values, describe())
    2. Graph 1: T_max distribution
    3. Graph 2: Ha vs T_max
    4. Graph 3: phi vs T_max
    5. Graph 4: u_in vs T_max
    6. Graph 5: Correlation heatmap
    7. Graph 6: Actual vs Predicted (requires a trained model; run train.py first)
    8. Graph 7: Feature importance (requires a trained model; run train.py first)

All figures are saved under results/.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = "training_data.csv"
MODEL_PATH = "battery_temperature_model.pkl"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = ["Ha", "phi", "u_in"]
TARGET = "T_max"
CORR_COLUMNS = ["Ha", "phi", "u_in", "T_max", "Nu", "Re", "Pr", "h_conv"]

sns.set_style("whitegrid")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_eda(df):
    section("1. Dataset Dimensions")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    section("2. Column Names")
    print(list(df.columns))

    section("3. Data Types")
    print(df.dtypes)

    section("4. Missing Values")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    section("5. Duplicate Rows")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")

    section("6. Descriptive Statistics (key variables)")
    print(df[["Ha", "phi", "u_in", "T_max"]].describe())

    section("7. Min / Max of Key Variables")
    for col in ["Ha", "phi", "u_in", "T_max"]:
        print(f"  {col:6s} -> min={df[col].min():.4f}, max={df[col].max():.4f}")


def plot_target_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.histplot(df[TARGET], kde=True, color="tomato")
    plt.title("Distribution of Maximum Battery Temperature (T_max)")
    plt.xlabel("T_max (°C)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "target_distribution.png", dpi=150)
    plt.close()
    print("Saved: results/target_distribution.png")


def plot_feature_vs_target(df, feature, filename, color):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[feature], y=df[TARGET], alpha=0.4, color=color)
    plt.title(f"{feature} vs T_max")
    plt.xlabel(feature)
    plt.ylabel("T_max (°C)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: results/{filename}")


def plot_correlation_heatmap(df):
    cols = [c for c in CORR_COLUMNS if c in df.columns]
    plt.figure(figsize=(8, 6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap of Key Thermal Variables")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: results/correlation_heatmap.png")


def plot_actual_vs_predicted(df):
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    features = model_bundle["features"]

    X = df[features]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    preds = model.predict(X_test)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, preds, alpha=0.4, color="teal")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Ideal prediction (y = x)")
    plt.xlabel("Actual T_max (°C)")
    plt.ylabel("Predicted T_max (°C)")
    plt.title("Actual vs Predicted Maximum Battery Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Saved: results/actual_vs_predicted.png")


def plot_feature_importance():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    features = model_bundle["features"]

    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="slateblue")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved: results/feature_importance.png")


def main():
    df = pd.read_csv(DATA_PATH)

    run_eda(df)

    section("Generating Graphs")
    plot_target_distribution(df)
    plot_feature_vs_target(df, "Ha", "ha_vs_tmax.png", "steelblue")
    plot_feature_vs_target(df, "phi", "phi_vs_tmax.png", "seagreen")
    plot_feature_vs_target(df, "u_in", "velocity_vs_tmax.png", "darkorange")
    plot_correlation_heatmap(df)

    model_path = Path(MODEL_PATH)
    if model_path.exists():
        plot_actual_vs_predicted(df)
        plot_feature_importance()
    else:
        print(f"\n[!] {MODEL_PATH} not found — run train.py first to generate "
              "the Actual-vs-Predicted and Feature-Importance graphs.")

    print("\nAll available graphs saved under the 'results/' folder.")


if __name__ == "__main__":
    main()
