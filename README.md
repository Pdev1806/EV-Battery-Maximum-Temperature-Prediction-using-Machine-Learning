# EV Battery Maximum Temperature Prediction using Machine Learning

**AI Use Case:** KJS-CES-02 — AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles
**ML Technique:** Random Forest Regression

---

## 1. Problem Statement

Electric vehicle (EV) batteries generate significant heat during operation.
Excessive battery temperature accelerates degradation, reduces performance,
and in extreme cases creates safety risks. Micro-channel cooling systems
regulate battery temperature, and predicting the **maximum battery surface
temperature (`T_max`)** in advance allows a thermal management system to
adjust cooling parameters proactively rather than reactively.

This project implements one clearly scoped ML component of the broader
KJS-CES-02 use case: predicting `T_max` from a small set of thermal-management
input parameters using Random Forest Regression.

## 2. Objectives

1. Analyze EV battery thermal-management data.
2. Identify variables related to maximum battery temperature.
3. Develop a Random Forest Regression model.
4. Predict maximum battery surface temperature from input conditions.
5. Evaluate model performance using standard regression metrics.
6. Analyze feature importance for engineering interpretation.
7. Demonstrate temperature prediction using user-provided inputs.

## 3. Dataset

- **Source:** Hugging Face — `Dijo-404/mhd-nanofluid-ev-thermal-dataset`
- **Size:** 5,000 rows × 15 columns, CSV format
- **Type:** Physics-based (simulation-derived) tabular regression data

**Input features used in this model:**

| Feature | Meaning |
|---|---|
| `Ha` | Hartmann Number |
| `phi` | Nanoparticle Volume Fraction |
| `u_in` | Inlet Flow Velocity |

**Target:** `T_max` — Maximum Battery Surface Temperature

The remaining 11 columns (`Nu`, `S_gen`, `k_ratio`, `BL_suppression`, `delta_T`,
`Re`, `Pr`, `h_conv`, `S_thermal_frac`, `S_viscous_frac`, `S_magnetic_frac`)
are used only for exploratory analysis and correlation context — not as
model inputs.

> Place your actual downloaded dataset at `training_data.csv` in this folder
> before running any script.

## 4. Project Structure

```text
miniproject/
│
├── training_data.csv                  # dataset (place your own here)
├── train.py                           # trains & evaluates the model
├── analysis.py                        # EDA + all graphs
├── predict.py                         # interactive prediction demo
├── battery_temperature_model.pkl      # saved model (created by train.py)
├── requirements.txt
├── README.md
└── results/                           # all generated graphs & tables
    ├── target_distribution.png
    ├── ha_vs_tmax.png
    ├── phi_vs_tmax.png
    ├── velocity_vs_tmax.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    ├── feature_importance.png
    ├── rf_config_sweep.csv
    ├── model_comparison.csv
    ├── feature_importance.csv
    └── sample_predictions.csv
```

## 5. Setup

```bash
pip install -r requirements.txt
```

## 6. Usage

Run in this order:

```bash
# 1. Train the model (also runs the n_estimators sweep and Linear Regression baseline)
python train.py

# 2. Run EDA and generate all graphs (uses the saved model for graphs 6 and 7)
python analysis.py

# 3. Try interactive predictions
python predict.py
```

`predict.py` first runs a short demo with three preset input combinations,
then lets you enter your own `Ha`, `phi`, and `u_in` values interactively
(type `q` to quit).

## 7. Methodology

```text
Raw Dataset
    ↓
Data Quality Check (missing values, duplicates, dtypes)
    ↓
Feature Selection (Ha, phi, u_in → T_max)
    ↓
80/20 Train/Test Split (random_state = 42)
    ↓
Random Forest Regression
    ↓
Evaluation (MAE, MSE, RMSE, R²)
```

**Why Random Forest Regression?**
Random Forest is an ensemble of decision trees, each trained on a bootstrap
sample of the data with a random subset of features considered at each
split. Predictions are averaged across all trees. This reduces overfitting
compared to a single decision tree, handles non-linear relationships well,
requires no feature scaling, and directly provides feature importance
scores — useful for engineering interpretation of which input most affects
`T_max`.

No feature scaling/normalization is applied, since tree-based models do not
require it (unlike distance-based algorithms such as SVM or KNN).

## 8. Results

`train.py` reports:

- A **configuration sweep** over `n_estimators` ∈ {50, 100, 200, 300} — see `results/rf_config_sweep.csv`
- A **baseline comparison** against Linear Regression — see `results/model_comparison.csv`
- **Feature importance** ranking — see `results/feature_importance.csv` / `.png`
- A **sample actual-vs-predicted table** for 10 test rows — see `results/sample_predictions.csv`

The final saved model uses 200 trees, an 80/20 split, and `random_state=42`
for reproducibility. Exact metric values depend on your actual dataset —
re-run `train.py` and read the printed summary at the end for your numbers.

## 9. Analysis Notes (fill in with your actual numbers after running train.py)

- **MAE** — average absolute difference (°C) between actual and predicted `T_max`.
- **RMSE** — same unit as `T_max`, penalizes larger errors more than MAE.
- **R²** — proportion of variance in `T_max` explained by the model; closer to 1 is better.
- Compare the Random Forest result to the Linear Regression baseline in
  `results/model_comparison.csv` to state whether the added model complexity
  was justified for this dataset.
- Use `results/feature_importance.png` to discuss, in engineering terms,
  which of `Ha`, `phi`, or `u_in` most strongly drives `T_max`, and why that
  makes physical sense (e.g., flow velocity affecting convective cooling).

## 10. Limitations

1. The dataset is physics-based/simulation-derived rather than measured from
   a real-world EV fleet.
2. Only three input features are used in this mini-project; the full
   KJS-CES-02 use case involves additional variables (hotspot location,
   thermal gradients, coolant flow rate, etc.).
3. Real EV battery systems have additional operating conditions not captured here.
4. Battery aging effects on thermal behaviour are not modeled.
5. Real-world sensor noise is not represented in simulation data.
6. Performance on real experimental or road-condition data would require
   separate validation before any deployment claim.

This model is a proof-of-concept for one ML component of the use case, not
a deployment-ready system.

## 11. Future Scope

- Incorporate experimental battery sensor data.
- Include battery aging as a variable.
- Add additional thermal variables (coolant flow rate, ambient temperature).
- Predict hotspot locations and full temperature distribution, not just the maximum.
- Integrate real-time sensor data and connect to a Battery Management System.
- Develop a digital-twin interface for continuous monitoring.
- Optimize coolant flow conditions based on predicted temperature.

## 12. Conclusion

This project demonstrates a complete, reproducible ML pipeline — data
analysis, feature selection, Random Forest Regression, evaluation, and
interactive prediction — for one clearly defined component of the KJS-CES-02
battery thermal-management use case: predicting maximum battery surface
temperature from Hartmann Number, nanoparticle volume fraction, and inlet
flow velocity.
