"""
verify_flask_app.py
-------------------
Tests and verifies the Flask application:
1. Loads app test client.
2. Measures page load time of GET /.
3. Runs /predict on sample_predictions.csv cases and tests prediction latency.
4. Checks /whatif endpoint and /api/static-data endpoint.
5. Verifies error handling on bad inputs.
"""

import time
import json
import pandas as pd
from app import app, FEATURES

client = app.test_client()

print("=" * 65)
print("1. VERIFYING PAGE LOAD (GET /)")
print("=" * 65)
t0 = time.perf_counter()
res_index = client.get("/")
t_index = (time.perf_counter() - t0) * 1000
print(f"GET / Status Code : {res_index.status_code}")
print(f"GET / Response Time: {t_index:.2f} ms")
assert res_index.status_code == 200, "Index page failed to load!"
assert b"AI-Enabled Micro-Channel Battery Thermal Management" in res_index.data

print("\n" + "=" * 65)
print("2. VERIFYING /api/static-data")
print("=" * 65)
t0 = time.perf_counter()
res_meta = client.get("/api/static-data")
t_meta = (time.perf_counter() - t0) * 1000
print(f"GET /api/static-data Status : {res_meta.status_code}")
print(f"GET /api/static-data Latency: {t_meta:.2f} ms")
meta_data = res_meta.get_json()["data"]
print(f"Model comparison models: {[m['Model'] for m in meta_data['model_comparison']]}")
print(f"Feature importance items: {[f['feature'] for f in meta_data['feature_importance']]}")

print("\n" + "=" * 65)
print("3. VERIFYING PREDICTIONS & LATENCY ON TEST CASES")
print("=" * 65)
# Preset test cases from predict.py demo
demo_cases = [
    {"Ha": 30.0, "phi": 0.03, "u_in": 0.20, "expected_approx": 41.59, "condition": "Moderate"},
    {"Ha": 55.0, "phi": 0.01, "u_in": 0.10, "expected_approx": 55.45, "condition": "Critical"},
    {"Ha": 10.0, "phi": 0.05, "u_in": 0.28, "expected_approx": 30.33, "condition": "Safe"},
]

latencies = []
for i, case in enumerate(demo_cases, 1):
    payload = {"ha": case["Ha"], "phi": case["phi"], "u_in": case["u_in"]}
    t0 = time.perf_counter()
    resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    lat_ms = (time.perf_counter() - t0) * 1000
    latencies.append(lat_ms)
    
    data = resp.get_json()
    print(f"Case {i}: Ha={case['Ha']}, phi={case['phi']}, u_in={case['u_in']}")
    print(f"  -> Predicted T_max : {data['t_max']} °C")
    print(f"  -> Risk Condition  : {data['condition']}")
    print(f"  -> Latency         : {lat_ms:.2f} ms")
    assert resp.status_code == 200
    assert data["condition"] == case["condition"], f"Expected {case['condition']}, got {data['condition']}"

# Now test against test split from training_data.csv matching sample_predictions.csv
df_data = pd.read_csv("training_data.csv")
from sklearn.model_selection import train_test_split
X = df_data[FEATURES]
y = df_data["T_max"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

df_sample = pd.read_csv("results/sample_predictions.csv")
print("\nVerifying first 3 rows against results/sample_predictions.csv:")
for idx in range(3):
    row_feat = X_test.iloc[idx]
    actual_t = df_sample.iloc[idx]["Actual_T_max"]
    expected_pred = df_sample.iloc[idx]["Predicted_T_max"]
    
    payload = {"ha": float(row_feat["Ha"]), "phi": float(row_feat["phi"]), "u_in": float(row_feat["u_in"])}
    t0 = time.perf_counter()
    resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    lat_ms = (time.perf_counter() - t0) * 1000
    latencies.append(lat_ms)
    
    pred_res = resp.get_json()
    flask_t = pred_res["t_max"]
    diff = abs(flask_t - expected_pred)
    print(f"Row {idx+1}: Actual={actual_t:.2f}°C, Saved Pred={expected_pred:.2f}°C, Flask={flask_t:.2f}°C (diff: {diff:.4f}°C), Latency: {lat_ms:.2f} ms")
    assert diff < 0.05, f"Prediction mismatch! Expected {expected_pred}, got {flask_t}"

print(f"\nAverage /predict call latency across all calls: {sum(latencies)/len(latencies):.2f} ms")

print("\n" + "=" * 65)
print("4. VERIFYING /whatif ENDPOINT")
print("=" * 65)
t0 = time.perf_counter()
resp_wi = client.post("/whatif", data=json.dumps({"ha": 30.0, "phi": 0.03}), content_type="application/json")
lat_wi = (time.perf_counter() - t0) * 1000
wi_data = resp_wi.get_json()
print(f"POST /whatif Status: {resp_wi.status_code}")
print(f"POST /whatif Latency: {lat_wi:.2f} ms")
print(f"Points generated: {len(wi_data['curve'])}")
print(f"Curve start: u_in={wi_data['curve'][0]['u_in']}, T_max={wi_data['curve'][0]['t_max']}°C")
print(f"Curve end  : u_in={wi_data['curve'][-1]['u_in']}, T_max={wi_data['curve'][-1]['t_max']}°C")

print("\n" + "=" * 65)
print("5. VERIFYING ERROR HANDLING (BAD & OUT-OF-BOUNDS INPUTS)")
print("=" * 65)
bad_cases = [
    {"payload": {"ha": 999.0, "phi": 0.03, "u_in": 0.20}, "reason": "Ha out of range"},
    {"payload": {"ha": 30.0, "phi": 0.99, "u_in": 0.20}, "reason": "phi out of range"},
    {"payload": {"ha": "not_a_number", "phi": 0.03, "u_in": 0.20}, "reason": "Invalid number"},
    {"payload": {"ha": 30.0}, "reason": "Missing inputs"},
]
for bc in bad_cases:
    res = client.post("/predict", data=json.dumps(bc["payload"]), content_type="application/json")
    err_msg = res.get_json().get("error", "")
    print(f"Tested '{bc['reason']}': Status={res.status_code}, Handled: '{err_msg}'")
    assert res.status_code == 400

print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
