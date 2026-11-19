"""
Master Task Verification Suite
Tests all 10 phases of the Intelligent Pattern Discovery Engine audit and implementation:
1. Hardware diagnostics reports
2. Dataset audit and clean dataset integrity
3. Leakage-safe time-series feature engineering and chronological splits
4. Forecast metrics (MAE, RMSE, R2, WAPE, sMAPE)
5. Model artifact serialization and zero-drift reload
6. Prediction intervals and recursive autoregressive forecasting
7. Authentic FP-Growth rules and basket validation
8. Authentic customer behavioral analytics (no fabricated demographics)
9. FastAPI Intelligence Explorer endpoints
10. Robust error handling on missing/invalid requests
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from main import app
from src.evaluation.metrics import calculate_forecast_metrics
from src.forecasting.xgboost_model import XGBoostForecaster

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
REPORTS_DIR = os.path.join(BACKEND_DIR, "reports")
DATA_DIR = os.path.join(BACKEND_DIR, "data")


# -------------------------------------------------------------
# Test Phase 1: Hardware Diagnostics Reports
# -------------------------------------------------------------
def test_system_diagnostics_reports():
    json_path = os.path.join(REPORTS_DIR, "system_compatibility.json")
    md_path = os.path.join(REPORTS_DIR, "system_compatibility.md")

    assert os.path.exists(json_path), "system_compatibility.json must exist"
    assert os.path.exists(md_path), "system_compatibility.md must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cpu" in data
    assert "operating_system" in data
    assert "workload_verdicts" in data
    assert data["cpu"]["architecture"] == "arm64"
    assert data["workload_verdicts"]["2_daily_demand_forecasting_xgboost"]["classification"] == "COMPATIBLE"


# -------------------------------------------------------------
# Test Phase 2: Dataset Audit & Clean Retail Integrity
# -------------------------------------------------------------
def test_clean_retail_dataset():
    csv_path = os.path.join(DATA_DIR, "online_retail_clean.csv")
    assert os.path.exists(csv_path), "online_retail_clean.csv must exist"

    df = pd.read_csv(csv_path, nrows=1000)
    required_cols = {"transaction_id", "product_name", "quantity", "price", "date", "customer_id", "category"}
    assert required_cols.issubset(set(df.columns)), f"Missing required columns: {required_cols - set(df.columns)}"

    # Ensure no negative quantities or prices in clean dataset
    assert (df["quantity"] > 0).all(), "Clean dataset must have strictly positive quantities"
    assert (df["price"] > 0).all(), "Clean dataset must have strictly positive unit prices"

    # Ensure no fabricated demographic fields like age or gender
    assert "age" not in df.columns, "No fabricated demographic age column allowed"
    assert "gender" not in df.columns, "No fabricated demographic gender column allowed"


# -------------------------------------------------------------
# Test Phase 3 & 4: Leakage Safety & Chronological Splits
# -------------------------------------------------------------
def test_leakage_safety_feature_engineering():
    # Synthetic series
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "quantity": np.arange(100, dtype=float) + 10.0,
        "product_name": "Product A"
    })

    forecaster = XGBoostForecaster(df, target_col="quantity", time_col="date")
    features_df, _ = forecaster._create_leakage_safe_features(df, granularity="D")

    # Lag 1 at index i must equal target at index i-1, NEVER at index i
    for i in range(15, 30):
        actual_prev_target = features_df.iloc[i - 1]["quantity"]
        lag_1_val = features_df.iloc[i]["lag_1"]
        assert lag_1_val == actual_prev_target, f"Leakage detected! lag_1={lag_1_val} vs prev={actual_prev_target}"

    # Rolling mean 7 at index i must be calculated strictly from shifted history
    for i in range(15, 30):
        history_window = features_df.iloc[i - 7 : i]["quantity"]
        expected_roll = history_window.mean()
        actual_roll = features_df.iloc[i]["rolling_mean_7"]
        assert np.isclose(actual_roll, expected_roll), f"Rolling leakage! {actual_roll} vs {expected_roll}"


# -------------------------------------------------------------
# Test Phase 5: Evaluation Metrics
# -------------------------------------------------------------
def test_scale_aware_forecast_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 0.0])
    y_pred = np.array([110.0, 190.0, 310.0, 10.0])

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "WAPE" in metrics
    assert "sMAPE" in metrics
    assert "R2" in metrics

    # WAPE = sum(|y_true - y_pred|) / sum(y_true) = (10 + 10 + 10 + 10) / 600 = 40 / 600 = 6.67%
    assert np.isclose(metrics["WAPE"], (40.0 / 600.0) * 100, atol=0.1)


# -------------------------------------------------------------
# Test Phase 6: Association Rules & Customer Analytics
# -------------------------------------------------------------
def test_precomputed_association_rules_and_behavior():
    rules_path = os.path.join(MODELS_DIR, "association_rules.json")
    behavior_path = os.path.join(MODELS_DIR, "customer_behavior.json")

    assert os.path.exists(rules_path), "association_rules.json must exist"
    assert os.path.exists(behavior_path), "customer_behavior.json must exist"

    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    assert len(rules) > 100, f"Expected >100 verified association rules, found {len(rules)}"
    first_rule = rules[0]
    assert "antecedents" in first_rule
    assert "consequents" in first_rule
    assert "lift" in first_rule
    assert first_rule["lift"] >= 1.0, "Discovered rules must have lift >= 1.0"

    with open(behavior_path, "r", encoding="utf-8") as f:
        behavior = json.load(f)

    assert "summary" in behavior
    assert "preferences" in behavior
    summary = behavior["summary"]
    assert summary["segmentation_type"] == "Behavioral & Frequency Tiers"
    assert summary["total_unique_customers"] > 0


# -------------------------------------------------------------
# Test Phase 7: Model Artifacts Reload & Zero-Drift Prediction
# -------------------------------------------------------------
def test_model_artifact_reload_and_zero_drift():
    daily_model_path = os.path.join(MODELS_DIR, "xgboost_overall_daily.json")
    assert os.path.exists(daily_model_path), "xgboost_overall_daily.json must exist"

    fc1 = XGBoostForecaster()
    fc1.load_model_artifact(daily_model_path)

    fc2 = XGBoostForecaster()
    fc2.load_model_artifact(daily_model_path)

    pred1 = fc1.forecast_future(horizon=14, freq="D")
    pred2 = fc2.forecast_future(horizon=14, freq="D")

    assert pred1["future_dates"] == pred2["future_dates"]
    assert np.allclose(pred1["future_preds"], pred2["future_preds"]), "Reloaded model generated divergent predictions!"
    assert len(pred1["future_preds"]) == 14
    assert len(pred1["future_lower_80"]) == 14
    assert len(pred1["future_upper_80"]) == 14


# -------------------------------------------------------------
# Test Phase 8: FastAPI Intelligence Explorer Integration
# -------------------------------------------------------------
def test_fastapi_endpoints():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"
    assert res_health.json()["artifacts_available"]["daily_xgboost"] is True

    # 2. Daily forecast endpoint
    res_f_daily = client.post(
        "/api/forecast/run",
        json={"dataset_id": "demo_retail_data", "target_product": "Overall", "horizon": 14, "granularity": "D"},
    )
    assert res_f_daily.status_code == 200
    data_daily = res_f_daily.json()
    assert len(data_daily["future_dates"]) == 14
    assert len(data_daily["future_preds"]) == 14
    assert data_daily["mae"] > 0
    assert data_daily["r2"] > 0

    # 3. Weekly forecast endpoint
    res_f_weekly = client.post(
        "/api/forecast/run",
        json={"dataset_id": "demo_retail_data", "target_product": "Overall", "horizon": 4, "granularity": "W"},
    )
    assert res_f_weekly.status_code == 200
    data_weekly = res_f_weekly.json()
    assert len(data_weekly["future_dates"]) == 4

    # 4. Association rules mining endpoint
    res_mining = client.post(
        "/api/mining/run",
        json={"dataset_id": "demo_retail_data", "min_support": 0.01, "min_confidence": 0.3},
    )
    assert res_mining.status_code == 200
    data_mining = res_mining.json()
    assert data_mining["rule_count"] > 0

    # 5. Customer behavioral analytics endpoint
    res_demo = client.post("/api/demographics/analyze", json={"dataset_id": "demo_retail_data"})
    assert res_demo.status_code == 200
    assert res_demo.json()["status"] == "success"

    # 6. Automated insights endpoint
    res_insights = client.post("/api/insights/generate", json={"dataset_id": "demo_retail_data"})
    assert res_insights.status_code == 200
    assert len(res_insights.json()["insights"]) > 0


# -------------------------------------------------------------
# Test Phase 8B: Missing Session Error Handling
# -------------------------------------------------------------
def test_missing_session_error_handling():
    client = TestClient(app)
    # Requesting non-existent product on unknown session must return 404
    res = client.post(
        "/api/forecast/run",
        json={"dataset_id": "unknown_session_xyz", "target_product": "NonExistentProduct123", "horizon": 7, "granularity": "D"},
    )
    assert res.status_code == 404


if __name__ == "__main__":
    tests = [
        ("Phase 1: System Diagnostics Reports", test_system_diagnostics_reports),
        ("Phase 2: Clean Retail Dataset Integrity", test_clean_retail_dataset),
        ("Phase 3 & 4: Leakage Safety & Chronological Features", test_leakage_safety_feature_engineering),
        ("Phase 5: Scale-Aware Forecast Metrics (WAPE/sMAPE)", test_scale_aware_forecast_metrics),
        ("Phase 6: Association Rules & Authentic Behavior", test_precomputed_association_rules_and_behavior),
        ("Phase 7: Model Reload & Zero-Drift Inference", test_model_artifact_reload_and_zero_drift),
        ("Phase 8: FastAPI Intelligence Explorer Endpoints", test_fastapi_endpoints),
        ("Phase 8B: Missing Session Error Handling", test_missing_session_error_handling),
    ]

    print("\n" + "=" * 65, flush=True)
    print("MASTER TASK AUTOMATED VERIFICATION SUITE", flush=True)
    print("=" * 65, flush=True)
    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f" [PASS] {name}", flush=True)
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("=" * 65, flush=True)
    print(f"RESULTS: {passed}/{len(tests)} tests passed successfully.", flush=True)
    print("=" * 65 + "\n", flush=True)
