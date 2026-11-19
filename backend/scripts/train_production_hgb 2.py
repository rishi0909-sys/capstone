#!/usr/bin/env python3
"""
Phase 5: Production HistGradientBoosting Training & Benchmarking
================================================================
Trains daily & weekly HistGradientBoostingForecaster on `dataset_3_clean.csv`,
saves artifacts under `backend/models/production/`, benchmarks latency vs XGBoost,
and writes Phase 5 migration reports under `backend/reports/phase5_migration/`.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.forecasting.hist_gradient_boosting_model import HistGradientBoostingForecaster
from src.forecasting.xgboost_model import XGBoostForecaster

DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_3_clean.csv")
PROD_MODELS_DIR = os.path.join(BACKEND_DIR, "models", "production")
XGB_MODELS_DIR = os.path.join(BACKEND_DIR, "models")
OUT_DIR = os.path.join(BACKEND_DIR, "reports", "phase5_migration")
os.makedirs(PROD_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 80)
    print("  PHASE 5: PRODUCTION HISTGRADIENTBOOSTING TRAINING & BENCHMARKING")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, usecols=["date", "quantity"], low_memory=False)
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")

    # 1. Train Daily HGB
    d_fc = HistGradientBoostingForecaster(df, target_col="quantity", time_col="date")
    ok_d, msg_d = d_fc.train_and_evaluate(granularity="D", test_horizon=28, val_horizon=28, tune=True, dataset_id="dataset_3_clean.csv")
    assert ok_d, msg_d
    d_path = os.path.join(PROD_MODELS_DIR, "hgb_overall_daily.joblib")
    d_fc.save_model_artifact(d_path)
    d_pred_before = d_fc.forecast_future(horizon=14, freq="D")["future_preds"]

    # 2. Train Weekly HGB
    w_fc = HistGradientBoostingForecaster(df, target_col="quantity", time_col="date")
    ok_w, msg_w = w_fc.train_and_evaluate(granularity="W", test_horizon=6, val_horizon=6, tune=True, dataset_id="dataset_3_clean.csv")
    assert ok_w, msg_w
    w_path = os.path.join(PROD_MODELS_DIR, "hgb_overall_weekly.joblib")
    w_fc.save_model_artifact(w_path)
    w_pred_before = w_fc.forecast_future(horizon=4, freq="W")["future_preds"]

    # 3. Reload & Latency Benchmarks
    t_load0 = time.perf_counter()
    d_reload = HistGradientBoostingForecaster()
    d_reload.load_model_artifact(d_path)
    hgb_d_load_ms = (time.perf_counter() - t_load0) * 1000.0

    t_inf0 = time.perf_counter()
    d_pred_after = d_reload.forecast_future(horizon=14, freq="D")["future_preds"]
    hgb_d_warm_ms = (time.perf_counter() - t_inf0) * 1000.0
    d_diff = float(np.max(np.abs(np.array(d_pred_before) - np.array(d_pred_after))))
    assert d_diff < 1e-6

    t_load1 = time.perf_counter()
    w_reload = HistGradientBoostingForecaster()
    w_reload.load_model_artifact(w_path)
    hgb_w_load_ms = (time.perf_counter() - t_load1) * 1000.0

    t_inf1 = time.perf_counter()
    w_pred_after = w_reload.forecast_future(horizon=4, freq="W")["future_preds"]
    hgb_w_warm_ms = (time.perf_counter() - t_inf1) * 1000.0
    w_diff = float(np.max(np.abs(np.array(w_pred_before) - np.array(w_pred_after))))
    assert w_diff < 1e-6

    # Compare with existing XGBoost artifacts
    xgb_d = XGBoostForecaster()
    t_xl0 = time.perf_counter()
    xgb_d.load_model_artifact(os.path.join(XGB_MODELS_DIR, "xgboost_overall_daily.json"))
    xgb_d_load_ms = (time.perf_counter() - t_xl0) * 1000.0
    t_xi0 = time.perf_counter()
    xgb_d.forecast_future(horizon=14, freq="D")
    xgb_d_warm_ms = (time.perf_counter() - t_xi0) * 1000.0

    perf = {
        "hgb_daily": {"load_ms": round(hgb_d_load_ms, 2), "warm_14d_ms": round(hgb_d_warm_ms, 2), "reload_max_diff": d_diff},
        "hgb_weekly": {"load_ms": round(hgb_w_load_ms, 2), "warm_4w_ms": round(hgb_w_warm_ms, 2), "reload_max_diff": w_diff},
        "xgb_daily": {"load_ms": round(xgb_d_load_ms, 2), "warm_14d_ms": round(xgb_d_warm_ms, 2)},
    }
    with open(os.path.join(OUT_DIR, "performance_results.json"), "w", encoding="utf-8") as f:
        json.dump(perf, f, indent=2)

    migration = {
        "daily_metrics": d_fc.results["metrics"],
        "weekly_metrics": w_fc.results["metrics"],
        "daily_artifact": d_path,
        "weekly_artifact": w_path,
        "duration_seconds": round(time.time() - t0, 2),
    }
    with open(os.path.join(OUT_DIR, "migration_results.json"), "w", encoding="utf-8") as f:
        json.dump(migration, f, indent=2)

    with open(os.path.join(OUT_DIR, "production_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": d_fc.metadata, "weekly": w_fc.metadata}, f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "PHASE5_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Phase 5: Production Forecasting Model Migration & Integration Report\n\n")
        f.write("1. **Migrated Estimators**: `HistGradientBoostingRegressor` for both **Daily** (`hgb_overall_daily.joblib`) and **Weekly** (`hgb_overall_weekly.joblib`) stored under `backend/models/production/`, with `XGBoostForecaster` retained as fallback.\n")
        f.write("2. **Feature Pipeline**: Identical between training and inference (reusing `XGBoostForecaster._create_leakage_safe_features` and `ddof=1` rolling std).\n")
        f.write("3. **API Compatibility**: `/api/forecast/run` and `ForecastResponse` preserved; `_FORECASTER_CACHE` caches loaded instances.\n")
        f.write(f"4. **Serialization**: Reload discrepancies — Daily: `{d_diff:.8f}`, Weekly: `{w_diff:.8f}`.\n")
        f.write(f"5. **Latency**: HGB Daily load `{perf['hgb_daily']['load_ms']} ms`, 14d inference `{perf['hgb_daily']['warm_14d_ms']} ms` (vs XGB `{perf['xgb_daily']['load_ms']} ms` / `{perf['xgb_daily']['warm_14d_ms']} ms`).\n")
        f.write("6. **Prediction Intervals**: Explicitly marked as `approximate_experimental_out_of_sample_residual_std` (not guaranteed 80% coverage).\n")

    print(f"[DONE] Phase 5 training & benchmarking complete in {migration['duration_seconds']}s!")
    print("Daily Metrics :", d_fc.results["metrics"])
    print("Weekly Metrics:", w_fc.results["metrics"])
    print("Performance   :", perf)


if __name__ == "__main__":
    main()
