import os
import sys
import tempfile
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.forecasting.hist_gradient_boosting_model import HistGradientBoostingForecaster
import main as backend_main


def _make_df(days=180, zero_every_7=True):
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    rng = np.random.default_rng(42)
    qty = []
    for i, d in enumerate(dates):
        if zero_every_7 and d.dayofweek == 5:
            qty.append(0)
        else:
            qty.append(int(100 + 20 * np.sin(2 * np.pi * i / 7) + rng.integers(-8, 12)))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "quantity": qty})


def test_hgb_save_load_daily_and_weekly():
    df = _make_df(days=180)
    with tempfile.TemporaryDirectory() as tmpdir:
        for gran, horiz in [("D", 14), ("W", 4)]:
            fc = HistGradientBoostingForecaster(df, target_col="quantity", time_col="date")
            ok, msg = fc.train_and_evaluate(granularity=gran, test_horizon=14, val_horizon=14, tune=True, dataset_id="synthetic")
            assert ok, msg
            f1 = fc.forecast_future(horizon=horiz, freq=gran)
            assert len(f1["future_preds"]) == horiz
            assert all(p >= 0.0 for p in f1["future_preds"])

            mpath = os.path.join(tmpdir, f"hgb_{gran}.joblib")
            fc.save_model_artifact(mpath)

            fc2 = HistGradientBoostingForecaster()
            fc2.load_model_artifact(mpath)
            f2 = fc2.forecast_future(horizon=horiz, freq=gran)
            assert float(np.max(np.abs(np.array(f1["future_preds"]) - np.array(f2["future_preds"])))) < 1e-6
            assert fc.feature_names == fc2.feature_names


def test_hgb_invalid_zero_and_corrupted_handling():
    # Short/invalid series
    df_short = _make_df(days=10)
    fc_short = HistGradientBoostingForecaster(df_short)
    ok, _ = fc_short.train_and_evaluate(granularity="D")
    assert not ok

    # Zero-sales history
    df_zero = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=90, freq="D").strftime("%Y-%m-%d %H:%M:%S"),
        "quantity": [0] * 90,
    })
    fc_zero = HistGradientBoostingForecaster(df_zero)
    ok_z, _ = fc_zero.train_and_evaluate(granularity="D", test_horizon=14, val_horizon=14, tune=False)
    assert ok_z
    fz = fc_zero.forecast_future(horizon=7, freq="D")
    assert all(p == 0.0 for p in fz["future_preds"])
    assert fz["interval_methodology"] == "uncalibrated"

    # Corrupted artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = os.path.join(tmpdir, "corrupt.joblib")
        with open(bad_path, "w") as f:
            f.write("not a joblib")
        fc_bad = HistGradientBoostingForecaster()
        try:
            fc_bad.load_model_artifact(bad_path)
            assert False, "Should have raised on missing metadata/corrupt file"
        except Exception:
            pass


def test_fastapi_endpoint_and_caching():
    backend_main._FORECASTER_CACHE.clear()
    client = TestClient(backend_main.app)

    resp_d = client.post("/api/forecast/run", json={"dataset_id": "demo", "granularity": "D", "horizon": 14, "target_product": "Overall"})
    assert resp_d.status_code == 200
    data_d = resp_d.json()
    assert len(data_d["future_preds"]) == 14
    assert "HistGradientBoosting" in data_d["model_name"]
    assert "hgb:D" in backend_main._FORECASTER_CACHE

    resp_w = client.post("/api/forecast/run", json={"dataset_id": "demo", "granularity": "W", "horizon": 4, "target_product": "Overall"})
    assert resp_w.status_code == 200
    data_w = resp_w.json()
    assert len(data_w["future_preds"]) == 4
    assert "HistGradientBoosting" in data_w["model_name"]
    assert "hgb:W" in backend_main._FORECASTER_CACHE


if __name__ == "__main__":
    test_hgb_save_load_daily_and_weekly()
    test_hgb_invalid_zero_and_corrupted_handling()
    test_fastapi_endpoint_and_caching()
    print("ALL PHASE 5 PRODUCTION TESTS PASSED!")
