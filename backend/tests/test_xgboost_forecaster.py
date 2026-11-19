import os
import sys
import tempfile
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.forecasting.xgboost_model import XGBoostForecaster


def _make_synthetic_df(days=120, zero_every_7=True):
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rng = np.random.default_rng(42)
    qty = []
    for i, d in enumerate(dates):
        if zero_every_7 and d.dayofweek == 5:
            qty.append(0)
        else:
            qty.append(int(100 + 20 * np.sin(2 * np.pi * i / 7) + rng.integers(-10, 15)))
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "quantity": qty,
    })


def test_out_of_sample_residual_calibration():
    df = _make_synthetic_df(days=120)
    fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    ok, msg = fc.train_and_evaluate(granularity="D", test_horizon=14, val_horizon=14, tune=True)
    assert ok, msg
    assert fc.residual_std > 0.0
    assert fc.residual_q80 > 0.0


def test_rolling_std_consistency_ddof1():
    df = _make_synthetic_df(days=90)
    fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    ok, _ = fc.train_and_evaluate(granularity="D", test_horizon=14, val_horizon=14, tune=False)
    assert ok
    # Verify ddof=1 matches pandas rolling(7).std()
    history = list(fc.results["train_actuals"]) + list(fc.results["test_actuals"])
    expected_std = float(pd.Series(history[-7:]).std())  # pandas default ddof=1
    actual_std = float(np.std(history[-7:], ddof=1))
    assert abs(expected_std - actual_std) < 1e-9


def test_weekly_seasonal_naive_lag_selection():
    fc = XGBoostForecaster()
    # Case 1: >= 52 weeks history -> uses t-52 and includes t-4
    train_long = [100.0 + i for i in range(60)]
    test_long = [160.0 + i for i in range(6)]
    b_long = fc.compute_baselines(train_long, test_long, granularity="W")
    assert "Seasonal Naive (t-52)" in b_long
    assert "Seasonal Naive (t-4)" in b_long

    # Case 2: < 52 weeks history -> falls back to t-4
    train_short = [100.0 + i for i in range(30)]
    test_short = [130.0 + i for i in range(6)]
    b_short = fc.compute_baselines(train_short, test_short, granularity="W")
    assert "Seasonal Naive (t-4)" in b_short
    assert "Seasonal Naive (t-52)" not in b_short


def test_zero_sales_periods_handled():
    df = _make_synthetic_df(days=90, zero_every_7=True)
    fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    ok, _ = fc.train_and_evaluate(granularity="D", test_horizon=14, val_horizon=14, tune=False)
    assert ok
    future = fc.forecast_future(horizon=7, freq="D")
    assert len(future["future_preds"]) == 7
    assert all(p >= 0.0 for p in future["future_preds"])


def test_short_historical_series():
    df = _make_synthetic_df(days=15)
    fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    ok, msg = fc.train_and_evaluate(granularity="D", test_horizon=14, val_horizon=14, tune=False)
    assert not ok
    assert "Insufficient" in msg


def test_daily_and_weekly_forecast_and_reload_consistency():
    df = _make_synthetic_df(days=180)
    with tempfile.TemporaryDirectory() as tmpdir:
        for gran, horiz in [("D", 14), ("W", 4)]:
            fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
            ok, msg = fc.train_and_evaluate(granularity=gran, test_horizon=14, val_horizon=14, tune=True)
            assert ok, msg
            f1 = fc.forecast_future(horizon=horiz, freq=gran)

            model_path = os.path.join(tmpdir, f"model_{gran}.json")
            fc.save_model_artifact(model_path)

            fc2 = XGBoostForecaster()
            fc2.load_model_artifact(model_path)
            f2 = fc2.forecast_future(horizon=horiz, freq=gran)

            max_diff = float(np.max(np.abs(np.array(f1["future_preds"]) - np.array(f2["future_preds"]))))
            assert max_diff < 1e-4
            assert abs(fc.residual_q80 - fc2.residual_q80) < 1e-6


if __name__ == "__main__":
    test_out_of_sample_residual_calibration()
    test_rolling_std_consistency_ddof1()
    test_weekly_seasonal_naive_lag_selection()
    test_zero_sales_periods_handled()
    test_short_historical_series()
    test_daily_and_weekly_forecast_and_reload_consistency()
    print("ALL 6 FORECASTER TESTS PASSED!")
