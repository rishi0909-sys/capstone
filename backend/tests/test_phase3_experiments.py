import os
import sys
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.evaluation.phase3_experiments import run_phase3_suite, build_features


def _make_df(days=200):
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    rng = np.random.default_rng(42)
    qty = [int(100 + 20 * np.sin(2 * np.pi * i / 7) + rng.integers(-8, 12)) for i in range(days)]
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "quantity": qty})


def test_phase3_suite_daily():
    df = _make_df(days=200)
    res = run_phase3_suite(df, granularity="D", horizon=14, n_origins=3, step_size=14, cal_window=28)
    assert "XGBoost" in res["overall_models"]
    assert "XGBoost_Enhanced" in res["overall_models"]
    assert "HistGradientBoosting" in res["overall_models"]
    assert "RandomForest" in res["overall_models"]
    assert "ExtraTrees" in res["overall_models"]
    assert "Ridge" in res["overall_models"]
    assert "normalized_conformal" in res["intervals"]
    assert "regime_conformal" in res["intervals"]
    assert len(res["permutation_importance"]) > 0


if __name__ == "__main__":
    test_phase3_suite_daily()
    print("PHASE 3 SUITE TEST PASSED!")
