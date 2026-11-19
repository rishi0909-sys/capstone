import os
import sys
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.evaluation.phase4_validation import run_phase4_validation


def _make_df(days=450):
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    rng = np.random.default_rng(42)
    qty = [int(100 + 20 * np.sin(2 * np.pi * i / 7) + rng.integers(-8, 12)) for i in range(days)]
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "quantity": qty})


def test_phase4_validation_daily():
    df = _make_df(days=450)
    res = run_phase4_validation(df, granularity="D", horizon=14, n_origins=3, step_size=14, cal_window=28)
    assert res["origin_summary"]["total_origins"] == 3
    assert "Days 1-3" in res["horizon_summary"]
    assert "Days 8-14" in res["horizon_summary"]
    assert "expanding" in res["window_robustness"]
    assert "12_month" in res["window_robustness"]
    assert "gaussian" in res["interval_summary"]
    assert len(res["permutation_importance"]) > 0


if __name__ == "__main__":
    test_phase4_validation_daily()
    print("PHASE 4 VALIDATION TEST PASSED!")
