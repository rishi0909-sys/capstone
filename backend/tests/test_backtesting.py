import os
import sys
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.evaluation.backtesting import run_rolling_origin_backtest


def _make_df(days=200):
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    rng = np.random.default_rng(42)
    qty = [int(100 + 20 * np.sin(2 * np.pi * i / 7) + rng.integers(-8, 12)) for i in range(days)]
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "quantity": qty})


def test_rolling_origin_daily_backtest():
    df = _make_df(days=200)
    res = run_rolling_origin_backtest(df, granularity="D", horizon=14, n_origins=4, step_size=14, cal_window=28)
    assert res["n_origins"] == 4
    assert res["origin_dates"] == sorted(res["origin_dates"])
    assert "XGBoost" in res["overall_metrics"]
    assert "Seasonal Naive (t-7)" in res["overall_metrics"]
    assert len(res["by_horizon_metrics"]) == 14
    for iv in ["gaussian", "global_conformal", "horizon_conformal"]:
        assert 0.0 <= res["interval_summary"][iv]["coverage_pct"] <= 100.0
        assert res["interval_summary"][iv]["avg_width"] >= 0.0


def test_rolling_origin_weekly_backtest_with_annual_lag():
    df = _make_df(days=500)  # ~71 weeks -> >= 52 weeks history
    res = run_rolling_origin_backtest(df, granularity="W", horizon=4, n_origins=3, step_size=4, cal_window=8)
    assert res["n_origins"] == 3
    assert "Seasonal Naive (t-52)" in res["overall_metrics"]
    assert "Seasonal Naive (t-4)" in res["overall_metrics"]
    assert len(res["by_horizon_metrics"]) == 4


if __name__ == "__main__":
    test_rolling_origin_daily_backtest()
    test_rolling_origin_weekly_backtest_with_annual_lag()
    print("ALL BACKTESTING TESTS PASSED!")
