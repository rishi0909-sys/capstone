"""
Rolling-Origin Backtesting & Conformal Prediction Interval Evaluator
==================================================================
Implements expanding-window chronological backtesting for daily and weekly
demand forecasting, comparing XGBoost against classical baselines and
evaluating Gaussian, Global Conformal, and Horizon-Specific Conformal intervals.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from src.evaluation.metrics import calculate_forecast_metrics
from src.forecasting.xgboost_model import XGBoostForecaster


def _recursive_predict(model, feature_names, history, last_date, horizon, granularity):
    """Autoregressively predicts `horizon` steps using only `history` up to `last_date`."""
    is_daily = granularity.upper() in ["D", "DAILY"]
    offset = pd.Timedelta(days=1) if is_daily else pd.Timedelta(weeks=1)
    hist = list(history)
    curr_date = pd.to_datetime(last_date)
    step_counter = len(hist)
    preds = []

    for step in range(1, horizon + 1):
        curr_date += offset
        month = curr_date.month
        day_of_month = curr_date.day
        day_of_week = curr_date.dayofweek
        quarter = curr_date.quarter
        is_weekend = 1 if day_of_week in [5, 6] else 0
        trend_step = step_counter + step

        if is_daily:
            lag_1 = hist[-1]
            lag_2 = hist[-2] if len(hist) >= 2 else lag_1
            lag_3 = hist[-3] if len(hist) >= 3 else lag_2
            lag_7 = hist[-7] if len(hist) >= 7 else lag_1
            lag_14 = hist[-14] if len(hist) >= 14 else lag_7
            recent_7 = hist[-7:]
            row_dict = {
                "month": month, "day_of_month": day_of_month,
                "day_of_week": day_of_week, "quarter": quarter,
                "is_weekend": is_weekend, "trend_step": trend_step,
                "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
                "lag_7": lag_7, "lag_14": lag_14,
                "rolling_mean_7": float(np.mean(recent_7)),
                "rolling_std_7": float(np.std(recent_7, ddof=1)) if len(recent_7) >= 2 else 0.0,
                "rolling_mean_14": float(np.mean(hist[-14:])),
            }
        else:
            lag_1 = hist[-1]
            lag_2 = hist[-2] if len(hist) >= 2 else lag_1
            lag_3 = hist[-3] if len(hist) >= 3 else lag_2
            lag_4 = hist[-4] if len(hist) >= 4 else lag_3
            rolling_4 = float(np.mean(hist[-4:]))
            rolling_8 = float(np.mean(hist[-8:])) if len(hist) >= 8 else rolling_4
            row_dict = {
                "month": month, "day_of_week": day_of_week,
                "quarter": quarter, "trend_step": trend_step,
                "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_4": lag_4,
                "rolling_mean_4": rolling_4, "rolling_mean_8": rolling_8,
            }

        x_step = pd.DataFrame([row_dict])[feature_names]
        pred = float(max(0.0, model.predict(x_step)[0]))
        preds.append(pred)
        hist.append(pred)

    return preds


def _compute_origin_baselines(history, horizon, granularity):
    """Computes multi-step recursive/seasonal baselines using only `history`."""
    is_daily = granularity.upper() in ["D", "DAILY"]
    naive_preds = [float(history[-1])] * horizon
    roll_w = 7 if is_daily else 4
    roll_preds = [float(np.mean(history[-roll_w:]))] * horizon

    out = {
        "Naive (t-1)": naive_preds,
        f"Rolling Mean ({roll_w})": roll_preds,
    }

    if is_daily:
        out["Seasonal Naive (t-7)"] = [
            float(history[-7 + (h % 7)]) if len(history) >= 7 else float(history[-1])
            for h in range(horizon)
        ]
    else:
        out["Seasonal Naive (t-4)"] = [
            float(history[-4 + (h % 4)]) if len(history) >= 4 else float(history[-1])
            for h in range(horizon)
        ]
        if len(history) >= 52:
            out["Seasonal Naive (t-52)"] = [
                float(history[-52 + (h % 52)])
                for h in range(horizon)
            ]
    return out


def run_rolling_origin_backtest(
    df: pd.DataFrame,
    granularity: str = "D",
    horizon: int = 14,
    n_origins: int = 8,
    step_size: int = 14,
    cal_window: int = 28,
    xgb_params: dict = None,
):
    """
    Executes expanding-window rolling-origin backtesting.
    Strictly prevents future data leakage at every origin.
    """
    is_daily = granularity.upper() in ["D", "DAILY"]
    fc_helper = XGBoostForecaster(df, target_col="quantity", time_col="date")
    ts = fc_helper._prepare_continuous_time_series(granularity=granularity)
    if ts is None or len(ts) < (horizon + cal_window + 30):
        raise ValueError("Insufficient time-series length for rolling-origin backtest.")

    params = xgb_params or (
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8}
        if is_daily
        else {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9}
    )

    total_len = len(ts)
    last_origin = total_len - horizon
    origins = [last_origin - i * step_size for i in range(n_origins)]
    origins = sorted([o for o in origins if o >= (cal_window + 30)])

    all_actuals = []
    model_preds = {"XGBoost": []}
    horizon_records = {h: {"actual": [], "XGBoost": []} for h in range(1, horizon + 1)}

    # Interval tracking: A=Gaussian, B=Global Conformal, C=Horizon Conformal
    iv_records = {
        "gaussian": {"covered": [], "width": [], "by_h": {h: [] for h in range(1, horizon + 1)}},
        "global_conformal": {"covered": [], "width": [], "by_h": {h: [] for h in range(1, horizon + 1)}},
        "horizon_conformal": {"covered": [], "width": [], "by_h": {h: [] for h in range(1, horizon + 1)}},
    }
    horizon_abs_errors_pool = {h: [] for h in range(1, horizon + 1)}

    for origin_idx in origins:
        ts_avail = ts.iloc[:origin_idx].copy()
        ts_future = ts.iloc[origin_idx : origin_idx + horizon].copy()
        actuals_h = ts_future["quantity"].tolist()

        df_feat, features = fc_helper._create_leakage_safe_features(ts_avail, granularity=granularity)
        split_idx = len(df_feat) - cal_window
        train_df = df_feat.iloc[:split_idx]
        cal_df = df_feat.iloc[split_idx:]

        # 1. Fit on Train only for split-conformal calibration
        cal_model = xgb.XGBRegressor(
            **params, colsample_bytree=0.8, random_state=42, n_jobs=4
        )
        cal_model.fit(train_df[features], train_df["quantity"])
        cal_preds = np.maximum(cal_model.predict(cal_df[features]), 0.0)
        cal_res = np.array(cal_df["quantity"], dtype=np.float64) - cal_preds
        res_std = float(np.std(cal_res, ddof=1)) if len(cal_res) > 1 else 0.0
        q80_global = float(np.quantile(np.abs(cal_res), 0.80))

        # 2. Refit on Train + Cal up to origin_idx for forecasting
        prod_model = xgb.XGBRegressor(
            **params, colsample_bytree=0.8, random_state=42, n_jobs=4
        )
        prod_model.fit(df_feat[features], df_feat["quantity"])

        history = ts_avail["quantity"].tolist()
        last_date = ts_avail["date"].iloc[-1]
        xgb_h = _recursive_predict(prod_model, features, history, last_date, horizon, granularity)

        # 3. Baselines
        b_h = _compute_origin_baselines(history, horizon, granularity)
        for b_name, b_vals in b_h.items():
            model_preds.setdefault(b_name, []).extend(b_vals)

        all_actuals.extend(actuals_h)
        model_preds["XGBoost"].extend(xgb_h)

        for step_idx in range(horizon):
            h = step_idx + 1
            y_true = actuals_h[step_idx]
            y_hat = xgb_h[step_idx]
            horizon_records[h]["actual"].append(y_true)
            horizon_records[h]["XGBoost"].append(y_hat)

            # A. Gaussian 80%
            g_half = 1.28 * res_std * np.sqrt(1.0 + 0.05 * h)
            g_lo, g_hi = max(0.0, y_hat - g_half), y_hat + g_half
            g_cov = int(g_lo <= y_true <= g_hi)
            iv_records["gaussian"]["covered"].append(g_cov)
            iv_records["gaussian"]["width"].append(g_hi - g_lo)
            iv_records["gaussian"]["by_h"][h].append(g_cov)

            # B. Global Conformal 80%
            c_lo, c_hi = max(0.0, y_hat - q80_global), y_hat + q80_global
            c_cov = int(c_lo <= y_true <= c_hi)
            iv_records["global_conformal"]["covered"].append(c_cov)
            iv_records["global_conformal"]["width"].append(c_hi - c_lo)
            iv_records["global_conformal"]["by_h"][h].append(c_cov)

            # C. Horizon-Specific Conformal 80% (using prior origins' h-step errors, fallback to q80_global)
            pool = horizon_abs_errors_pool[h]
            q80_h = float(np.quantile(pool, 0.80)) if len(pool) >= 3 else q80_global
            hc_lo, hc_hi = max(0.0, y_hat - q80_h), y_hat + q80_h
            hc_cov = int(hc_lo <= y_true <= hc_hi)
            iv_records["horizon_conformal"]["covered"].append(hc_cov)
            iv_records["horizon_conformal"]["width"].append(hc_hi - hc_lo)
            iv_records["horizon_conformal"]["by_h"][h].append(hc_cov)

            # Update pool AFTER evaluating this origin (no future leakage)
            horizon_abs_errors_pool[h].append(abs(y_true - y_hat))

    # Aggregate metrics
    overall_metrics = {}
    for m_name, m_preds in model_preds.items():
        if len(m_preds) == len(all_actuals):
            overall_metrics[m_name] = calculate_forecast_metrics(all_actuals, m_preds)

    by_horizon_metrics = {
        h: calculate_forecast_metrics(horizon_records[h]["actual"], horizon_records[h]["XGBoost"])
        for h in range(1, horizon + 1)
    }

    median_demand = float(np.median(all_actuals))
    low_mask = np.array(all_actuals) <= median_demand
    high_mask = np.array(all_actuals) > median_demand

    interval_summary = {}
    for iv_name, iv_data in iv_records.items():
        cov_arr = np.array(iv_data["covered"])
        w_arr = np.array(iv_data["width"])
        interval_summary[iv_name] = {
            "coverage_pct": round(float(np.mean(cov_arr) * 100.0), 2),
            "avg_width": round(float(np.mean(w_arr)), 2),
            "low_demand_coverage_pct": round(float(np.mean(cov_arr[low_mask]) * 100.0), 2) if np.any(low_mask) else 0.0,
            "high_demand_coverage_pct": round(float(np.mean(cov_arr[high_mask]) * 100.0), 2) if np.any(high_mask) else 0.0,
            "coverage_by_horizon_pct": {
                h: round(float(np.mean(iv_data["by_h"][h]) * 100.0), 2)
                for h in range(1, horizon + 1)
            },
        }

    return {
        "granularity": granularity,
        "horizon": horizon,
        "n_origins": len(origins),
        "origin_dates": [str(ts["date"].iloc[o - 1])[:10] for o in origins],
        "date_range": f"{str(ts['date'].min())[:10]} to {str(ts['date'].max())[:10]}",
        "overall_metrics": overall_metrics,
        "by_horizon_metrics": by_horizon_metrics,
        "interval_summary": interval_summary,
    }
