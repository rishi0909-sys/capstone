"""
Phase 3: Forecasting Model Optimization, Robustness & Demand-Dependent Intervals
==============================================================================
Experimental framework for:
1. Per-origin robustness & win-rate analysis
2. Leakage-safe retail calendar & momentum feature engineering + permutation importance
3. Multi-model comparison (XGBoost, HistGradientBoosting, RandomForest, ExtraTrees, Ridge)
4. Demand-dependent / normalized conformal prediction intervals
"""

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

from src.evaluation.metrics import calculate_forecast_metrics
from src.forecasting.xgboost_model import XGBoostForecaster
from src.evaluation.backtesting import _compute_origin_baselines


UK_HOLIDAY_MONTH_DAYS = {
    (1, 1), (12, 25), (12, 26), (11, 5), (11, 25), (11, 26), (5, 1), (8, 30)
}


def build_features(ts_df: pd.DataFrame, granularity: str = "D", enhanced: bool = False):
    """Builds strictly leakage-safe baseline or enhanced retail features."""
    fc = XGBoostForecaster(target_col="quantity", time_col="date")
    df_feat, base_features = fc._create_leakage_safe_features(ts_df, granularity=granularity)
    if not enhanced:
        return df_feat, base_features

    dt = df_feat["date"]
    df_feat["is_month_end"] = dt.dt.is_month_end.astype(int)
    df_feat["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    df_feat["is_q4_peak"] = dt.dt.month.isin([11, 12]).astype(int)
    df_feat["is_uk_holiday"] = [
        1 if (m, d) in UK_HOLIDAY_MONTH_DAYS else 0 for m, d in zip(dt.dt.month, dt.dt.day)
    ]

    is_daily = granularity.upper() in ["D", "DAILY"]
    if is_daily:
        df_feat["momentum_7_14"] = df_feat["rolling_mean_7"] - df_feat["rolling_mean_14"]
        extra = ["is_month_end", "is_quarter_end", "is_q4_peak", "is_uk_holiday", "momentum_7_14"]
    else:
        df_feat["momentum_4_8"] = df_feat["rolling_mean_4"] - df_feat["rolling_mean_8"]
        extra = ["is_month_end", "is_quarter_end", "is_q4_peak", "momentum_4_8"]

    return df_feat, base_features + extra


def _recursive_predict_generic(model, feature_names, history, last_date, horizon, granularity, enhanced=False):
    is_daily = granularity.upper() in ["D", "DAILY"]
    offset = pd.Timedelta(days=1) if is_daily else pd.Timedelta(weeks=1)
    hist = list(history)
    curr_date = pd.to_datetime(last_date)
    step_counter = len(hist)
    preds = []

    for step in range(1, horizon + 1):
        curr_date += offset
        row = {
            "month": curr_date.month,
            "day_of_month": curr_date.day,
            "day_of_week": curr_date.dayofweek,
            "quarter": curr_date.quarter,
            "is_weekend": 1 if curr_date.dayofweek in [5, 6] else 0,
            "trend_step": step_counter + step,
        }
        if is_daily:
            recent_7 = hist[-7:]
            r7 = float(np.mean(recent_7))
            r14 = float(np.mean(hist[-14:]))
            row.update({
                "lag_1": hist[-1],
                "lag_2": hist[-2] if len(hist) >= 2 else hist[-1],
                "lag_3": hist[-3] if len(hist) >= 3 else hist[-1],
                "lag_7": hist[-7] if len(hist) >= 7 else hist[-1],
                "lag_14": hist[-14] if len(hist) >= 14 else hist[-1],
                "rolling_mean_7": r7,
                "rolling_std_7": float(np.std(recent_7, ddof=1)) if len(recent_7) >= 2 else 0.0,
                "rolling_mean_14": r14,
            })
            if enhanced:
                row["momentum_7_14"] = r7 - r14
        else:
            r4 = float(np.mean(hist[-4:]))
            r8 = float(np.mean(hist[-8:])) if len(hist) >= 8 else r4
            row.update({
                "lag_1": hist[-1],
                "lag_2": hist[-2] if len(hist) >= 2 else hist[-1],
                "lag_3": hist[-3] if len(hist) >= 3 else hist[-1],
                "lag_4": hist[-4] if len(hist) >= 4 else hist[-1],
                "rolling_mean_4": r4,
                "rolling_mean_8": r8,
            })
            if enhanced:
                row["momentum_4_8"] = r4 - r8

        if enhanced:
            row["is_month_end"] = int(curr_date.is_month_end)
            row["is_quarter_end"] = int(curr_date.is_quarter_end)
            row["is_q4_peak"] = 1 if curr_date.month in [11, 12] else 0
            row["is_uk_holiday"] = 1 if (curr_date.month, curr_date.day) in UK_HOLIDAY_MONTH_DAYS else 0

        x_step = pd.DataFrame([row])[feature_names]
        pred = float(max(0.0, model.predict(x_step)[0]))
        preds.append(pred)
        hist.append(pred)
    return preds


def _get_candidate_models(is_daily: bool):
    xgb_p = (
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8}
        if is_daily
        else {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9}
    )
    return {
        "XGBoost": lambda: xgb.XGBRegressor(**xgb_p, colsample_bytree=0.8, random_state=42, n_jobs=4),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=4),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=4),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def run_phase3_suite(
    df: pd.DataFrame,
    granularity: str = "D",
    horizon: int = 14,
    n_origins: int = 10,
    step_size: int = 14,
    cal_window: int = 28,
):
    is_daily = granularity.upper() in ["D", "DAILY"]
    fc_helper = XGBoostForecaster(df, target_col="quantity", time_col="date")
    ts = fc_helper._prepare_continuous_time_series(granularity=granularity)

    total_len = len(ts)
    last_origin = total_len - horizon
    origins = sorted([last_origin - i * step_size for i in range(n_origins) if (last_origin - i * step_size) >= (cal_window + 30)])

    candidates = _get_candidate_models(is_daily)
    all_actuals = []
    model_preds = {m: [] for m in candidates}
    model_preds["XGBoost_Enhanced"] = []
    baseline_preds = {}
    per_origin_wape = {m: [] for m in list(candidates.keys()) + ["XGBoost_Enhanced"]}

    # Demand-dependent interval tracking (for XGBoost)
    iv_data = {
        "gaussian": {"cov": [], "w": []},
        "normalized_conformal": {"cov": [], "w": []},
        "regime_conformal": {"cov": [], "w": []},
    }
    perm_importances = {}

    for idx_o, origin_idx in enumerate(origins):
        ts_avail = ts.iloc[:origin_idx].copy()
        actuals_h = ts.iloc[origin_idx : origin_idx + horizon]["quantity"].tolist()
        all_actuals.extend(actuals_h)
        history = ts_avail["quantity"].tolist()
        last_date = ts_avail["date"].iloc[-1]

        # Baselines
        b_h = _compute_origin_baselines(history, horizon, granularity)
        for b_name, b_vals in b_h.items():
            baseline_preds.setdefault(b_name, []).extend(b_vals)

        # Base features
        df_base, base_feats = build_features(ts_avail, granularity=granularity, enhanced=False)
        split_idx = len(df_base) - cal_window
        train_b, cal_b = df_base.iloc[:split_idx], df_base.iloc[split_idx:]

        for m_name, m_fn in candidates.items():
            t0 = time.time()
            model = m_fn()
            model.fit(df_base[base_feats], df_base["quantity"])
            preds_h = _recursive_predict_generic(model, base_feats, history, last_date, horizon, granularity, enhanced=False)
            model_preds[m_name].extend(preds_h)
            per_origin_wape[m_name].append(calculate_forecast_metrics(actuals_h, preds_h)["WAPE"])

            # Demand-dependent intervals for XGBoost
            if m_name == "XGBoost":
                cal_m = m_fn()
                cal_m.fit(train_b[base_feats], train_b["quantity"])
                cal_p = np.maximum(cal_m.predict(cal_b[base_feats]), 0.0)
                cal_res = np.array(cal_b["quantity"], dtype=np.float64) - cal_p
                res_std = float(np.std(cal_res, ddof=1)) if len(cal_res) > 1 else 0.0

                # Normalized conformal: scale by rolling_std_7 (or rolling_mean_4 * 0.2 for weekly)
                scale_col = "rolling_std_7" if is_daily else "rolling_mean_4"
                cal_scales = np.maximum(cal_b[scale_col].values, 1.0)
                norm_q80 = float(np.quantile(np.abs(cal_res) / cal_scales, 0.80))

                # Regime-specific conformal
                cal_med = float(np.median(cal_b["quantity"]))
                low_r = np.abs(cal_res[cal_b["quantity"].values <= cal_med])
                high_r = np.abs(cal_res[cal_b["quantity"].values > cal_med])
                q80_low = float(np.quantile(low_r, 0.80)) if len(low_r) >= 3 else float(np.quantile(np.abs(cal_res), 0.80))
                q80_high = float(np.quantile(high_r, 0.80)) if len(high_r) >= 3 else float(np.quantile(np.abs(cal_res), 0.80))

                recent_scale = float(np.std(history[-7:], ddof=1)) if is_daily else float(np.mean(history[-4:]))
                recent_scale = max(recent_scale, 1.0)

                for h_i, (yt, yh) in enumerate(zip(actuals_h, preds_h), 1):
                    g_half = 1.28 * res_std * np.sqrt(1.0 + 0.05 * h_i)
                    iv_data["gaussian"]["cov"].append(int(max(0.0, yh - g_half) <= yt <= yh + g_half))
                    iv_data["gaussian"]["w"].append((yh + g_half) - max(0.0, yh - g_half))

                    n_half = norm_q80 * recent_scale
                    iv_data["normalized_conformal"]["cov"].append(int(max(0.0, yh - n_half) <= yt <= yh + n_half))
                    iv_data["normalized_conformal"]["w"].append((yh + n_half) - max(0.0, yh - n_half))

                    r_half = q80_low if yh <= cal_med else q80_high
                    iv_data["regime_conformal"]["cov"].append(int(max(0.0, yh - r_half) <= yt <= yh + r_half))
                    iv_data["regime_conformal"]["w"].append((yh + r_half) - max(0.0, yh - r_half))

        # Enhanced XGBoost
        df_enh, enh_feats = build_features(ts_avail, granularity=granularity, enhanced=True)
        enh_model = candidates["XGBoost"]()
        enh_model.fit(df_enh[enh_feats], df_enh["quantity"])
        enh_preds_h = _recursive_predict_generic(enh_model, enh_feats, history, last_date, horizon, granularity, enhanced=True)
        model_preds["XGBoost_Enhanced"].extend(enh_preds_h)
        per_origin_wape["XGBoost_Enhanced"].append(calculate_forecast_metrics(actuals_h, enh_preds_h)["WAPE"])

        # Permutation importance on last origin's calibration window
        if idx_o == len(origins) - 1:
            split_e = len(df_enh) - cal_window
            cal_e = df_enh.iloc[split_e:]
            perm = permutation_importance(enh_model, cal_e[enh_feats], cal_e["quantity"], n_repeats=5, random_state=42)
            perm_importances = {
                f: round(float(imp), 4)
                for f, imp in sorted(zip(enh_feats, perm.importances_mean), key=lambda x: x[1], reverse=True)
            }

    overall = {m: calculate_forecast_metrics(all_actuals, p) for m, p in model_preds.items()}
    baselines = {b: calculate_forecast_metrics(all_actuals, p) for b, p in baseline_preds.items()}

    med = float(np.median(all_actuals))
    low_m = np.array(all_actuals) <= med
    high_m = np.array(all_actuals) > med
    intervals = {}
    for k, v in iv_data.items():
        ca, wa = np.array(v["cov"]), np.array(v["w"])
        intervals[k] = {
            "coverage_pct": round(float(np.mean(ca) * 100.0), 2),
            "avg_width": round(float(np.mean(wa)), 2),
            "low_demand_cov_pct": round(float(np.mean(ca[low_m]) * 100.0), 2),
            "high_demand_cov_pct": round(float(np.mean(ca[high_m]) * 100.0), 2),
        }

    return {
        "granularity": granularity,
        "n_origins": len(origins),
        "horizon": horizon,
        "overall_models": overall,
        "baselines": baselines,
        "per_origin_wape": per_origin_wape,
        "intervals": intervals,
        "permutation_importance": perm_importances,
    }
