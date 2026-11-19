"""
Phase 4: Model Validation, Statistical Reliability & Production Readiness Audit
=============================================================================
Evaluates HistGradientBoosting vs XGBoost vs strongest baseline across:
1. Per-origin performance & paired differences
2. Training-window robustness (expanding, 12-month, 18-month)
3. Horizon-group breakdown (Daily: 1-3, 4-7, 8-14; Weekly: 1, 2, 3-4)
4. Origin-defined demand-regime interval reliability (no future leakage)
5. Permutation importance on chronological validation window
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.evaluation.metrics import calculate_forecast_metrics
from src.forecasting.xgboost_model import XGBoostForecaster
from src.evaluation.backtesting import _compute_origin_baselines
from src.evaluation.phase3_experiments import build_features, _recursive_predict_generic


def _fit_and_predict(model_name: str, is_daily: bool, train_df: pd.DataFrame, features: list, history: list, last_date, horizon: int, granularity: str):
    if model_name == "HistGradientBoosting":
        model = HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42)
    else:
        xgb_p = (
            {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8}
            if is_daily
            else {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9}
        )
        model = xgb.XGBRegressor(**xgb_p, colsample_bytree=0.8, random_state=42, n_jobs=4)

    model.fit(train_df[features], train_df["quantity"])
    preds = _recursive_predict_generic(model, features, history, last_date, horizon, granularity, enhanced=False)
    return model, preds


def run_phase4_validation(
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

    strongest_baseline = "Seasonal Naive (t-7)" if is_daily else "Rolling Mean (4)"
    models_to_eval = ["HistGradientBoosting", "XGBoost"]

    per_origin = []
    horizon_groups = (
        {"Days 1-3": [1, 2, 3], "Days 4-7": [4, 5, 6, 7], "Days 8-14": list(range(8, 15))}
        if is_daily
        else {"Week 1": [1], "Week 2": [2], "Weeks 3-4": [3, 4]}
    )
    h_group_data = {
        g: {"actual": [], "HistGradientBoosting": [], "XGBoost": [], "gaussian_cov": [], "norm_cov": []}
        for g in horizon_groups
    }

    iv_records = {
        "gaussian": {"cov": [], "w": [], "regime": [], "origin": []},
        "normalized_conformal": {"cov": [], "w": [], "regime": [], "origin": []},
        "regime_conformal": {"cov": [], "w": [], "regime": [], "origin": []},
    }
    perm_importance = {}

    for idx_o, origin_idx in enumerate(origins):
        ts_avail = ts.iloc[:origin_idx].copy()
        actuals_h = ts.iloc[origin_idx : origin_idx + horizon]["quantity"].tolist()
        history = ts_avail["quantity"].tolist()
        last_date = ts_avail["date"].iloc[-1]
        origin_date_str = str(last_date)[:10]

        df_feat, features = build_features(ts_avail, granularity=granularity, enhanced=False)
        split_idx = len(df_feat) - cal_window
        train_df, cal_df = df_feat.iloc[:split_idx], df_feat.iloc[split_idx:]

        # Origin-defined demand regime thresholds (using ONLY history up to origin)
        recent_level = float(np.mean(history[-7:])) if is_daily else float(np.mean(history[-4:]))
        q33, q67 = float(np.quantile(history, 0.33)), float(np.quantile(history, 0.67))
        origin_regime = "low" if recent_level <= q33 else ("medium" if recent_level <= q67 else "high")

        # Baselines
        b_h = _compute_origin_baselines(history, horizon, granularity)
        base_preds = b_h[strongest_baseline]
        base_met = calculate_forecast_metrics(actuals_h, base_preds)

        origin_entry = {
            "origin_date": origin_date_str,
            "train_rows": len(df_feat),
            "horizon": horizon,
            "models": {strongest_baseline: base_met},
        }

        preds_map = {}
        for m_name in models_to_eval:
            model, preds_h = _fit_and_predict(m_name, is_daily, df_feat, features, history, last_date, horizon, granularity)
            preds_map[m_name] = preds_h
            origin_entry["models"][m_name] = calculate_forecast_metrics(actuals_h, preds_h)

            if m_name == "HistGradientBoosting" and idx_o == len(origins) - 1:
                perm = permutation_importance(model, cal_df[features], cal_df["quantity"], n_repeats=5, random_state=42)
                perm_importance = {
                    f: round(float(imp), 4)
                    for f, imp in sorted(zip(features, perm.importances_mean), key=lambda x: x[1], reverse=True)
                }

        # Interval calibration on HistGradientBoosting (split conformal on cal_df)
        cal_m, _ = _fit_and_predict("HistGradientBoosting", is_daily, train_df, features, history[:-cal_window], ts_avail["date"].iloc[-cal_window - 1], horizon, granularity)
        cal_p = np.maximum(cal_m.predict(cal_df[features]), 0.0)
        cal_res = np.array(cal_df["quantity"], dtype=np.float64) - cal_p
        res_std = float(np.std(cal_res, ddof=1)) if len(cal_res) > 1 else 0.0

        scale_col = "rolling_std_7" if is_daily else "rolling_mean_4"
        cal_scales = np.maximum(cal_df[scale_col].values, 1.0)
        norm_q80 = float(np.quantile(np.abs(cal_res) / cal_scales, 0.80))
        recent_scale = max(float(np.std(history[-7:], ddof=1)) if is_daily else float(np.mean(history[-4:])), 1.0)

        cal_med = float(np.median(cal_df["quantity"]))
        low_r = np.abs(cal_res[cal_df["quantity"].values <= cal_med])
        high_r = np.abs(cal_res[cal_df["quantity"].values > cal_med])
        q80_low = float(np.quantile(low_r, 0.80)) if len(low_r) >= 3 else float(np.quantile(np.abs(cal_res), 0.80))
        q80_high = float(np.quantile(high_r, 0.80)) if len(high_r) >= 3 else float(np.quantile(np.abs(cal_res), 0.80))

        hgb_preds = preds_map["HistGradientBoosting"]
        xgb_preds = preds_map["XGBoost"]

        for step_i in range(horizon):
            h = step_i + 1
            yt = actuals_h[step_i]
            yh = hgb_preds[step_i]

            g_half = 1.28 * res_std * np.sqrt(1.0 + 0.05 * h)
            g_cov = int(max(0.0, yh - g_half) <= yt <= yh + g_half)
            iv_records["gaussian"]["cov"].append(g_cov)
            iv_records["gaussian"]["w"].append((yh + g_half) - max(0.0, yh - g_half))
            iv_records["gaussian"]["regime"].append(origin_regime)
            iv_records["gaussian"]["origin"].append(origin_date_str)

            n_half = norm_q80 * recent_scale
            n_cov = int(max(0.0, yh - n_half) <= yt <= yh + n_half)
            iv_records["normalized_conformal"]["cov"].append(n_cov)
            iv_records["normalized_conformal"]["w"].append((yh + n_half) - max(0.0, yh - n_half))
            iv_records["normalized_conformal"]["regime"].append(origin_regime)
            iv_records["normalized_conformal"]["origin"].append(origin_date_str)

            r_half = q80_low if origin_regime == "low" else q80_high
            r_cov = int(max(0.0, yh - r_half) <= yt <= yh + r_half)
            iv_records["regime_conformal"]["cov"].append(r_cov)
            iv_records["regime_conformal"]["w"].append((yh + r_half) - max(0.0, yh - r_half))
            iv_records["regime_conformal"]["regime"].append(origin_regime)
            iv_records["regime_conformal"]["origin"].append(origin_date_str)

            for g_name, h_list in horizon_groups.items():
                if h in h_list:
                    h_group_data[g_name]["actual"].append(yt)
                    h_group_data[g_name]["HistGradientBoosting"].append(yh)
                    h_group_data[g_name]["XGBoost"].append(xgb_preds[step_i])
                    h_group_data[g_name]["gaussian_cov"].append(g_cov)
                    h_group_data[g_name]["norm_cov"].append(n_cov)

        hgb_wape = origin_entry["models"]["HistGradientBoosting"]["WAPE"]
        xgb_wape = origin_entry["models"]["XGBoost"]["WAPE"]
        origin_entry["paired_wape_diff_xgb_minus_hgb"] = round(xgb_wape - hgb_wape, 2)
        per_origin.append(origin_entry)

    # Summary stats across origins
    hgb_wapes = [o["models"]["HistGradientBoosting"]["WAPE"] for o in per_origin]
    xgb_wapes = [o["models"]["XGBoost"]["WAPE"] for o in per_origin]
    base_wapes = [o["models"][strongest_baseline]["WAPE"] for o in per_origin]
    paired_diffs = [o["paired_wape_diff_xgb_minus_hgb"] for o in per_origin]

    origin_summary = {
        "hgb_wins_vs_xgb": int(sum(1 for d in paired_diffs if d > 0)),
        "hgb_wins_vs_baseline": int(sum(1 for h, b in zip(hgb_wapes, base_wapes) if h < b)),
        "xgb_wins_vs_baseline": int(sum(1 for x, b in zip(xgb_wapes, base_wapes) if x < b)),
        "total_origins": len(per_origin),
        "hgb_wape_stats": {
            "mean": round(float(np.mean(hgb_wapes)), 2),
            "median": round(float(np.median(hgb_wapes)), 2),
            "std": round(float(np.std(hgb_wapes, ddof=1)), 2) if len(hgb_wapes) > 1 else 0.0,
            "best": round(float(np.min(hgb_wapes)), 2),
            "worst": round(float(np.max(hgb_wapes)), 2),
        },
        "xgb_wape_stats": {
            "mean": round(float(np.mean(xgb_wapes)), 2),
            "median": round(float(np.median(xgb_wapes)), 2),
            "std": round(float(np.std(xgb_wapes, ddof=1)), 2) if len(xgb_wapes) > 1 else 0.0,
            "best": round(float(np.min(xgb_wapes)), 2),
            "worst": round(float(np.max(xgb_wapes)), 2),
        },
        "paired_diff_xgb_minus_hgb": {
            "mean": round(float(np.mean(paired_diffs)), 2),
            "median": round(float(np.median(paired_diffs)), 2),
        },
    }

    # Horizon group summary
    horizon_summary = {}
    for g_name, gd in h_group_data.items():
        horizon_summary[g_name] = {
            "HistGradientBoosting": calculate_forecast_metrics(gd["actual"], gd["HistGradientBoosting"]),
            "XGBoost": calculate_forecast_metrics(gd["actual"], gd["XGBoost"]),
            "gaussian_cov_pct": round(float(np.mean(gd["gaussian_cov"]) * 100.0), 2),
            "normalized_conformal_cov_pct": round(float(np.mean(gd["norm_cov"]) * 100.0), 2),
        }

    # Interval summary by origin-defined regime
    interval_summary = {}
    for iv_name, ivd in iv_records.items():
        cov_arr = np.array(ivd["cov"])
        w_arr = np.array(ivd["w"])
        reg_arr = np.array(ivd["regime"])
        interval_summary[iv_name] = {
            "overall_cov_pct": round(float(np.mean(cov_arr) * 100.0), 2),
            "avg_width": round(float(np.mean(w_arr)), 2),
            "by_regime_cov_pct": {
                r: round(float(np.mean(cov_arr[reg_arr == r]) * 100.0), 2)
                for r in ["low", "medium", "high"] if np.any(reg_arr == r)
            },
        }

    # Training-window robustness (expanding vs 12-month vs 18-month) on identical origins
    win_sizes = {"expanding": None, "12_month": 365 if is_daily else 52, "18_month": 548 if is_daily else 78}
    window_robustness = {}
    for w_label, w_len in win_sizes.items():
        w_actuals, w_hgb, w_xgb = [], [], []
        for origin_idx in origins:
            start_idx = 0 if w_len is None else max(0, origin_idx - w_len)
            ts_w = ts.iloc[start_idx:origin_idx].copy()
            if len(ts_w) < 30:
                continue
            df_fw, feats_w = build_features(ts_w, granularity=granularity, enhanced=False)
            hist_w = ts_w["quantity"].tolist()
            last_dw = ts_w["date"].iloc[-1]
            act_w = ts.iloc[origin_idx : origin_idx + horizon]["quantity"].tolist()
            _, p_hgb = _fit_and_predict("HistGradientBoosting", is_daily, df_fw, feats_w, hist_w, last_dw, horizon, granularity)
            _, p_xgb = _fit_and_predict("XGBoost", is_daily, df_fw, feats_w, hist_w, last_dw, horizon, granularity)
            w_actuals.extend(act_w)
            w_hgb.extend(p_hgb)
            w_xgb.extend(p_xgb)
        if w_actuals:
            window_robustness[w_label] = {
                "HistGradientBoosting": calculate_forecast_metrics(w_actuals, w_hgb),
                "XGBoost": calculate_forecast_metrics(w_actuals, w_xgb),
            }

    return {
        "granularity": granularity,
        "strongest_baseline": strongest_baseline,
        "origin_summary": origin_summary,
        "per_origin": per_origin,
        "horizon_summary": horizon_summary,
        "interval_summary": interval_summary,
        "window_robustness": window_robustness,
        "permutation_importance": perm_importance,
    }
