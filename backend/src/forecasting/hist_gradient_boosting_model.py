"""
Production HistGradientBoosting Demand Forecasting Engine
=========================================================
Leakage-safe daily and weekly demand forecaster using scikit-learn's
HistGradientBoostingRegressor, sharing the validated feature pipeline
with XGBoostForecaster and supporting joblib + JSON metadata serialization.
"""

import os
import json
import time
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation.metrics import calculate_forecast_metrics, mean_absolute_error
from src.forecasting.xgboost_model import XGBoostForecaster


class HistGradientBoostingForecaster:
    MODEL_VERSION = "1.0.0"

    def __init__(self, df: pd.DataFrame = None, target_col: str = "quantity", time_col: str = "date"):
        self.raw_df = df
        self.target_col = target_col
        self.time_col = time_col

        self.model = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )

        self.feature_names = []
        self.best_params = {"max_iter": 150, "max_depth": 4, "learning_rate": 0.05}
        self.results = {}
        self.baseline_results = {}
        self.granularity = "D"
        self.residual_std = 0.0
        self.residual_q80 = 0.0
        self.metadata = {}

        # Reuse validated preprocessing & feature helpers from XGBoostForecaster
        self._helper = XGBoostForecaster(df=df, target_col=target_col, time_col=time_col)

    def _prepare_continuous_time_series(self, product_name=None, category=None, granularity="D"):
        self._helper.raw_df = self.raw_df
        return self._helper._prepare_continuous_time_series(product_name, category, granularity)

    def _create_leakage_safe_features(self, df: pd.DataFrame, granularity="D"):
        return self._helper._create_leakage_safe_features(df, granularity)

    def compute_baselines(self, train_actuals, test_actuals, granularity="D"):
        return self._helper.compute_baselines(train_actuals, test_actuals, granularity)

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        param_grid = [
            {"max_iter": 100, "max_depth": 3, "learning_rate": 0.05},
            {"max_iter": 150, "max_depth": 4, "learning_rate": 0.05},
            {"max_iter": 200, "max_depth": 4, "learning_rate": 0.03},
            {"max_iter": 150, "max_depth": 5, "learning_rate": 0.05},
        ]
        best_val_mae = float("inf")
        best_model = None
        best_p = param_grid[0]

        for p in param_grid:
            m = HistGradientBoostingRegressor(
                max_iter=p["max_iter"],
                max_depth=p["max_depth"],
                learning_rate=p["learning_rate"],
                random_state=42,
            )
            m.fit(X_train, y_train)
            val_preds = np.maximum(m.predict(X_val), 0.0)
            val_mae = mean_absolute_error(y_val, val_preds)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model = m
                best_p = p

        return best_model, best_p

    def train_and_evaluate(
        self,
        product_name=None,
        category=None,
        granularity="D",
        test_horizon=28,
        val_horizon=28,
        tune=True,
        dataset_id="unknown",
    ):
        self.granularity = granularity
        data = self._prepare_continuous_time_series(product_name, category, granularity)
        if data is None or len(data) < 25:
            return False, "Insufficient data points for time-series modeling."

        # Sanitize target for non-finite values
        data[self.target_col] = pd.to_numeric(data[self.target_col], errors="coerce").fillna(0.0)
        data[self.target_col] = np.maximum(data[self.target_col].values, 0.0)

        df_feat, features = self._create_leakage_safe_features(data, granularity)
        if len(df_feat) < 15:
            return False, "Insufficient post-lag observations for time-series modeling."

        self.feature_names = features
        n_rows = len(df_feat)
        is_daily = granularity.upper() in ["D", "DAILY"]

        t_horiz = test_horizon if is_daily else min(8, max(4, int(n_rows * 0.15)))
        v_horiz = val_horizon if is_daily else min(8, max(4, int(n_rows * 0.15)))

        if n_rows <= (t_horiz + v_horiz + 15):
            t_horiz = max(4, int(n_rows * 0.15))
            v_horiz = max(4, int(n_rows * 0.15))

        split_train_val = n_rows - t_horiz - v_horiz
        split_val_test = n_rows - t_horiz

        train = df_feat.iloc[:split_train_val]
        val = df_feat.iloc[split_train_val:split_val_test]
        test = df_feat.iloc[split_val_test:]

        X_train, y_train = train[features], train[self.target_col]
        X_val, y_val = val[features], val[self.target_col]
        X_test, y_test = test[features], test[self.target_col]

        if tune and len(val) >= 4:
            self.model, self.best_params = self.tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)

        # Strictly out-of-sample validation residuals BEFORE Train+Val refit
        if len(X_val) > 0:
            val_preds = np.maximum(self.model.predict(X_val), 0.0)
            val_residuals = np.array(y_val, dtype=np.float64) - np.array(val_preds, dtype=np.float64)
            self.residual_std = float(np.std(val_residuals, ddof=1)) if len(val_residuals) > 1 else float(np.std(val_residuals))
            self.residual_q80 = float(np.quantile(np.abs(val_residuals), 0.80))
        else:
            self.residual_std = 0.0
            self.residual_q80 = 0.0

        # Refit final production model on Train + Validation
        if len(val) >= 4:
            X_train_val = pd.concat([X_train, X_val])
            y_train_val = pd.concat([y_train, y_val])
            self.model.fit(X_train_val, y_train_val)

        preds = np.maximum(self.model.predict(X_test), 0.0)
        test_metrics = calculate_forecast_metrics(y_test.tolist(), preds.tolist())

        train_val_actuals = pd.concat([y_train, y_val]).tolist()
        baselines = self.compute_baselines(train_val_actuals, y_test.tolist(), granularity)
        self.baseline_results = baselines

        self.results = {
            "train_dates": df_feat[self.time_col].iloc[:split_val_test].dt.strftime("%Y-%m-%d").tolist(),
            "train_actuals": df_feat[self.target_col].iloc[:split_val_test].tolist(),
            "val_dates": val[self.time_col].dt.strftime("%Y-%m-%d").tolist(),
            "val_actuals": y_val.tolist(),
            "test_dates": test[self.time_col].dt.strftime("%Y-%m-%d").tolist(),
            "test_actuals": y_test.tolist(),
            "test_preds": [round(float(p), 2) for p in preds],
            "metrics": test_metrics,
            "baselines": baselines,
            "features": features,
            "best_params": self.best_params,
            "split_summary": {
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "train_period": f"{train[self.time_col].min().strftime('%Y-%m-%d')} to {train[self.time_col].max().strftime('%Y-%m-%d')}",
                "test_period": f"{test[self.time_col].min().strftime('%Y-%m-%d')} to {test[self.time_col].max().strftime('%Y-%m-%d')}",
            },
        }

        self.metadata = {
            "model_type": "HistGradientBoostingRegressor",
            "model_version": self.MODEL_VERSION,
            "granularity": self.granularity,
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "dataset_id": dataset_id,
            "date_range": f"{df_feat[self.time_col].min().strftime('%Y-%m-%d')} to {df_feat[self.time_col].max().strftime('%Y-%m-%d')}",
            "training_config": {"test_horizon": t_horiz, "val_horizon": v_horiz, "tune": tune},
            "hyperparameters": self.best_params,
            "library_versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "evaluation_metrics": test_metrics,
            "calibration_metadata": {
                "interval_methodology": "approximate_experimental_out_of_sample_residual_std",
                "nominal_coverage": 0.80,
                "is_guaranteed_coverage": False,
                "residual_std": self.residual_std,
                "residual_q80": self.residual_q80,
                "calibration_samples": len(X_val),
            },
        }

        return True, "HistGradientBoosting model trained and evaluated successfully."

    def forecast_future(self, horizon=14, freq=None):
        if not hasattr(self, "results") or not self.results:
            return None

        freq = freq or self.granularity
        is_daily = freq.upper() in ["D", "DAILY"]
        offset = pd.Timedelta(days=1) if is_daily else pd.Timedelta(weeks=1)

        test_dates = self.results.get("test_dates", [])
        if not test_dates:
            return None
        last_date = pd.to_datetime(test_dates[-1])

        raw_hist = list(self.results.get("train_actuals", [])) + list(self.results.get("test_actuals", []))
        history = [float(x) if np.isfinite(x) else 0.0 for x in raw_hist]
        if not history:
            return None

        total_steps = len(history)
        future_dates = []
        future_preds = []
        has_cal = self.residual_std > 0.0
        future_lower_80 = [] if has_cal else None
        future_upper_80 = [] if has_cal else None

        curr_date = last_date
        z_80 = 1.28

        for step in range(1, horizon + 1):
            curr_date += offset
            future_dates.append(curr_date.strftime("%Y-%m-%d"))

            month = curr_date.month
            day_of_month = curr_date.day
            day_of_week = curr_date.dayofweek
            quarter = curr_date.quarter
            is_weekend = 1 if day_of_week in [5, 6] else 0
            trend_step = total_steps + step

            if is_daily:
                lag_1 = history[-1]
                lag_2 = history[-2] if len(history) >= 2 else lag_1
                lag_3 = history[-3] if len(history) >= 3 else lag_2
                lag_7 = history[-7] if len(history) >= 7 else lag_1
                lag_14 = history[-14] if len(history) >= 14 else lag_7
                recent_7 = history[-7:]
                row_dict = {
                    "month": month, "day_of_month": day_of_month,
                    "day_of_week": day_of_week, "quarter": quarter,
                    "is_weekend": is_weekend, "trend_step": trend_step,
                    "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
                    "lag_7": lag_7, "lag_14": lag_14,
                    "rolling_mean_7": float(np.mean(recent_7)),
                    "rolling_std_7": float(np.std(recent_7, ddof=1)) if len(recent_7) >= 2 else 0.0,
                    "rolling_mean_14": float(np.mean(history[-14:])),
                }
            else:
                lag_1 = history[-1]
                lag_2 = history[-2] if len(history) >= 2 else lag_1
                lag_3 = history[-3] if len(history) >= 3 else lag_2
                lag_4 = history[-4] if len(history) >= 4 else lag_3
                rolling_4 = float(np.mean(history[-4:]))
                rolling_8 = float(np.mean(history[-8:])) if len(history) >= 8 else rolling_4
                row_dict = {
                    "month": month, "day_of_week": day_of_week,
                    "quarter": quarter, "trend_step": trend_step,
                    "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_4": lag_4,
                    "rolling_mean_4": rolling_4, "rolling_mean_8": rolling_8,
                }

            x_step = pd.DataFrame([row_dict])[self.feature_names]
            raw_pred = float(self.model.predict(x_step)[0])
            pred = float(max(0.0, raw_pred)) if np.isfinite(raw_pred) else 0.0
            future_preds.append(round(pred, 2))
            history.append(pred)

            if has_cal:
                step_uncertainty = self.residual_std * np.sqrt(1.0 + 0.05 * step)
                future_lower_80.append(max(0.0, round(pred - z_80 * step_uncertainty, 2)))
                future_upper_80.append(round(pred + z_80 * step_uncertainty, 2))

        return {
            "future_dates": future_dates,
            "future_preds": future_preds,
            "future_lower_80": future_lower_80,
            "future_upper_80": future_upper_80,
            "interval_methodology": "approximate_experimental_out_of_sample_residual_std" if has_cal else "uncalibrated",
        }

    def save_model_artifact(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)
        meta_path = filepath.rsplit(".", 1)[0] + "_metadata.json"
        payload = {
            **self.metadata,
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "granularity": self.granularity,
            "best_params": self.best_params,
            "residual_std": self.residual_std,
            "residual_q80": self.residual_q80,
            "results": self.results,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return filepath, meta_path

    def load_model_artifact(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found: {filepath}")
        meta_path = filepath.rsplit(".", 1)[0] + "_metadata.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.model = joblib.load(filepath)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            raise ValueError("Invalid metadata: missing feature_names")

        self.metadata = metadata
        self.feature_names = feature_names
        self.target_col = metadata.get("target_col", "quantity")
        self.time_col = metadata.get("time_col", "date")
        self.granularity = metadata.get("granularity", "D")
        self.best_params = metadata.get("hyperparameters", metadata.get("best_params", {}))
        self.residual_std = float(metadata.get("residual_std", 0.0))
        self.residual_q80 = float(metadata.get("residual_q80", 0.0))
        self.results = metadata.get("results", {})
        return True
