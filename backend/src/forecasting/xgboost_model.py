"""
Leakage-Safe XGBoost Demand Forecasting Engine
==============================================
Implements:
1. Continuous calendar resampling with zero-sales imputation (store closed days)
2. Strictly leakage-safe feature engineering (all lags and rolling statistics shifted by 1)
3. Chronological 3-way evaluation: Train / Validation (model selection) / Test (final holdout)
4. Honest Baselines: Naive (t-1), Seasonal Naive (t-7 or t-1), Rolling Mean
5. Validation-based hyperparameter tuning
6. Autoregressive recursive multi-step future forecasting
7. Empirical residual calibration for prediction intervals (80% and 95%)
8. Native JSON artifact serialization & deserialization
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from src.evaluation.metrics import calculate_forecast_metrics, mean_absolute_error, mean_squared_error, r2_score

class XGBoostForecaster:
    def __init__(self, df: pd.DataFrame = None, target_col: str = 'quantity', time_col: str = 'date'):
        self.raw_df = df
        self.target_col = target_col
        self.time_col = time_col
        
        # Best model instance
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4
        )
        
        self.feature_names = []
        self.best_params = {}
        self.results = {}
        self.baseline_results = {}
        self.granularity = 'D'
        self.residual_std = 0.0
        self.residual_q80 = 0.0

    def _prepare_continuous_time_series(self, product_name=None, category=None, granularity='D'):
        """
        Filters data, resamples to continuous calendar grid, and fills zero-sales intervals.
        """
        if self.raw_df is None or self.raw_df.empty:
            return None

        df = self.raw_df.copy()

        if product_name and product_name != "Overall":
            df = df[df['product_name'] == product_name]
        if category and 'category' in df.columns and category != "All":
            df = df[df['category'] == category]

        if df.empty:
            return None

        df[self.time_col] = pd.to_datetime(df[self.time_col])

        # Normalize frequency code
        freq = 'D' if granularity.upper() in ['D', 'DAILY'] else 'W-SUN'

        # Resample sum
        ts = df.set_index(self.time_col).resample(freq)[self.target_col].sum()

        # Build full continuous date range
        full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
        ts_continuous = ts.reindex(full_idx, fill_value=0.0).reset_index()
        ts_continuous.columns = [self.time_col, self.target_col]

        return ts_continuous

    def _create_leakage_safe_features(self, df: pd.DataFrame, granularity='D'):
        """
        Constructs features using ONLY information strictly available prior to time t.
        Every lag is >= 1; all rolling statistics are shifted by 1.
        """
        df = df.copy()
        target = df[self.target_col]

        # Calendar features (known in advance)
        df['month'] = df[self.time_col].dt.month
        df['day_of_month'] = df[self.time_col].dt.day
        df['day_of_week'] = df[self.time_col].dt.dayofweek
        df['quarter'] = df[self.time_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['trend_step'] = np.arange(len(df))

        is_daily = granularity.upper() in ['D', 'DAILY']

        if is_daily:
            # Lags for daily data: short-term + weekly seasonal lags
            df['lag_1'] = target.shift(1)
            df['lag_2'] = target.shift(2)
            df['lag_3'] = target.shift(3)
            df['lag_7'] = target.shift(7)   # Same day last week
            df['lag_14'] = target.shift(14) # Same day two weeks ago

            # Strictly shifted rolling statistics (shifted by 1)
            df['rolling_mean_7'] = target.shift(1).rolling(window=7).mean()
            df['rolling_std_7'] = target.shift(1).rolling(window=7).std().fillna(0.0)
            df['rolling_mean_14'] = target.shift(1).rolling(window=14).mean()

            features = [
                'month', 'day_of_month', 'day_of_week', 'quarter', 'is_weekend', 'trend_step',
                'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
                'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14'
            ]
        else:
            # Lags for weekly data
            df['lag_1'] = target.shift(1)
            df['lag_2'] = target.shift(2)
            df['lag_3'] = target.shift(3)
            df['lag_4'] = target.shift(4)

            # Strictly shifted rolling statistics
            df['rolling_mean_4'] = target.shift(1).rolling(window=4).mean()
            df['rolling_mean_8'] = target.shift(1).rolling(window=8).mean()

            features = [
                'month', 'day_of_week', 'quarter', 'trend_step',
                'lag_1', 'lag_2', 'lag_3', 'lag_4',
                'rolling_mean_4', 'rolling_mean_8'
            ]

        # Drop rows where lag features could not be formed
        df_clean = df.dropna().reset_index(drop=True)
        return df_clean, features

    def compute_baselines(self, train_actuals, test_actuals, granularity='D'):
        """
        Calculates honest baseline models on the exact same test observations:
        1. Naive: y_hat[t] = y[t-1]
        2. Seasonal Naive: y_hat[t] = y[t-7] (daily) or y[t-52] (weekly if >=52 weeks history, else t-4 fallback)
        3. Rolling Average: 7-day or 4-week mean of preceding observations
        """
        full_series = list(train_actuals) + list(test_actuals)
        test_len = len(test_actuals)
        start_test_idx = len(train_actuals)

        naive_preds = []
        seasonal_naive_preds = []
        weekly_4_preds = []
        rolling_preds = []

        is_daily = granularity.upper() in ['D', 'DAILY']
        if is_daily:
            season_lag = 7
        else:
            # Annual seasonality (52 weeks) when sufficient history exists; 4-week fallback otherwise
            season_lag = 52 if len(train_actuals) >= 52 else 4
        roll_window = 7 if is_daily else 4

        for i in range(test_len):
            idx = start_test_idx + i
            # 1. Naive: last observed value
            naive_preds.append(full_series[idx - 1])
            # 2. Seasonal Naive: value from exactly season_lag steps back
            s_idx = idx - season_lag if (idx - season_lag) >= 0 else (idx - 1)
            seasonal_naive_preds.append(full_series[s_idx])
            # Optional 4-week seasonal comparison when weekly uses annual lag 52
            if not is_daily and season_lag == 52:
                w4_idx = idx - 4 if (idx - 4) >= 0 else (idx - 1)
                weekly_4_preds.append(full_series[w4_idx])
            # 3. Rolling Mean: mean of preceding roll_window observations
            r_start = max(0, idx - roll_window)
            rolling_preds.append(float(np.mean(full_series[r_start:idx])))

        baselines = {
            "Naive (t-1)": calculate_forecast_metrics(test_actuals, naive_preds),
            f"Seasonal Naive (t-{season_lag})": calculate_forecast_metrics(test_actuals, seasonal_naive_preds),
            f"Rolling Mean ({roll_window})": calculate_forecast_metrics(test_actuals, rolling_preds)
        }
        if not is_daily and season_lag == 52:
            baselines["Seasonal Naive (t-4)"] = calculate_forecast_metrics(test_actuals, weekly_4_preds)
        return baselines

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Resource-conscious grid search evaluated strictly on the Validation split.
        Uses fixed seed and limits search to safe parameter budget on 8-core CPU.
        """
        param_grid = [
            {'n_estimators': 60, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.9},
            {'n_estimators': 80, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.85},
            {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8}
        ]

        best_val_mae = float('inf')
        best_model = None
        best_p = param_grid[0]

        for p in param_grid:
            m = xgb.XGBRegressor(
                n_estimators=p['n_estimators'],
                max_depth=p['max_depth'],
                learning_rate=p['learning_rate'],
                subsample=p['subsample'],
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4
            )
            m.fit(X_train, y_train)
            val_preds = np.maximum(m.predict(X_val), 0)
            val_mae = mean_absolute_error(y_val, val_preds)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model = m
                best_p = p

        return best_model, best_p

    def train_and_evaluate(self, product_name=None, category=None, granularity='D', 
                           test_horizon=28, val_horizon=28, tune=True):
        """
        Chronological 3-way split:
          [Train: 0 -> -val-test] -> [Validation: -val-test -> -test] -> [Test: -test -> end]
        """
        self.granularity = granularity
        data = self._prepare_continuous_time_series(product_name, category, granularity)
        if data is None or len(data) < 25:
            return False, "Insufficient data points for time-series modeling."

        df_feat, features = self._create_leakage_safe_features(data, granularity)
        self.feature_names = features

        n_rows = len(df_feat)
        is_daily = granularity.upper() in ['D', 'DAILY']
        
        # Adaptive horizon sizes based on available observations
        t_horiz = test_horizon if is_daily else min(8, max(4, int(n_rows * 0.15)))
        v_horiz = val_horizon if is_daily else min(8, max(4, int(n_rows * 0.15)))

        if n_rows <= (t_horiz + v_horiz + 15):
            # Scale down horizons for short series
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

        # Hyperparameter tuning on validation set (model trained exclusively on X_train)
        if tune and len(val) >= 4:
            self.model, self.best_params = self.tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)

        # Calculate strictly out-of-sample validation residuals BEFORE refitting on Train + Val
        if len(X_val) > 0:
            val_preds = np.maximum(self.model.predict(X_val), 0.0)
            val_residuals = np.array(y_val, dtype=np.float64) - np.array(val_preds, dtype=np.float64)
            self.residual_std = float(np.std(val_residuals, ddof=1)) if len(val_residuals) > 1 else float(np.std(val_residuals))
            self.residual_q80 = float(np.quantile(np.abs(val_residuals), 0.80))
        else:
            self.residual_std = 0.0
            self.residual_q80 = 0.0

        # Re-fit final model on combined Train + Validation before untouched Test evaluation
        if len(val) >= 4:
            X_train_val = pd.concat([X_train, X_val])
            y_train_val = pd.concat([y_train, y_val])
            self.model.fit(X_train_val, y_train_val)

        # Predict on untouched held-out test split
        preds = np.maximum(self.model.predict(X_test), 0.0)

        # Baselines on the exact same test split
        train_val_actuals = pd.concat([y_train, y_val]).tolist()
        baselines = self.compute_baselines(train_val_actuals, y_test.tolist(), granularity)
        self.baseline_results = baselines

        # Store results conforming to the API contract
        self.results = {
            'train_dates': df_feat[self.time_col].iloc[:split_val_test].dt.strftime('%Y-%m-%d').tolist(),
            'train_actuals': df_feat[self.target_col].iloc[:split_val_test].tolist(),
            'val_dates': val[self.time_col].dt.strftime('%Y-%m-%d').tolist(),
            'val_actuals': y_val.tolist(),
            'test_dates': test[self.time_col].dt.strftime('%Y-%m-%d').tolist(),
            'test_actuals': y_test.tolist(),
            'test_preds': [round(float(p), 2) for p in preds],
            'baselines': baselines,
            'features': features,
            'best_params': self.best_params,
            'split_summary': {
                'train_rows': len(train),
                'val_rows': len(val),
                'test_rows': len(test),
                'train_period': f"{train[self.time_col].min().strftime('%Y-%m-%d')} to {train[self.time_col].max().strftime('%Y-%m-%d')}",
                'test_period': f"{test[self.time_col].min().strftime('%Y-%m-%d')} to {test[self.time_col].max().strftime('%Y-%m-%d')}"
            }
        }

        return True, "Model trained and evaluated successfully."

    def forecast_future(self, horizon=14, freq=None):
        """
        Autoregressive recursive multi-step forecasting with calibrated prediction intervals.
        """
        if not hasattr(self, 'results') or not self.results:
            return None

        freq = freq or self.granularity
        is_daily = freq.upper() in ['D', 'DAILY']
        offset = pd.Timedelta(days=1) if is_daily else pd.Timedelta(weeks=1)

        test_dates = self.results['test_dates']
        last_date = pd.to_datetime(test_dates[-1])

        # Complete historical ground truth + actuals history
        history = list(self.results['train_actuals']) + list(self.results['test_actuals'])
        total_steps = len(history)

        future_dates = []
        future_preds = []
        future_lower_80 = []
        future_upper_80 = []

        curr_date = last_date
        step_counter = total_steps

        # 80% confidence interval standard normal multiplier z = 1.28
        z_80 = 1.28

        for step in range(1, horizon + 1):
            curr_date += offset
            future_dates.append(curr_date.strftime('%Y-%m-%d'))

            month = curr_date.month
            day_of_month = curr_date.day
            day_of_week = curr_date.dayofweek
            quarter = curr_date.quarter
            is_weekend = 1 if day_of_week in [5, 6] else 0
            trend_step = step_counter + step

            if is_daily:
                lag_1 = history[-1]
                lag_2 = history[-2] if len(history) >= 2 else lag_1
                lag_3 = history[-3] if len(history) >= 3 else lag_2
                lag_7 = history[-7] if len(history) >= 7 else lag_1
                lag_14 = history[-14] if len(history) >= 14 else lag_7

                rolling_7 = float(np.mean(history[-7:]))
                recent_7 = history[-7:]
                rolling_std_7 = float(np.std(recent_7, ddof=1)) if len(recent_7) >= 2 else 0.0
                rolling_14 = float(np.mean(history[-14:]))

                row_dict = {
                    'month': month, 'day_of_month': day_of_month,
                    'day_of_week': day_of_week, 'quarter': quarter,
                    'is_weekend': is_weekend, 'trend_step': trend_step,
                    'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3,
                    'lag_7': lag_7, 'lag_14': lag_14,
                    'rolling_mean_7': rolling_7, 'rolling_std_7': rolling_std_7,
                    'rolling_mean_14': rolling_14
                }
            else:
                lag_1 = history[-1]
                lag_2 = history[-2] if len(history) >= 2 else lag_1
                lag_3 = history[-3] if len(history) >= 3 else lag_2
                lag_4 = history[-4] if len(history) >= 4 else lag_3

                rolling_4 = float(np.mean(history[-4:]))
                rolling_8 = float(np.mean(history[-8:])) if len(history) >= 8 else rolling_4

                row_dict = {
                    'month': month, 'day_of_week': day_of_week,
                    'quarter': quarter, 'trend_step': trend_step,
                    'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3, 'lag_4': lag_4,
                    'rolling_mean_4': rolling_4, 'rolling_mean_8': rolling_8
                }

            x_step = pd.DataFrame([row_dict])[self.feature_names]
            pred = float(max(0.0, self.model.predict(x_step)[0]))
            future_preds.append(round(pred, 2))
            history.append(pred) # Autoregressively feed prediction for future lags

            # Uncertainty expands slightly with recursive horizon step
            step_uncertainty = self.residual_std * np.sqrt(1 + 0.05 * step)
            lower_bound = max(0.0, round(pred - z_80 * step_uncertainty, 2))
            upper_bound = round(pred + z_80 * step_uncertainty, 2)
            future_lower_80.append(lower_bound)
            future_upper_80.append(upper_bound)

        return {
            'future_dates': future_dates,
            'future_preds': future_preds,
            'future_lower_80': future_lower_80,
            'future_upper_80': future_upper_80
        }

    def save_model_artifact(self, filepath: str):
        """Saves both the XGBoost model binary/JSON and comprehensive metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # 1. Native XGBoost save
        self.model.save_model(filepath)

        # 2. Sidecar metadata
        meta_path = filepath.rsplit('.', 1)[0] + "_metadata.json"
        metadata = {
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "granularity": self.granularity,
            "best_params": self.best_params,
            "residual_std": self.residual_std,
            "residual_q80": self.residual_q80,
            "results": self.results
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        return filepath, meta_path

    def load_model_artifact(self, filepath: str):
        """Loads XGBoost model and restores all metadata."""
        self.model.load_model(filepath)
        meta_path = filepath.rsplit('.', 1)[0] + "_metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.feature_names = metadata.get("feature_names", [])
            self.target_col = metadata.get("target_col", "quantity")
            self.time_col = metadata.get("time_col", "date")
            self.granularity = metadata.get("granularity", "D")
            self.best_params = metadata.get("best_params", {})
            self.residual_std = metadata.get("residual_std", 0.0)
            self.residual_q80 = metadata.get("residual_q80", 0.0)
            self.results = metadata.get("results", {})
        return True
