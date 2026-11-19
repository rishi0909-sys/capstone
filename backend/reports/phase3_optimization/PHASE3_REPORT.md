# Phase 3: Forecasting Model Optimization & Robustness Report

## Daily (14-Day Horizon, 10 Origins)

### Model Comparison
| Model | MAE | RMSE | WAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 6,422.1 | 9,595.6 | 32.10% | 0.5131 |
| HistGradientBoosting | 6,102.8 | 9,538.2 | 30.51% | 0.5190 |
| RandomForest | 6,832.1 | 10,259.2 | 34.15% | 0.4435 |
| ExtraTrees | 6,133.6 | 9,774.3 | 30.66% | 0.4948 |
| Ridge | 6,912.1 | 9,750.6 | 34.55% | 0.4973 |
| XGBoost_Enhanced | 6,422.4 | 9,447.4 | 32.10% | 0.5281 |

### Demand-Dependent Prediction Intervals (Target: 80%)
| Method | Coverage (%) | Avg Width | Low-Demand Cov (%) | High-Demand Cov (%) |
| :--- | :---: | :---: | :---: | :---: |
| gaussian | 80.00% | 19,326.9 | 84.29% | 75.71% |
| normalized_conformal | 73.57% | 15,293.8 | 80.00% | 67.14% |
| regime_conformal | 75.71% | 16,610.5 | 81.43% | 70.00% |

### Top 5 Features (Permutation Importance)
- `day_of_week`: 0.3472
- `lag_7`: 0.3044
- `lag_3`: 0.0608
- `lag_14`: 0.0489
- `lag_1`: 0.0274

## Weekly (4-Week Horizon, 8 Origins)

### Model Comparison
| Model | MAE | RMSE | WAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 19,147.9 | 27,096.0 | 15.88% | 0.5549 |
| HistGradientBoosting | 18,619.5 | 26,553.5 | 15.44% | 0.5725 |
| RandomForest | 19,371.6 | 26,325.4 | 16.07% | 0.5798 |
| ExtraTrees | 21,666.6 | 28,033.9 | 17.97% | 0.5235 |
| Ridge | 20,368.7 | 29,332.3 | 16.89% | 0.4784 |
| XGBoost_Enhanced | 19,577.0 | 27,089.0 | 16.24% | 0.5551 |

### Demand-Dependent Prediction Intervals (Target: 80%)
| Method | Coverage (%) | Avg Width | Low-Demand Cov (%) | High-Demand Cov (%) |
| :--- | :---: | :---: | :---: | :---: |
| gaussian | 81.25% | 57,500.2 | 93.75% | 68.75% |
| normalized_conformal | 84.38% | 63,760.4 | 93.75% | 75.00% |
| regime_conformal | 71.88% | 57,199.7 | 81.25% | 62.50% |

### Top 5 Features (Permutation Importance)
- `momentum_4_8`: 0.4001
- `lag_3`: 0.3173
- `lag_1`: 0.2411
- `lag_4`: 0.1410
- `month`: 0.1117

