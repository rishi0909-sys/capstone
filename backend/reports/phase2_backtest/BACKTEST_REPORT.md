# Phase 2: Rolling-Origin Backtesting & Prediction Interval Calibration Report

**Dataset**: `backend/data/dataset_3_clean.csv` (1,037,366 records, `2009-12-01 to 2011-12-09`)

## Daily (14-Day Horizon)
- **Origins**: 10 (`2011-07-22` to `2011-11-25`)

### Model Comparison
| Model | MAE | RMSE | WAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 6,422.1 | 9,595.6 | 32.10% | 0.5131 |
| Naive (t-1) | 10,048.4 | 13,571.5 | 50.23% | 0.0261 |
| Rolling Mean (7) | 10,022.1 | 13,283.5 | 50.10% | 0.0670 |
| Seasonal Naive (t-7) | 6,755.9 | 10,660.8 | 33.77% | 0.3991 |

### Prediction Interval Calibration (Target: 80%)
| Method | Coverage (%) | Avg Width | Low-Demand Cov (%) | High-Demand Cov (%) |
| :--- | :---: | :---: | :---: | :---: |
| gaussian | 80.00% | 19,326.9 | 84.29% | 75.71% |
| global_conformal | 72.86% | 14,899.4 | 74.29% | 71.43% |
| horizon_conformal | 71.43% | 16,092.9 | 72.86% | 70.00% |

## Weekly (4-Week Horizon)
- **Origins**: 8 (`2011-05-01` to `2011-11-13`)

### Model Comparison
| Model | MAE | RMSE | WAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 19,147.9 | 27,096.0 | 15.88% | 0.5549 |
| Naive (t-1) | 24,421.0 | 31,037.7 | 20.25% | 0.4160 |
| Rolling Mean (4) | 20,616.1 | 28,768.9 | 17.10% | 0.4982 |
| Seasonal Naive (t-4) | 23,524.8 | 30,000.6 | 19.51% | 0.4543 |
| Seasonal Naive (t-52) | 26,658.1 | 37,354.1 | 22.11% | 0.1541 |

### Prediction Interval Calibration (Target: 80%)
| Method | Coverage (%) | Avg Width | Low-Demand Cov (%) | High-Demand Cov (%) |
| :--- | :---: | :---: | :---: | :---: |
| gaussian | 81.25% | 57,500.2 | 93.75% | 68.75% |
| global_conformal | 87.50% | 58,098.3 | 93.75% | 81.25% |
| horizon_conformal | 78.12% | 47,173.7 | 87.50% | 68.75% |

