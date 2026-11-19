# Phase 4: Model Validation, Statistical Reliability & Production Readiness Audit

## Daily (14-Day Horizon, 10 Origins)
- **HGB Wins vs XGBoost**: 9 / 10 origins
- **HGB Wins vs Seasonal Naive (t-7)**: 8 / 10 origins
- **HGB Per-Origin WAPE**: mean=32.39%, median=30.01%, std=10.71%
- **XGB Per-Origin WAPE**: mean=34.16%, median=31.2%, std=10.97%
- **Paired WAPE Diff (XGB - HGB)**: mean=1.77%, median=1.08%

### Horizon Breakdown
| Horizon | HGB WAPE | XGB WAPE | Gaussian Cov | Norm-Conformal Cov |
| :--- | :---: | :---: | :---: | :---: |
| Days 1-3 | 35.04% | 37.22% | 83.33% | 83.33% |
| Days 4-7 | 23.56% | 24.04% | 82.50% | 77.50% |
| Days 8-14 | 34.29% | 36.54% | 84.29% | 75.71% |

### Training-Window Robustness
| Window | HGB WAPE | XGB WAPE |
| :--- | :---: | :---: |
| expanding | 30.51% | 32.10% |
| 12_month | 32.11% | 32.84% |
| 18_month | 34.45% | 31.54% |

## Weekly (4-Week Horizon, 8 Origins)
- **HGB Wins vs XGBoost**: 5 / 8 origins
- **HGB Wins vs Rolling Mean (4)**: 5 / 8 origins
- **HGB Per-Origin WAPE**: mean=14.87%, median=15.64%, std=4.4%
- **XGB Per-Origin WAPE**: mean=15.79%, median=15.27%, std=4.61%
- **Paired WAPE Diff (XGB - HGB)**: mean=0.92%, median=0.59%

### Horizon Breakdown
| Horizon | HGB WAPE | XGB WAPE | Gaussian Cov | Norm-Conformal Cov |
| :--- | :---: | :---: | :---: | :---: |
| Week 1 | 12.90% | 19.73% | 75.00% | 87.50% |
| Week 2 | 13.29% | 11.66% | 75.00% | 62.50% |
| Weeks 3-4 | 17.57% | 16.01% | 62.50% | 56.25% |

### Training-Window Robustness
| Window | HGB WAPE | XGB WAPE |
| :--- | :---: | :---: |
| expanding | 15.44% | 15.88% |
| 12_month | 25.31% | 20.04% |
| 18_month | 15.08% | 15.30% |

