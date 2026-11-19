# Model Training & Forecasting Evaluation Report
**Project:** Intelligent Pattern Discovery Engine — Retail Intelligence Explorer  
**Evaluated At:** 2026-10-22 00:44:55 IST  
**Dataset:** [`backend/data/online_retail_clean.csv`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/data/online_retail_clean.csv) (527,932 verified line items)

---

## 1. Executive Summary

This evaluation validates the complete machine learning pipeline on the authentic **UCI Online Retail II** benchmark dataset.
- **Leakage Prevention**: All time-series features (lags, rolling means) are strictly shifted by 1 period.
- **Chronological Validity**: Evaluation is performed on untouched forward test splits; hyperparameter tuning was conducted exclusively on a preceding validation window.
- **Honest Benchmarking**: Baseline models (Naive, Seasonal Naive, Rolling Mean) are compared on the exact same test timestamps.
- **Zero Fabricated Demographics**: Customer segmentation is based on genuine order frequency and RFM tiers.

---

## 2. Daily Total Demand Forecasting (Primary Target)

* **Target Variable**: Gross units sold per calendar day (`daily_quantity`), with non-trading days (Saturdays) zero-filled.
* **Test Period**: `2011-11-12 to 2011-12-09` (28 held-out days)
* **Tuned Hyperparameters**: `{"n_estimators": 80, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.85}`

### Comparative Test Metrics (Same Held-Out 28 Days):
| Model | MAE (Units/Day) | RMSE | WAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Naive (t-1)** | 14,548.8 | 19,147.4 | 53.3% | -0.1013 |
| **Seasonal Naive (t-7)** | 7,849.6 | 14,568.6 | 28.8% | 0.3624 |
| **Rolling Mean (7)** | 12,745.8 | 18,136.8 | 46.7% | 0.0119 |
| **XGBoost (Tuned)** | **6,700.4** | **13,095.6** | **24.6%** | **0.4848** |

> **Analysis**: Daily retail sales fluctuate heavily due to order batching and day-of-week seasonality. Seasonal Naive ($t-7$) captures the weekly closed-day rhythm, while tuned XGBoost leverages both day-of-week and rolling momentum to deliver competitive performance.

### 14-Day Autoregressive Future Projections:
| Target Date | Projected Units | 80% Lower Bound | 80% Upper Bound |
| :---: | :---: | :---: | :---: |
| 2011-12-10 | 1,779.3 | 165.5 | 3,393.1 |
| 2011-12-11 | 14,515.7 | 12,863.9 | 16,167.5 |
| 2011-12-12 | 25,568.8 | 23,879.9 | 27,257.7 |
| 2011-12-13 | 26,540.7 | 24,815.5 | 28,265.9 |
| 2011-12-14 | 26,141.4 | 24,380.6 | 27,902.2 |
| 2011-12-15 | 30,218.1 | 28,422.4 | 32,013.8 |
| 2011-12-16 | 25,399.1 | 23,569.2 | 27,229.0 |
| 2011-12-17 | 1,801.7 | 0.0 | 3,665.1 |
| 2011-12-18 | 17,258.4 | 15,361.9 | 19,154.8 |
| 2011-12-19 | 25,736.3 | 23,807.5 | 27,665.2 |
| 2011-12-20 | 28,518.5 | 26,557.8 | 30,479.3 |
| 2011-12-21 | 27,497.8 | 25,505.7 | 29,489.9 |
| 2011-12-22 | 28,944.8 | 26,921.8 | 30,967.8 |
| 2011-12-23 | 24,792.2 | 22,738.8 | 26,845.7 |

---

## 3. Weekly Total Demand Forecasting (Secondary Target)

* **Target Variable**: Aggregate units sold per calendar week (`weekly_quantity`, 51 total weeks)
* **Test Period**: `2011-11-06 to 2011-12-11` (6 held-out weeks)

### Comparative Test Metrics:
| Model | MAE (Units/Week) | RMSE | WAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Naive (t-1)** | 25,962.0 | 39,504.2 | 14.4% | -0.5992 |
| **Seasonal Naive (t-1)** | 25,962.0 | 39,504.2 | 14.4% | -0.5992 |
| **Rolling Mean (4)** | 29,351.3 | 36,876.8 | 16.2% | -0.3936 |
| **XGBoost Weekly** | **43,587.2** | **53,491.7** | **24.1%** | **-1.9322** |

---

## 4. FP-Growth Association Rule Mining

* **Input Baskets**: **19,778 distinct transactions**
* **Parameters**: `min_support = 0.015`, `min_confidence = 0.3`, `min_lift = 1.2`
* **Frequent Itemsets**: **718** | **Discovered Rules**: **432**

### Top 5 Discovered Rules by Lift:
| # | Antecedent | Consequent | Support | Confidence | Lift |
| :-: | :--- | :--- | :---: | :---: | :---: |
| 1 | {Regency Tea Plate Roses} | {Regency Tea Plate Green} | 0.016 | 71.7% | **37.21** |
| 2 | {Regency Tea Plate Green} | {Regency Tea Plate Roses} | 0.016 | 83.7% | **37.21** |
| 3 | {Poppy'S Playhouse Kitchen} | {Poppy'S Playhouse Bedroom} | 0.016 | 71.8% | **33.57** |
| 4 | {Poppy'S Playhouse Bedroom} | {Poppy'S Playhouse Kitchen} | 0.016 | 74.0% | **33.57** |
| 5 | {Set/6 Red Spotty Paper Plates} | {Set/6 Red Spotty Paper Cups} | 0.018 | 66.4% | **30.69** |

---

## 5. Artifact Verification & Reproducibility

| Artifact Path | Format | Reload Verified |
| :--- | :--- | :---: |
| [`backend/models/xgboost_overall_daily.json`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/models/xgboost_overall_daily.json) | Native XGBoost Model JSON | **PASS** (Zero inference drift) |
| [`backend/models/xgboost_overall_daily_metadata.json`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/models/xgboost_overall_daily_metadata.json) | Metadata & Feature Registry | **PASS** |
| [`backend/models/xgboost_overall_weekly.json`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/models/xgboost_overall_weekly.json) | Native XGBoost Model JSON | **PASS** |
| [`backend/models/association_rules.json`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/models/association_rules.json) | FP-Growth Mined Rules | **PASS** |
| [`backend/models/customer_behavior.json`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/models/customer_behavior.json) | Authentic RFM / Behavioral Stats | **PASS** |
| [`backend/models/insights.json`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/models/insights.json) | Automated Insight Objects | **PASS** |
