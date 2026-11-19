# Dataset Audit & Forecasting Target Specification Report
**Project:** Intelligent Pattern Discovery Engine — Retail Intelligence Explorer  
**Source Dataset:** UCI Online Retail II (`online_retail_II.xlsx` — Year 2010–2011 Benchmark)  
**Clean Output File:** [`backend/data/online_retail_clean.csv`](file:///Users/rishabhmudgal/Documents/components/capstone/backend/data/online_retail_clean.csv)

---

## 1. Raw Dataset Audit & Exclusion Register

| Audit Dimension | Raw Count / Value | Treatment / Exclusion Decision |
| :--- | :--- | :--- |
| **Total Raw Rows (Sheet 2)** | **541,910 lines** | Base input for analysis. |
| **Cancelled Invoices (`C...`)** | **9,291 lines** | **EXCLUDED**: Cancellations represent reverse logistics/returns, not positive forward customer demand. |
| **Admin / Non-Product Codes** | **2,186 lines** | **EXCLUDED**: Stripped non-product items (`POST`, `D`, `M`, `BANK CHARGES`, `AMAZON FEE`). |
| **Non-Positive Quantity (<=0)** | **433 lines** | **EXCLUDED**: Returns, write-offs, and damaged inventory entries. |
| **Zero / Negative Price (<=0)** | **572 lines** | **EXCLUDED**: Bad debt adjustments, damaged stock liquidations, and gift adjustments. |
| **Missing Product Description** | **1,496 lines** | **EXCLUDED**: Unlabeled inventory tracking records. |
| **Unregistered Guest Checkout** | **135,080 lines** | **PRESERVED**: Retained for market-basket transaction mining and demand aggregation as `UNREGISTERED_GUEST`. |
| **Customer Demographics** | **NONE IN SOURCE** | **CONFIRMED ABSENT**: No customer age, gender, or income exists in raw retailer logs. **Zero fabricated demographics added.** |
| **Final Clean Retained Records**| **527,932 lines** | **100% verified transactional line items.** |

---

## 2. Cleaned Dataset Characteristics

* **Date Coverage**: `2010-12-01 08:26:00` to `2011-12-09 12:50:00` (374 calendar days, 51 continuous weeks)
* **Unique Baskets / Invoices**: **19,778 distinct transactions**
* **Unique Product Catalog Items**: **4,008 physical retail items**
* **Registered Customers**: **4,334 unique accounts** (+ unregistered guest sessions)
* **Total Gross Units Sold**: **5,577,277 items**
* **Total Gross Revenue**: **£10,279,232.92 GBP**
* **Line Item Nature**: Every row is a single product purchase line within an invoice, **not** an independent daily observation.

### Top Product Categories (Derived from Product Attributes):
- **Home Decor & Lighting**: 154,919 line items
- **Kitchenware & Dining**: 109,974 line items
- **General Merchandise**: 92,845 line items
- **Stationery & Craft**: 64,569 line items
- **Bags & Personal Items**: 48,326 line items
- **Seasonal & Holiday**: 43,604 line items

### Geographic Distribution:
- **United Kingdom**: 484,067 line items
- **Germany**: 8,658 line items
- **France**: 8,102 line items
- **EIRE**: 7,885 line items
- **Spain**: 2,422 line items

---

## 3. Explicit Forecasting Target Definition

| Target Dimension | Specification | Rationale & Defense |
| :--- | :--- | :--- |
| **Primary Target** | **Daily Total Units Sold (`daily_quantity`)** | Essential metric for retail supply chain and daily warehouse dispatch. |
| **Secondary Target** | **Weekly Total Units Sold (`weekly_quantity`)** | Macro inventory planning across 51 continuous calendar weeks. |
| **Tertiary Target** | **Daily Revenue (`revenue`) & Top Product Demand** | Financial planning and high-velocity SKU replenishment. |
| **Aggregation Procedure** | Resample `date` by calendar frequency (`D` or `W-SUN`) summing `quantity`. | Sums all confirmed item sales on each calendar date. |
| **Zero-Sales Handling** | Calendar re-indexing (`reindex(all_dates, fill_value=0)`). | The store is closed on Saturdays (0 orders). Filling zero-sales dates ensures a strictly continuous time series. |
| **Prediction Horizon** | **14 Days** (Daily) / **4 Weeks** (Weekly) | Practical operational procurement lead-time. |
| **Evaluation Metrics** | **MAE, RMSE, WAPE, R²** | WAPE is scale-aware and avoids zero-division errors of MAPE on low-volume days. |
