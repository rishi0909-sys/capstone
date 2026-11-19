#!/usr/bin/env python3
"""
Dataset 4 Full ML Training & Verification Pipeline
==================================================
Source: docs/dataset4.zip -> /tmp/dataset4_extracted/ecommerce_sales_analytics_5000.csv
Schema: 5,000 daily e-commerce orders with discounts, delivery days, ratings, and 4 categories.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.preprocessing.pipeline import PreprocessingPipeline
from src.mining.fp_growth import FPGrowthMiner
from src.demographics.analysis import DemographicAnalyzer
from src.forecasting.xgboost_model import XGBoostForecaster
from src.evaluation.metrics import calculate_forecast_metrics
from src.insights.engine import InsightEngine

RAW_CSV_PATH = "/tmp/dataset4_extracted/ecommerce_sales_analytics_5000.csv"
CLEAN_CSV_PATH = os.path.join(BACKEND_DIR, "data", "dataset_4_clean.csv")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)


def main():
    print("=" * 80)
    print("  TRAINING PIPELINE: DATASET 4 (E-Commerce Sales Analytics 5000)")
    print("=" * 80)
    t_start = time.time()

    # 1. Load & Normalize
    print(f"\n[1/6] Loading and normalizing {RAW_CSV_PATH}...")
    raw_df = pd.read_csv(RAW_CSV_PATH)
    df = pd.DataFrame({
        "transaction_id": raw_df["order_id"].astype(str),
        "customer_id": "CUST_" + raw_df["customer_id"].astype(str),
        "date": pd.to_datetime(raw_df["order_date"], format="mixed").dt.strftime("%Y-%m-%d 00:00:00"),
        "product_name": raw_df["product_category"].astype(str),
        "category": raw_df["product_category"].astype(str),
        "price": raw_df["unit_price"].astype(float),
        "quantity": raw_df["quantity"].astype(int),
        "revenue": raw_df["revenue"].astype(float),
        "country": raw_df["region"].astype(str),
    })
    df.to_csv(CLEAN_CSV_PATH, index=False)

    total_records = len(df)
    unique_invoices = int(df["transaction_id"].nunique())
    unique_customers = int(df["customer_id"].nunique())
    date_min = str(df["date"].min())
    date_max = str(df["date"].max())

    print(f"  --> Clean Records Saved : {total_records:,} -> {CLEAN_CSV_PATH}")
    print(f"  --> Unique Orders       : {unique_invoices:,}")
    print(f"  --> Unique Customers    : {unique_customers:,}")
    print(f"  --> Date Range          : {date_min} to {date_max}")

    # 2. Daily Demand Forecasting
    print("\n[2/6] Training Daily XGBoost Forecaster (28-day horizon with tuning)...")
    daily_fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    d_ok, d_msg = daily_fc.train_and_evaluate(
        granularity="D",
        test_horizon=28,
        val_horizon=28,
        tune=True,
    )
    assert d_ok, f"Daily training failed: {d_msg}"

    daily_test_metrics = calculate_forecast_metrics(
        daily_fc.results["test_actuals"],
        daily_fc.results["test_preds"],
    )
    daily_future = daily_fc.forecast_future(horizon=14, freq="D")

    print(f"  --> Best Tuned Parameters: {daily_fc.best_params}")
    print("  --> Daily Test Metrics  :", daily_test_metrics)
    for b_name, b_met in daily_fc.baseline_results.items():
        if b_met:
            print(f"      {b_name:<22}: MAE={b_met['MAE']:>8.2f} | RMSE={b_met['RMSE']:>8.2f} | WAPE={b_met['WAPE']:>5.2f}% | R2={b_met['R2']:>7.4f}")
    print(f"      {'XGBoost (Tuned)':<22}: MAE={daily_test_metrics['MAE']:>8.2f} | RMSE={daily_test_metrics['RMSE']:>8.2f} | WAPE={daily_test_metrics['WAPE']:>5.2f}% | R2={daily_test_metrics['R2']:>7.4f}")

    daily_model_path = os.path.join(MODELS_DIR, "xgboost_overall_daily.json")
    daily_fc.save_model_artifact(daily_model_path)

    # 3. Weekly Demand Forecasting
    print("\n[3/6] Training Weekly XGBoost Forecaster (6-week horizon with tuning)...")
    weekly_fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    w_ok, w_msg = weekly_fc.train_and_evaluate(
        granularity="W",
        test_horizon=6,
        val_horizon=6,
        tune=True,
    )
    assert w_ok, f"Weekly training failed: {w_msg}"

    weekly_test_metrics = calculate_forecast_metrics(
        weekly_fc.results["test_actuals"],
        weekly_fc.results["test_preds"],
    )
    weekly_future = weekly_fc.forecast_future(horizon=4, freq="W")

    print(f"  --> Best Tuned Parameters: {weekly_fc.best_params}")
    print("  --> Weekly Test Metrics :", weekly_test_metrics)

    weekly_model_path = os.path.join(MODELS_DIR, "xgboost_overall_weekly.json")
    weekly_fc.save_model_artifact(weekly_model_path)

    # 4. FP-Growth Market Basket Association Mining (grouped by customer_id for cross-category affinity)
    print("\n[4/6] Mining Customer-Level Category Association Rules with FP-Growth...")
    basket_input = df.copy()
    basket_input["transaction_id"] = basket_input["customer_id"]
    pipeline = PreprocessingPipeline(basket_input)
    basket_df = pipeline.get_basket_df()

    miner = FPGrowthMiner(basket_df)
    min_sup = 0.05
    min_conf = 0.30
    min_lift = 1.00

    freq_items, rules = miner.mine_patterns(
        min_support=min_sup,
        min_confidence=min_conf,
        min_lift=min_lift,
    )
    print(f"  --> Frequent itemsets: {len(freq_items):,}")
    print(f"  --> Association rules: {len(rules):,}")

    top_rules = rules.sort_values(by="lift", ascending=False) if not rules.empty else rules
    for idx, (_, r) in enumerate(top_rules.head(5).iterrows(), 1):
        ant = " + ".join(r["antecedents"])
        con = " + ".join(r["consequents"])
        print(f"    {idx}. {{{ant}}} ==> {{{con}}} | Support: {r['support']:.3f} | Conf: {r['confidence']:.1%} | Lift: {r['lift']:.2f}")

    rules_path = os.path.join(MODELS_DIR, "association_rules.json")
    rules.to_json(rules_path, orient="records", indent=2)

    # 5. Customer Behavioral & RFM Segmentation
    print("\n[5/6] Customer Behavioral Analytics & RFM Segmentation...")
    demo_analyzer = DemographicAnalyzer(df)
    customer_summary = demo_analyzer.get_summary()
    category_prefs = demo_analyzer.analyze_category_preferences()

    print(f"  --> Total Analyzed Shoppers: {customer_summary.get('total_unique_customers', unique_customers):,}")
    print(f"  --> Repeat Customers: {customer_summary.get('repeat_customers'):,}")
    print(f"  --> Repeat Purchase Rate: {customer_summary.get('repeat_purchase_rate_percent')}%")
    print(f"  --> Average Basket Units: {customer_summary.get('average_units_per_basket')}")
    print(f"  --> Segment Breakdown: {customer_summary.get('segments')}")

    behavior_path = os.path.join(MODELS_DIR, "customer_behavior.json")
    with open(behavior_path, "w", encoding="utf-8") as f_cb:
        json.dump({
            "summary": customer_summary,
            "preferences": category_prefs.to_dict(orient="records") if category_prefs is not None else [],
        }, f_cb, indent=2, default=str)

    engine = InsightEngine()
    if not rules.empty:
        engine.generate_association_insights(rules, top_n=5)
    if category_prefs is not None:
        engine.generate_demographic_insights(category_prefs, threshold=0.25)
    insights = engine.get_insights()

    insights_path = os.path.join(MODELS_DIR, "insights.json")
    with open(insights_path, "w", encoding="utf-8") as f_ins:
        json.dump(insights, f_ins, indent=2, default=str)
    print(f"  --> Saved {len(insights)} automated business insights")

    # 6. Artifact Verification
    print("\n[6/6] Verifying Artifact Reload & Numerical Inference Consistency...")
    reload_daily_fc = XGBoostForecaster()
    reload_daily_fc.load_model_artifact(daily_model_path)
    reloaded_future = reload_daily_fc.forecast_future(horizon=14, freq="D")
    max_diff = float(np.max(np.abs(np.array(daily_future["future_preds"]) - np.array(reloaded_future["future_preds"]))))
    assert max_diff < 1e-4, f"Daily reload verification failed! Max discrepancy: {max_diff}"
    print(f"  [PASS] Daily Forecaster reload verified. Discrepancy: {max_diff:.8f}")

    reload_weekly_fc = XGBoostForecaster()
    reload_weekly_fc.load_model_artifact(weekly_model_path)
    reloaded_weekly_future = reload_weekly_fc.forecast_future(horizon=4, freq="W")
    w_max_diff = float(np.max(np.abs(np.array(weekly_future["future_preds"]) - np.array(reloaded_weekly_future["future_preds"]))))
    assert w_max_diff < 1e-4, f"Weekly reload verification failed! Max discrepancy: {w_max_diff}"
    print(f"  [PASS] Weekly Forecaster reload verified. Discrepancy: {w_max_diff:.8f}")

    top_rule_dict = top_rules.iloc[0].to_dict() if len(top_rules) > 0 else {}
    results = {
        "dataset_name": "dataset_4 (E-Commerce Sales Analytics 5000)",
        "total_records": total_records,
        "unique_invoices": unique_invoices,
        "unique_customers": unique_customers,
        "date_min": date_min,
        "date_max": date_max,
        "daily_best_params": daily_fc.best_params,
        "daily_test_metrics": daily_test_metrics,
        "daily_baselines": daily_fc.baseline_results,
        "weekly_best_params": weekly_fc.best_params,
        "weekly_test_metrics": weekly_test_metrics,
        "weekly_baselines": weekly_fc.baseline_results,
        "rules_count": len(rules),
        "frequent_itemsets_count": len(freq_items),
        "top_rule": top_rule_dict,
        "customer_summary": customer_summary,
        "insights_count": len(insights),
        "duration_seconds": round(time.time() - t_start, 2),
    }

    with open("/tmp/d4_results.json", "w", encoding="utf-8") as f_res:
        json.dump(results, f_res, indent=2, default=str)
    print(f"\n[DONE] Full Dataset 4 training complete in {results['duration_seconds']}s!")


if __name__ == "__main__":
    main()
