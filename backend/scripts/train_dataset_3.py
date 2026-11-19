#!/usr/bin/env python3
"""
Dataset 3 Full ML Training & Verification Pipeline
==================================================
Trains the Intelligent Pattern Discovery Engine on Dataset 3
(Full 2-Year UCI Online Retail II dataset, 2009-2011).
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

DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_3_clean.csv")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
REPORTS_DIR = os.path.join(BACKEND_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("  TRAINING PIPELINE: DATASET 3 (UCI Online Retail II 2-Year Benchmark)")
    print("=" * 80)
    t_start = time.time()

    print(f"\n[1/6] Loading Clean Dataset 3 from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    total_records = len(df)
    unique_invoices = int(df['transaction_id'].nunique())
    unique_customers = int(df['customer_id'].nunique())
    date_min = str(df['date'].min())
    date_max = str(df['date'].max())

    # Count registered vs guests
    guest_count = int(df['customer_id'].str.startswith('GUEST_').sum())
    unique_guests = int(df[df['customer_id'].str.startswith('GUEST_')]['customer_id'].nunique())
    unique_registered = int(df[~df['customer_id'].str.startswith('GUEST_')]['customer_id'].nunique())

    print(f"  --> Total Clean Records : {total_records:,}")
    print(f"  --> Unique Invoices     : {unique_invoices:,}")
    print(f"  --> Unique Customers    : {unique_customers:,} ({unique_registered:,} registered + {unique_guests:,} guest carts)")
    print(f"  --> Date Range          : {date_min} to {date_max}")

    # -------------------------------------------------------------
    # 2. Daily Demand Forecasting
    # -------------------------------------------------------------
    print("\n[2/6] Training Daily XGBoost Forecaster (28-day horizon with tuning)...")
    daily_fc = XGBoostForecaster(df, target_col='quantity', time_col='date')
    d_ok, d_msg = daily_fc.train_and_evaluate(
        granularity='D',
        test_horizon=28,
        val_horizon=28,
        tune=True
    )
    assert d_ok, f"Daily training failed: {d_msg}"

    daily_test_metrics = calculate_forecast_metrics(
        daily_fc.results['test_actuals'],
        daily_fc.results['test_preds']
    )
    daily_future = daily_fc.forecast_future(horizon=14, freq='D')

    print(f"  --> Best Tuned Parameters: {daily_fc.best_params}")
    print("  --> Daily Test Metrics  :", daily_test_metrics)
    print("  --> Daily Baseline Comparison:")
    for b_name, b_met in daily_fc.baseline_results.items():
        if b_met:
            print(f"      {b_name:<22}: MAE={b_met['MAE']:>8.1f} | RMSE={b_met['RMSE']:>8.1f} | WAPE={b_met['WAPE']:>5.2f}% | R2={b_met['R2']:>7.4f}")
    print(f"      {'XGBoost (Tuned)':<22}: MAE={daily_test_metrics['MAE']:>8.1f} | RMSE={daily_test_metrics['RMSE']:>8.1f} | WAPE={daily_test_metrics['WAPE']:>5.2f}% | R2={daily_test_metrics['R2']:>7.4f}")

    daily_model_path = os.path.join(MODELS_DIR, "xgboost_overall_daily.json")
    daily_fc.save_model_artifact(daily_model_path)
    print(f"  --> Saved daily model artifact: {daily_model_path}")

    # -------------------------------------------------------------
    # 3. Weekly Demand Forecasting
    # -------------------------------------------------------------
    print("\n[3/6] Training Weekly XGBoost Forecaster (6-week horizon with tuning)...")
    weekly_fc = XGBoostForecaster(df, target_col='quantity', time_col='date')
    w_ok, w_msg = weekly_fc.train_and_evaluate(
        granularity='W',
        test_horizon=6,
        val_horizon=6,
        tune=True
    )
    assert w_ok, f"Weekly training failed: {w_msg}"

    weekly_test_metrics = calculate_forecast_metrics(
        weekly_fc.results['test_actuals'],
        weekly_fc.results['test_preds']
    )
    weekly_future = weekly_fc.forecast_future(horizon=4, freq='W')

    print(f"  --> Best Tuned Parameters: {weekly_fc.best_params}")
    print("  --> Weekly Test Metrics :", weekly_test_metrics)
    print("  --> Weekly Baseline Comparison:")
    for b_name, b_met in weekly_fc.baseline_results.items():
        if b_met:
            print(f"      {b_name:<22}: MAE={b_met['MAE']:>8.1f} | RMSE={b_met['RMSE']:>8.1f} | WAPE={b_met['WAPE']:>5.2f}% | R2={b_met['R2']:>7.4f}")
    print(f"      {'XGBoost (Weekly)':<22}: MAE={weekly_test_metrics['MAE']:>8.1f} | RMSE={weekly_test_metrics['RMSE']:>8.1f} | WAPE={weekly_test_metrics['WAPE']:>5.2f}% | R2={weekly_test_metrics['R2']:>7.4f}")

    weekly_model_path = os.path.join(MODELS_DIR, "xgboost_overall_weekly.json")
    weekly_fc.save_model_artifact(weekly_model_path)
    print(f"  --> Saved weekly model artifact: {weekly_model_path}")

    # -------------------------------------------------------------
    # 4. FP-Growth Market Basket Association Mining
    # -------------------------------------------------------------
    print("\n[4/6] Mining Association Rules with FP-Growth...")
    pipeline = PreprocessingPipeline(df)
    basket_df = pipeline.get_basket_df()

    miner = FPGrowthMiner(basket_df)
    min_sup = 0.015
    min_conf = 0.30
    min_lift = 1.20

    t_m0 = time.time()
    freq_items, rules = miner.mine_patterns(
        min_support=min_sup,
        min_confidence=min_conf,
        min_lift=min_lift
    )
    t_mine = time.time() - t_m0
    print(f"  --> Mining finished in {t_mine:.2f}s")
    print(f"  --> Frequent itemsets: {len(freq_items):,}")
    print(f"  --> Association rules: {len(rules):,}")

    top_rules = rules.sort_values(by='lift', ascending=False)
    print("\n  Top 5 Association Rules by Lift:")
    for idx, (_, r) in enumerate(top_rules.head(5).iterrows(), 1):
        ant = " + ".join(r['antecedents'])
        con = " + ".join(r['consequents'])
        print(f"    {idx}. {{{ant}}} ==> {{{con}}}")
        print(f"       Support: {r['support']:.3f} | Confidence: {r['confidence']:.1%} | Lift: {r['lift']:.2f}")

    rules_path = os.path.join(MODELS_DIR, "association_rules.json")
    rules.to_json(rules_path, orient="records", indent=2)
    print(f"  --> Saved association rules to: {rules_path}")

    # -------------------------------------------------------------
    # 5. Customer Behavioral & RFM Segmentation
    # -------------------------------------------------------------
    print("\n[5/6] Customer Behavioral Analytics & Authentic RFM Segmentation...")
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
            "preferences": category_prefs.to_dict(orient="records") if category_prefs is not None else []
        }, f_cb, indent=2, default=str)
    print(f"  --> Saved customer behavior analytics to: {behavior_path}")

    # Generate Automated Insights
    engine = InsightEngine()
    if not rules.empty:
        engine.generate_association_insights(rules, top_n=5)
    if category_prefs is not None:
        engine.generate_demographic_insights(category_prefs, threshold=0.25)
    insights = engine.get_insights()

    insights_path = os.path.join(MODELS_DIR, "insights.json")
    with open(insights_path, "w", encoding="utf-8") as f_ins:
        json.dump(insights, f_ins, indent=2, default=str)
    print(f"  --> Saved {len(insights)} automated business insights to: {insights_path}")

    # -------------------------------------------------------------
    # 6. Artifact Verification (Reload & Inference Matching)
    # -------------------------------------------------------------
    print("\n[6/6] Verifying Artifact Reload & Numerical Inference Consistency...")
    reload_daily_fc = XGBoostForecaster()
    reload_daily_fc.load_model_artifact(daily_model_path)
    reloaded_future = reload_daily_fc.forecast_future(horizon=14, freq='D')

    diffs = np.abs(np.array(daily_future['future_preds']) - np.array(reloaded_future['future_preds']))
    max_diff = float(np.max(diffs))
    assert max_diff < 1e-4, f"Daily reload verification failed! Max discrepancy: {max_diff}"
    print(f"  [PASS] Daily Forecaster reload verified. Discrepancy: {max_diff:.8f}")

    reload_weekly_fc = XGBoostForecaster()
    reload_weekly_fc.load_model_artifact(weekly_model_path)
    reloaded_weekly_future = reload_weekly_fc.forecast_future(horizon=4, freq='W')
    w_diffs = np.abs(np.array(weekly_future['future_preds']) - np.array(reloaded_weekly_future['future_preds']))
    w_max_diff = float(np.max(w_diffs))
    assert w_max_diff < 1e-4, f"Weekly reload verification failed! Max discrepancy: {w_max_diff}"
    print(f"  [PASS] Weekly Forecaster reload verified. Discrepancy: {w_max_diff:.8f}")

    # Save summary results JSON for logging
    top_rule_dict = top_rules.iloc[0].to_dict() if len(top_rules) > 0 else {}
    results = {
        "dataset_name": "dataset3 (UCI Online Retail II 2-Year Dataset)",
        "total_records": total_records,
        "unique_invoices": unique_invoices,
        "unique_customers": unique_customers,
        "unique_registered": unique_registered,
        "unique_guests": unique_guests,
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
        "duration_seconds": round(time.time() - t_start, 2)
    }

    results_path = "/tmp/d3_results.json"
    with open(results_path, "w", encoding="utf-8") as f_res:
        json.dump(results, f_res, indent=2, default=str)
    print(f"\n[DONE] Full Dataset 3 training complete in {results['duration_seconds']}s!")
    print(f"Results recorded to {results_path}")

if __name__ == "__main__":
    main()
