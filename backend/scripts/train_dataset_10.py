#!/usr/bin/env python3
"""
Dataset 10 Full ML Training & Verification Pipeline
===================================================
Source: docs/dataset10.zip -> /tmp/dataset10_extracted/Dataset.csv
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

RAW_CSV_PATH = "/tmp/dataset10_extracted/Dataset.csv"
CLEAN_CSV_PATH = os.path.join(BACKEND_DIR, "data", "dataset_10_clean.csv")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)

CATEGORY_KEYWORDS = {
    "Seasonal & Holiday": ["christmas", "xmas", "tree", "easter", "halloween", "decoration", "santa", "snow", "advent", "holly", "reindeer", "noel"],
    "Home & Living": ["heart", "cushion", "clock", "frame", "mirror", "candle", "t-light", "holder", "doormat", "wall", "sign", "vase", "plaque", "light", "lights", "lamp", "curtain", "lantern", "rug", "hook"],
    "Kitchen & Dining": ["mug", "plate", "bowl", "cup", "saucer", "teapot", "cutlery", "glass", "bottle", "jar", "baking", "kitchen", "tray", "tin", "coffee", "tea", "cake", "spoon", "fork", "dish", "jug", "pan"],
    "Stationery & Crafts": ["pen", "pencil", "notebook", "card", "wrap", "sticker", "tape", "paper", "calendar", "box", "envelope", "stamp", "craft", "chalkboard", "rubber", "eraser", "ruler", "scissors"],
    "Fashion & Accessories": ["bag", "tote", "purse", "wallet", "scarf", "glove", "sock", "apron", "jewellery", "necklace", "bracelet", "charm", "earring", "pendant", "umbrella"],
    "Toys & Games": ["toy", "game", "puzzle", "doll", "plush", "teddy", "child", "kid", "mini", "ball", "play", "circus", "magic"],
}


def categorize_product(desc: str) -> str:
    d = str(desc).lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in d for kw in kws):
            return cat
    return "General Merchandise"


def main():
    print("=" * 80)
    print("  TRAINING PIPELINE: DATASET 10 (Dataset.csv)")
    print("=" * 80)
    t_start = time.time()

    print(f"\n[1/6] Loading and cleaning {RAW_CSV_PATH}...")
    raw_df = pd.read_csv(RAW_CSV_PATH, encoding="latin1", low_memory=False)
    raw_rows = len(raw_df)

    df = raw_df.rename(columns={
        "InvoiceNo": "transaction_id",
        "Description": "product_name",
        "InvoiceDate": "date",
        "CustomerID": "customer_id",
        "UnitPrice": "price",
        "Quantity": "quantity",
        "Country": "country",
    })
    df = df[~df["transaction_id"].astype(str).str.startswith("C")]
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]
    df = df.dropna(subset=["transaction_id", "product_name"])

    df["customer_id"] = df["customer_id"].apply(lambda x: f"CUST_{int(x)}" if pd.notna(x) else None)
    df["customer_id"] = df["customer_id"].fillna("GUEST_" + df["transaction_id"].astype(str))
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["product_name"] = df["product_name"].astype(str).str.strip().str.title()
    df["category"] = df["product_name"].apply(categorize_product)
    df["revenue"] = (df["price"] * df["quantity"]).round(2)

    df = df[["transaction_id", "customer_id", "date", "product_name", "category", "price", "quantity", "revenue", "country"]]
    df.to_csv(CLEAN_CSV_PATH, index=False)

    print(f"  --> Raw Rows: {raw_rows:,} | Clean Rows: {len(df):,} | Invoices: {df['transaction_id'].nunique():,} | Customers: {df['customer_id'].nunique():,}")

    print("\n[2/6] Training Daily XGBoost Forecaster...")
    daily_fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    d_ok, d_msg = daily_fc.train_and_evaluate(granularity="D", test_horizon=28, val_horizon=28, tune=True)
    assert d_ok, f"Daily training failed: {d_msg}"
    daily_test_metrics = calculate_forecast_metrics(daily_fc.results["test_actuals"], daily_fc.results["test_preds"])
    daily_future = daily_fc.forecast_future(horizon=14, freq="D")
    daily_model_path = os.path.join(MODELS_DIR, "xgboost_overall_daily.json")
    daily_fc.save_model_artifact(daily_model_path)
    print("  --> Daily Metrics:", daily_test_metrics)

    print("\n[3/6] Training Weekly XGBoost Forecaster...")
    weekly_fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
    w_ok, w_msg = weekly_fc.train_and_evaluate(granularity="W", test_horizon=6, val_horizon=6, tune=True)
    assert w_ok, f"Weekly training failed: {w_msg}"
    weekly_test_metrics = calculate_forecast_metrics(weekly_fc.results["test_actuals"], weekly_fc.results["test_preds"])
    weekly_future = weekly_fc.forecast_future(horizon=4, freq="W")
    weekly_model_path = os.path.join(MODELS_DIR, "xgboost_overall_weekly.json")
    weekly_fc.save_model_artifact(weekly_model_path)
    print("  --> Weekly Metrics:", weekly_test_metrics)

    print("\n[4/6] Running FP-Growth...")
    pipeline = PreprocessingPipeline(df)
    basket_df = pipeline.get_basket_df()
    miner = FPGrowthMiner(basket_df)
    freq_items, rules = miner.mine_patterns(min_support=0.015, min_confidence=0.30, min_lift=1.20)
    rules.to_json(os.path.join(MODELS_DIR, "association_rules.json"), orient="records", indent=2)
    print(f"  --> Frequent itemsets: {len(freq_items):,}, Rules: {len(rules):,}")

    print("\n[5/6] Analyzing Customer Behavior...")
    demo_analyzer = DemographicAnalyzer(df)
    customer_summary = demo_analyzer.get_summary()
    category_prefs = demo_analyzer.analyze_category_preferences()
    with open(os.path.join(MODELS_DIR, "customer_behavior.json"), "w", encoding="utf-8") as f_cb:
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
    with open(os.path.join(MODELS_DIR, "insights.json"), "w", encoding="utf-8") as f_ins:
        json.dump(insights, f_ins, indent=2, default=str)

    print("\n[6/6] Verifying Artifact Reload...")
    r_d = XGBoostForecaster()
    r_d.load_model_artifact(daily_model_path)
    assert float(np.max(np.abs(np.array(daily_future["future_preds"]) - np.array(r_d.forecast_future(horizon=14, freq="D")["future_preds"])))) < 1e-4
    r_w = XGBoostForecaster()
    r_w.load_model_artifact(weekly_model_path)
    assert float(np.max(np.abs(np.array(weekly_future["future_preds"]) - np.array(r_w.forecast_future(horizon=4, freq="W")["future_preds"])))) < 1e-4
    print("  [PASS] Daily & Weekly reload verified (0.00000000 discrepancy)")

    print(f"[DONE] Dataset 10 training complete in {round(time.time() - t_start, 2)}s!")


if __name__ == "__main__":
    main()
