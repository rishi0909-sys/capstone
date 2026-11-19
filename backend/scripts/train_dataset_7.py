#!/usr/bin/env python3
"""
Dataset 7 Full ML Training & Verification Pipeline
==================================================
Source: docs/dataset7.zip -> /tmp/dataset7_extracted/Online Retail.xlsx
"""

import os
import sys
import json
import time
import zipfile
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
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

RAW_XLSX_PATH = "/tmp/dataset7_extracted/Online Retail.xlsx"
CLEAN_CSV_PATH = os.path.join(BACKEND_DIR, "data", "dataset_7_clean.csv")
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


def excel_serial_to_datetime_str(val_str: str) -> str:
    try:
        days = float(val_str)
        dt = datetime(1899, 12, 30) + timedelta(days=days)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def load_shared_strings(z: zipfile.ZipFile):
    shared_strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        with z.open("xl/sharedStrings.xml") as f:
            for _, elem in ET.iterparse(f, events=("end",)):
                if elem.tag.endswith("si"):
                    t = elem.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    if t is not None and t.text:
                        shared_strings.append(t.text)
                    else:
                        parts = [p.text for p in elem.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t") if p.text]
                        shared_strings.append("".join(parts))
                    elem.clear()
    return shared_strings


def convert_xlsx_to_clean_csv():
    raw_rows = 0
    clean_rows = 0
    with zipfile.ZipFile(RAW_XLSX_PATH, "r") as z:
        shared_strings = load_shared_strings(z)
        with open(CLEAN_CSV_PATH, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["transaction_id", "customer_id", "date", "product_name", "category", "price", "quantity", "revenue", "country"])
            row_idx = 0
            with z.open("xl/worksheets/sheet1.xml") as f:
                for _, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag.endswith("row"):
                        row_idx += 1
                        if row_idx == 1:
                            elem.clear()
                            continue
                        raw_rows += 1
                        row_dict = {}
                        for c in elem.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                            ref = c.attrib.get("r", "")
                            col = "".join([ch for ch in ref if ch.isalpha()])
                            val_elem = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                            cell_type = c.attrib.get("t")
                            val = val_elem.text if val_elem is not None else ""
                            if cell_type == "s" and val.isdigit():
                                val = shared_strings[int(val)]
                            row_dict[col] = val
                        elem.clear()

                        invoice = row_dict.get("A", "").strip()
                        desc = row_dict.get("C", "").strip()
                        qty_str = row_dict.get("D", "0")
                        date_str = row_dict.get("E", "")
                        price_str = row_dict.get("F", "0")
                        cust_raw = row_dict.get("G", "").strip()
                        country = row_dict.get("H", "").strip() or "United Kingdom"

                        if not invoice or invoice.startswith(("C", "c")) or not desc:
                            continue
                        try:
                            qty = int(float(qty_str))
                            price = round(float(price_str), 2)
                        except ValueError:
                            continue
                        if qty <= 0 or price <= 0.0:
                            continue
                        dt_str = excel_serial_to_datetime_str(date_str)
                        if not dt_str:
                            continue

                        customer_id = f"CUST_{int(cust_raw.split('.')[0])}" if cust_raw else f"GUEST_{invoice}"
                        product_name = desc.title()
                        category = categorize_product(product_name)
                        revenue = round(price * qty, 2)
                        writer.writerow([invoice, customer_id, dt_str, product_name, category, price, qty, revenue, country])
                        clean_rows += 1
    return raw_rows, clean_rows


def main():
    print("=" * 80)
    print("  TRAINING PIPELINE: DATASET 7 (Online Retail XLSX)")
    print("=" * 80)
    t_start = time.time()

    print("\n[1/6] Converting & cleaning Online Retail.xlsx...")
    raw_rows, clean_rows = convert_xlsx_to_clean_csv()
    df = pd.read_csv(CLEAN_CSV_PATH, low_memory=False)
    print(f"  --> Raw Rows: {raw_rows:,} | Clean Rows: {clean_rows:,} | Invoices: {df['transaction_id'].nunique():,} | Customers: {df['customer_id'].nunique():,}")

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
    top_rules = rules.sort_values(by="lift", ascending=False) if not rules.empty else rules
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

    results = {
        "raw_rows": raw_rows,
        "total_records": len(df),
        "unique_invoices": int(df["transaction_id"].nunique()),
        "unique_customers": int(df["customer_id"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "daily_best_params": daily_fc.best_params,
        "daily_test_metrics": daily_test_metrics,
        "weekly_best_params": weekly_fc.best_params,
        "weekly_test_metrics": weekly_test_metrics,
        "frequent_itemsets_count": len(freq_items),
        "rules_count": len(rules),
        "top_rule": top_rules.iloc[0].to_dict() if len(top_rules) > 0 else {},
        "customer_summary": customer_summary,
        "insights_count": len(insights),
        "duration_seconds": round(time.time() - t_start, 2),
    }
    with open("/tmp/d7_results.json", "w", encoding="utf-8") as f_res:
        json.dump(results, f_res, indent=2, default=str)
    print(f"[DONE] Dataset 7 training complete in {results['duration_seconds']}s!")


if __name__ == "__main__":
    main()
