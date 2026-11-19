#!/usr/bin/env python3
"""
Phase 7: Dataset Unification, Artifact Regeneration & Final Backend Freeze
==========================================================================
1. Establishes `backend/config/dataset_manifest.json` from `dataset_3_clean.csv`.
2. Regenerates `association_rules.json`, `customer_behavior.json`, `insights.json`
   on `dataset_3_clean.csv` so all 4 modules share the same 2-year dataset.
3. Benchmarks optimized HGB inference latency.
4. Writes all 7 deliverables under `backend/reports/phase7_finalization/`.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.preprocessing.pipeline import PreprocessingPipeline
from src.mining.fp_growth import FPGrowthMiner
from src.demographics.analysis import DemographicAnalyzer
from src.insights.engine import InsightEngine
from src.forecasting.hist_gradient_boosting_model import HistGradientBoostingForecaster

DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_3_clean.csv")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
PROD_DIR = os.path.join(MODELS_DIR, "production")
OUT_DIR = os.path.join(BACKEND_DIR, "reports", "phase7_finalization")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("=" * 80)
    print("  PHASE 7: DATASET UNIFICATION & FINAL BACKEND FREEZE")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    checksum = sha256_file(DATA_PATH)

    manifest = {
        "dataset_id": "dataset_3_clean.csv",
        "dataset_filename": "backend/data/dataset_3_clean.csv",
        "dataset_version": "2.0.0-uci-2yr",
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
        "row_count": int(len(df)),
        "unique_transaction_count": int(df["transaction_id"].nunique()),
        "unique_customer_count": int(df["customer_id"].nunique()),
        "unique_product_count": int(df["product_name"].nunique()),
        "cleaning_rules": [
            "Exclude cancellations (Invoice starting with C/A)",
            "Exclude non-positive quantity and price",
            "Exclude administrative stock codes (POST, BANK CHARGES, etc.)",
            "Retain guest checkouts via synthetic GUEST_<transaction_id>",
        ],
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sha256_checksum": checksum,
    }
    with open(os.path.join(CONFIG_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Regenerate FP-Growth, Customer Behavior, Insights on dataset_3_clean.csv
    print("\n[1/3] Regenerating FP-Growth rules, Customer Behavior, and Insights...")
    pipeline = PreprocessingPipeline(df)
    basket_df = pipeline.get_basket_df()
    miner = FPGrowthMiner(basket_df)
    freq_items, rules = miner.mine_patterns(min_support=0.015, min_confidence=0.30, min_lift=1.20)
    rules.to_json(os.path.join(MODELS_DIR, "association_rules.json"), orient="records", indent=2)

    demo = DemographicAnalyzer(df)
    cust_summary = demo.get_summary()
    cat_prefs = demo.analyze_category_preferences()
    with open(os.path.join(MODELS_DIR, "customer_behavior.json"), "w") as f:
        json.dump({
            "summary": cust_summary,
            "preferences": cat_prefs.to_dict(orient="records") if cat_prefs is not None else [],
        }, f, indent=2, default=str)

    engine = InsightEngine()
    if not rules.empty:
        engine.generate_association_insights(rules, top_n=5)
    if cat_prefs is not None:
        engine.generate_demographic_insights(cat_prefs, threshold=0.25)
    insights = engine.get_insights()
    with open(os.path.join(MODELS_DIR, "insights.json"), "w") as f:
        json.dump(insights, f, indent=2, default=str)

    regen = {
        "dataset_id": manifest["dataset_id"],
        "frequent_itemsets": len(freq_items),
        "association_rules": len(rules),
        "unique_customers": cust_summary.get("total_unique_customers"),
        "repeat_purchase_rate_percent": cust_summary.get("repeat_purchase_rate_percent"),
        "insights_count": len(insights),
    }
    with open(os.path.join(OUT_DIR, "regeneration_results.json"), "w") as f:
        json.dump(regen, f, indent=2)

    # Benchmark optimized HGB
    print("[2/3] Benchmarking optimized HGB inference...")
    hgb_d = HistGradientBoostingForecaster()
    hgb_d.load_model_artifact(os.path.join(PROD_DIR, "hgb_overall_daily.joblib"))
    t_inf = time.perf_counter()
    hgb_d.forecast_future(horizon=14, freq="D")
    opt_ms = round((time.perf_counter() - t_inf) * 1000.0, 2)

    perf_comp = {
        "hgb_daily_14d_before_ms": 260.22,
        "hgb_daily_14d_after_ms": opt_ms,
        "speedup_factor": round(260.22 / max(opt_ms, 0.01), 2),
    }
    with open(os.path.join(OUT_DIR, "performance_comparison.json"), "w") as f:
        json.dump(perf_comp, f, indent=2)

    # Artifact validation
    with open(os.path.join(PROD_DIR, "hgb_overall_daily_metadata.json")) as f:
        d_meta = json.load(f)
    with open(os.path.join(PROD_DIR, "hgb_overall_weekly_metadata.json")) as f:
        w_meta = json.load(f)

    art_val = {
        "manifest_dataset_id": manifest["dataset_id"],
        "hgb_daily_dataset_id": d_meta.get("dataset_id"),
        "hgb_weekly_dataset_id": w_meta.get("dataset_id"),
        "all_modules_synchronized": (d_meta.get("dataset_id") == manifest["dataset_id"] and w_meta.get("dataset_id") == manifest["dataset_id"]),
    }
    with open(os.path.join(OUT_DIR, "artifact_validation.json"), "w") as f:
        json.dump(art_val, f, indent=2)

    # API Contract
    with open(os.path.join(OUT_DIR, "API_CONTRACT.md"), "w") as f:
        f.write("# Intelligent Pattern Discovery Engine — Frozen API Contract\n\n")
        f.write("- `GET /api/mining/rules?dataset_id=demo&limit=50&offset=0` → `List[Rule]`\n")
        f.write("- `POST /api/demographics/analyze` (`{\"dataset_id\": \"demo\"}`) → Behavioral & RFM summary + category preferences\n")
        f.write("- `POST /api/temporal/analyze` (`{\"dataset_id\": \"demo\"}`) → Temporal sales trends\n")
        f.write("- `POST /api/forecast/run` (`{\"dataset_id\": \"demo\", \"granularity\": \"D|W\", \"horizon\": 14, \"target_product\": \"Overall\"}`) → `ForecastResponse`\n")
        f.write("- `POST /api/insights/generate` (`{\"dataset_id\": \"demo\"}`) → Automated business insights\n")

    with open(os.path.join(OUT_DIR, "PHASE7_REPORT.md"), "w") as f:
        f.write("# Phase 7: Dataset Unification, Artifact Regeneration & Final Backend Freeze\n\n")
        f.write(f"- **Unified Dataset**: `{manifest['dataset_id']}` (`{manifest['row_count']:,}` rows, `{manifest['date_range']}`)\n")
        f.write(f"- **Regenerated Artifacts**: {regen['association_rules']} rules, {regen['unique_customers']:,} customers, {regen['insights_count']} insights\n")
        f.write(f"- **HGB 14-Day Inference Optimization**: `{perf_comp['hgb_daily_14d_before_ms']} ms` → `{perf_comp['hgb_daily_14d_after_ms']} ms` (**{perf_comp['speedup_factor']}x** speedup)\n")
        f.write(f"- **Synchronized**: `{art_val['all_modules_synchronized']}`\n")

    print(f"[DONE] Phase 7 complete in {round(time.time() - t0, 2)}s!")
    print("Regeneration:", regen)
    print("Performance :", perf_comp)


if __name__ == "__main__":
    main()
