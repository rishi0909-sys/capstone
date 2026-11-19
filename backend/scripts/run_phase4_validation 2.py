#!/usr/bin/env python3
"""
Phase 4: Model Validation, Statistical Reliability & Production Readiness Runner
==============================================================================
Saves all 7 required artifacts under `backend/reports/phase4_validation/`.
"""

import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.evaluation.phase4_validation import run_phase4_validation

DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_3_clean.csv")
OUT_DIR = os.path.join(BACKEND_DIR, "reports", "phase4_validation")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 80)
    print("  PHASE 4: MODEL VALIDATION & PRODUCTION READINESS AUDIT")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, usecols=["date", "quantity"], low_memory=False)
    config = {
        "dataset": "backend/data/dataset_3_clean.csv",
        "daily": {"horizon": 14, "n_origins": 10, "step_size": 14, "cal_window": 28},
        "weekly": {"horizon": 4, "n_origins": 8, "step_size": 4, "cal_window": 8},
        "random_seed": 42,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with open(os.path.join(OUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n[1/2] Running Daily Phase 4 Validation...")
    daily_res = run_phase4_validation(df, **config["daily"], granularity="D")

    print("\n[2/2] Running Weekly Phase 4 Validation...")
    weekly_res = run_phase4_validation(df, **config["weekly"], granularity="W")

    with open(os.path.join(OUT_DIR, "phase4_results.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res, "weekly": weekly_res}, f, indent=2)

    with open(os.path.join(OUT_DIR, "per_origin_results.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res["per_origin"], "weekly": weekly_res["per_origin"]}, f, indent=2)

    with open(os.path.join(OUT_DIR, "horizon_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res["horizon_summary"], "weekly": weekly_res["horizon_summary"]}, f, indent=2)

    with open(os.path.join(OUT_DIR, "interval_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res["interval_summary"], "weekly": weekly_res["interval_summary"]}, f, indent=2)

    with open(os.path.join(OUT_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res["permutation_importance"], "weekly": weekly_res["permutation_importance"]}, f, indent=2)

    # Generate PHASE4_REPORT.md
    md_path = os.path.join(OUT_DIR, "PHASE4_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Model Validation, Statistical Reliability & Production Readiness Audit\n\n")
        for label, res in [("Daily (14-Day Horizon, 10 Origins)", daily_res), ("Weekly (4-Week Horizon, 8 Origins)", weekly_res)]:
            os_sum = res["origin_summary"]
            f.write(f"## {label}\n")
            f.write(f"- **HGB Wins vs XGBoost**: {os_sum['hgb_wins_vs_xgb']} / {os_sum['total_origins']} origins\n")
            f.write(f"- **HGB Wins vs {res['strongest_baseline']}**: {os_sum['hgb_wins_vs_baseline']} / {os_sum['total_origins']} origins\n")
            f.write(f"- **HGB Per-Origin WAPE**: mean={os_sum['hgb_wape_stats']['mean']}%, median={os_sum['hgb_wape_stats']['median']}%, std={os_sum['hgb_wape_stats']['std']}%\n")
            f.write(f"- **XGB Per-Origin WAPE**: mean={os_sum['xgb_wape_stats']['mean']}%, median={os_sum['xgb_wape_stats']['median']}%, std={os_sum['xgb_wape_stats']['std']}%\n")
            f.write(f"- **Paired WAPE Diff (XGB - HGB)**: mean={os_sum['paired_diff_xgb_minus_hgb']['mean']}%, median={os_sum['paired_diff_xgb_minus_hgb']['median']}%\n\n")

            f.write("### Horizon Breakdown\n| Horizon | HGB WAPE | XGB WAPE | Gaussian Cov | Norm-Conformal Cov |\n| :--- | :---: | :---: | :---: | :---: |\n")
            for hg, hm in res["horizon_summary"].items():
                f.write(f"| {hg} | {hm['HistGradientBoosting']['WAPE']:.2f}% | {hm['XGBoost']['WAPE']:.2f}% | {hm['gaussian_cov_pct']:.2f}% | {hm['normalized_conformal_cov_pct']:.2f}% |\n")

            f.write("\n### Training-Window Robustness\n| Window | HGB WAPE | XGB WAPE |\n| :--- | :---: | :---: |\n")
            for w_lbl, wm in res["window_robustness"].items():
                f.write(f"| {w_lbl} | {wm['HistGradientBoosting']['WAPE']:.2f}% | {wm['XGBoost']['WAPE']:.2f}% |\n")
            f.write("\n")

    print(f"\n[DONE] Phase 4 complete in {round(time.time() - t0, 2)}s!")
    for label, res in [("DAILY", daily_res), ("WEEKLY", weekly_res)]:
        os_sum = res["origin_summary"]
        print(f"\n--- {label} SUMMARY ---")
        print(f"  HGB wins vs XGB: {os_sum['hgb_wins_vs_xgb']}/{os_sum['total_origins']} | Paired WAPE diff (XGB-HGB): mean={os_sum['paired_diff_xgb_minus_hgb']['mean']}%, median={os_sum['paired_diff_xgb_minus_hgb']['median']}%")
        print(f"  HGB WAPE: mean={os_sum['hgb_wape_stats']['mean']}%, std={os_sum['hgb_wape_stats']['std']}% | XGB WAPE: mean={os_sum['xgb_wape_stats']['mean']}%, std={os_sum['xgb_wape_stats']['std']}%")


if __name__ == "__main__":
    main()
