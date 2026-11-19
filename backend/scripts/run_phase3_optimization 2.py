#!/usr/bin/env python3
"""
Phase 3: Forecasting Model Optimization & Robustness Runner
===========================================================
Saves all outputs under `backend/reports/phase3_optimization/`.
Does not modify production models.
"""

import os
import sys
import json
import time
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.evaluation.phase3_experiments import run_phase3_suite

DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_3_clean.csv")
OUT_DIR = os.path.join(BACKEND_DIR, "reports", "phase3_optimization")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 80)
    print("  PHASE 3: MODEL OPTIMIZATION, ROBUSTNESS & DEMAND-DEPENDENT INTERVALS")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, usecols=["date", "quantity"], low_memory=False)
    config = {
        "dataset": "backend/data/dataset_3_clean.csv",
        "daily": {"horizon": 14, "n_origins": 10, "step_size": 14, "cal_window": 28},
        "weekly": {"horizon": 4, "n_origins": 8, "step_size": 4, "cal_window": 8},
        "random_seed": 42,
    }
    with open(os.path.join(OUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n[1/2] Running Daily Phase 3 Suite...")
    daily_res = run_phase3_suite(df, **config["daily"], granularity="D")

    print("\n[2/2] Running Weekly Phase 3 Suite...")
    weekly_res = run_phase3_suite(df, **config["weekly"], granularity="W")

    with open(os.path.join(OUT_DIR, "phase3_results.json"), "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res, "weekly": weekly_res}, f, indent=2)

    md_path = os.path.join(OUT_DIR, "PHASE3_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3: Forecasting Model Optimization & Robustness Report\n\n")
        for label, res in [("Daily (14-Day Horizon, 10 Origins)", daily_res), ("Weekly (4-Week Horizon, 8 Origins)", weekly_res)]:
            f.write(f"## {label}\n\n### Model Comparison\n")
            f.write("| Model | MAE | RMSE | WAPE (%) | R² |\n| :--- | :---: | :---: | :---: | :---: |\n")
            for m, met in res["overall_models"].items():
                f.write(f"| {m} | {met['MAE']:,.1f} | {met['RMSE']:,.1f} | {met['WAPE']:.2f}% | {met['R2']:.4f} |\n")
            f.write("\n### Demand-Dependent Prediction Intervals (Target: 80%)\n")
            f.write("| Method | Coverage (%) | Avg Width | Low-Demand Cov (%) | High-Demand Cov (%) |\n| :--- | :---: | :---: | :---: | :---: |\n")
            for iv, im in res["intervals"].items():
                f.write(f"| {iv} | {im['coverage_pct']:.2f}% | {im['avg_width']:,.1f} | {im['low_demand_cov_pct']:.2f}% | {im['high_demand_cov_pct']:.2f}% |\n")
            f.write("\n### Top 5 Features (Permutation Importance)\n")
            for feat, imp in list(res["permutation_importance"].items())[:5]:
                f.write(f"- `{feat}`: {imp:.4f}\n")
            f.write("\n")

    print(f"\n[DONE] Phase 3 complete in {round(time.time() - t0, 2)}s!")
    print("\n--- DAILY MODEL COMPARISON ---")
    for m, met in daily_res["overall_models"].items():
        print(f"  {m:<22}: WAPE={met['WAPE']:>6.2f}% | MAE={met['MAE']:>8.1f} | R2={met['R2']:>7.4f}")
    print("--- DAILY INTERVALS ---")
    for iv, im in daily_res["intervals"].items():
        print(f"  {iv:<22}: Cov={im['coverage_pct']:>6.2f}% | Width={im['avg_width']:>8.1f} | Low={im['low_demand_cov_pct']:>5.1f}% | High={im['high_demand_cov_pct']:>5.1f}%")

    print("\n--- WEEKLY MODEL COMPARISON ---")
    for m, met in weekly_res["overall_models"].items():
        print(f"  {m:<22}: WAPE={met['WAPE']:>6.2f}% | MAE={met['MAE']:>8.1f} | R2={met['R2']:>7.4f}")
    print("--- WEEKLY INTERVALS ---")
    for iv, im in weekly_res["intervals"].items():
        print(f"  {iv:<22}: Cov={im['coverage_pct']:>6.2f}% | Width={im['avg_width']:>8.1f} | Low={im['low_demand_cov_pct']:>5.1f}% | High={im['high_demand_cov_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
