#!/usr/bin/env python3
"""
Phase 2: Rolling-Origin Backtesting & Interval Calibration Runner
=================================================================
Evaluates Daily (14-day horizon) and Weekly (4-week horizon) forecasting
on the 2-year Online Retail II dataset (`backend/data/dataset_3_clean.csv`).
Saves results to `backend/reports/phase2_backtest/` without touching production models.
"""

import os
import sys
import json
import time
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.evaluation.backtesting import run_rolling_origin_backtest

DATA_PATH = os.path.join(BACKEND_DIR, "data", "dataset_3_clean.csv")
OUT_DIR = os.path.join(BACKEND_DIR, "reports", "phase2_backtest")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 80)
    print("  PHASE 2: ROLLING-ORIGIN BACKTESTING & INTERVAL CALIBRATION")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, usecols=["date", "quantity"], low_memory=False)
    print(f"Loaded {len(df):,} records from {DATA_PATH}")

    print("\n[1/2] Running Daily Rolling-Origin Backtest (10 origins, 14-day horizon)...")
    daily_res = run_rolling_origin_backtest(
        df, granularity="D", horizon=14, n_origins=10, step_size=14, cal_window=28
    )

    print("\n[2/2] Running Weekly Rolling-Origin Backtest (8 origins, 4-week horizon)...")
    weekly_res = run_rolling_origin_backtest(
        df, granularity="W", horizon=4, n_origins=8, step_size=4, cal_window=8
    )

    out_json = os.path.join(OUT_DIR, "backtest_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"daily": daily_res, "weekly": weekly_res}, f, indent=2)

    # Generate Markdown Report
    md_path = os.path.join(OUT_DIR, "BACKTEST_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2: Rolling-Origin Backtesting & Prediction Interval Calibration Report\n\n")
        f.write(f"**Dataset**: `backend/data/dataset_3_clean.csv` ({len(df):,} records, `{daily_res['date_range']}`)\n\n")

        for label, res in [("Daily (14-Day Horizon)", daily_res), ("Weekly (4-Week Horizon)", weekly_res)]:
            f.write(f"## {label}\n")
            f.write(f"- **Origins**: {res['n_origins']} (`{res['origin_dates'][0]}` to `{res['origin_dates'][-1]}`)\n\n")
            f.write("### Model Comparison\n")
            f.write("| Model | MAE | RMSE | WAPE (%) | R² |\n| :--- | :---: | :---: | :---: | :---: |\n")
            for m, met in res["overall_metrics"].items():
                f.write(f"| {m} | {met['MAE']:,.1f} | {met['RMSE']:,.1f} | {met['WAPE']:.2f}% | {met['R2']:.4f} |\n")

            f.write("\n### Prediction Interval Calibration (Target: 80%)\n")
            f.write("| Method | Coverage (%) | Avg Width | Low-Demand Cov (%) | High-Demand Cov (%) |\n| :--- | :---: | :---: | :---: | :---: |\n")
            for iv, iv_m in res["interval_summary"].items():
                f.write(f"| {iv} | {iv_m['coverage_pct']:.2f}% | {iv_m['avg_width']:,.1f} | {iv_m['low_demand_coverage_pct']:.2f}% | {iv_m['high_demand_coverage_pct']:.2f}% |\n")
            f.write("\n")

    print(f"\n[DONE] Phase 2 backtest complete in {round(time.time() - t0, 2)}s!")
    print(f"Report saved to: {md_path}")
    print("\n--- DAILY OVERALL METRICS ---")
    for m, met in daily_res["overall_metrics"].items():
        print(f"  {m:<22}: WAPE={met['WAPE']:>6.2f}% | MAE={met['MAE']:>8.1f} | R2={met['R2']:>7.4f}")
    print("--- DAILY INTERVALS (80% target) ---")
    for iv, iv_m in daily_res["interval_summary"].items():
        print(f"  {iv:<22}: Cov={iv_m['coverage_pct']:>6.2f}% | Width={iv_m['avg_width']:>8.1f} | Low={iv_m['low_demand_coverage_pct']:>5.1f}% | High={iv_m['high_demand_coverage_pct']:>5.1f}%")

    print("\n--- WEEKLY OVERALL METRICS ---")
    for m, met in weekly_res["overall_metrics"].items():
        print(f"  {m:<22}: WAPE={met['WAPE']:>6.2f}% | MAE={met['MAE']:>8.1f} | R2={met['R2']:>7.4f}")
    print("--- WEEKLY INTERVALS (80% target) ---")
    for iv, iv_m in weekly_res["interval_summary"].items():
        print(f"  {iv:<22}: Cov={iv_m['coverage_pct']:>6.2f}% | Width={iv_m['avg_width']:>8.1f} | Low={iv_m['low_demand_coverage_pct']:>5.1f}% | High={iv_m['high_demand_coverage_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
