"""
Multi-Dataset Collective Intelligence & Entity Resolution Engine
================================================================
Processes multiple heterogeneous retail/e-commerce datasets, resolves
canonical product/category entities with provenance, computes global
coverage/frequency/growth statistics, and persists the unified knowledge base.
"""

import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

CANONICAL_CATEGORY_MAP = {
    "home": ("Home & Living", 0.95),
    "home & living": ("Home & Living", 1.0),
    "home decor & lighting": ("Home & Living", 0.95),
    "clothing": ("Fashion & Accessories", 0.92),
    "fashion & accessories": ("Fashion & Accessories", 1.0),
    "bags & personal items": ("Fashion & Accessories", 0.94),
    "kitchen & dining": ("Kitchen & Dining", 1.0),
    "kitchenware & dining": ("Kitchen & Dining", 0.98),
    "stationery & crafts": ("Stationery & Crafts", 1.0),
    "stationery & craft": ("Stationery & Crafts", 0.98),
    "seasonal & holiday": ("Seasonal & Holiday", 1.0),
    "toys & games": ("Toys & Games", 1.0),
    "toys & children": ("Toys & Games", 0.95),
    "general merchandise": ("General Merchandise", 1.0),
    "electronics": ("Electronics", 1.0),
    "beauty": ("Beauty & Personal Care", 0.96),
}


def resolve_canonical_category(raw_cat: str) -> tuple:
    key = str(raw_cat).strip().lower()
    if key in CANONICAL_CATEGORY_MAP:
        return CANONICAL_CATEGORY_MAP[key]
    return (str(raw_cat).strip().title() or "General Merchandise", 0.70)


class CollectiveIntelligenceEngine:
    def __init__(self, models_dir: str, data_dir: str):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.collective_dir = os.path.join(models_dir, "collective")
        os.makedirs(self.collective_dir, exist_ok=True)
        self.registry_path = os.path.join(self.collective_dir, "registry.json")
        self.report_path = os.path.join(self.collective_dir, "learning_report.json")
        self.snapshots_path = os.path.join(self.collective_dir, "snapshots.json")
        self.catalogue_path = os.path.join(self.collective_dir, "product_catalogue.json")

    def build_from_datasets(self, dataset_files: List[Dict[str, str]]) -> Dict[str, Any]:
        t0 = time.time()
        registry = []
        cat_stats: Dict[str, Dict[str, Any]] = {}
        prod_stats: Dict[str, Dict[str, Any]] = {}
        total_records = 0
        total_datasets = len(dataset_files)

        for ds_info in dataset_files:
            ds_id = ds_info["id"]
            ds_name = ds_info["name"]
            path = ds_info["path"]
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path, low_memory=False)
            n_rows = len(df)
            total_records += n_rows

            date_min = str(df["date"].min())[:10] if "date" in df.columns else "N/A"
            date_max = str(df["date"].max())[:10] if "date" in df.columns else "N/A"
            n_cust = int(df["customer_id"].nunique()) if "customer_id" in df.columns else 0
            n_tx = int(df["transaction_id"].nunique()) if "transaction_id" in df.columns else n_rows

            registry.append({
                "dataset_id": ds_id,
                "dataset_name": ds_name,
                "filename": os.path.basename(path),
                "records": n_rows,
                "unique_transactions": n_tx,
                "unique_customers": n_cust,
                "date_range": f"{date_min} to {date_max}",
                "columns": list(df.columns),
                "status": "Processed",
            })

            # Canonical Category Aggregation
            if "category" in df.columns:
                df["_canon_cat"] = df["category"].map(lambda c: resolve_canonical_category(c)[0])
                vc = df["_canon_cat"].value_counts()
                for c_name, cnt in vc.items():
                    entry = cat_stats.setdefault(c_name, {
                        "canonical_category": c_name,
                        "total_occurrences": 0,
                        "datasets": set(),
                        "by_dataset": {},
                    })
                    entry["total_occurrences"] += int(cnt)
                    entry["datasets"].add(ds_id)
                    entry["by_dataset"][ds_id] = int(cnt)

            # Complete Canonical Product Aggregation (no top-50 cap)
            if "product_name" in df.columns:
                valid_p = df[df["product_name"].notna()].copy()
                valid_p["_raw_p"] = valid_p["product_name"].astype(str).str.strip()
                valid_p = valid_p[valid_p["_raw_p"] != ""]
                valid_p["_canon_p"] = valid_p["_raw_p"].str.title()
                if "category" in valid_p.columns:
                    valid_p["_canon_c"] = valid_p["category"].map(lambda c: resolve_canonical_category(c)[0])
                else:
                    valid_p["_canon_c"] = "General Merchandise"

                grp_p = valid_p.groupby("_canon_p").agg(
                    cnt=("_canon_p", "size"),
                    cat=("_canon_c", "first"),
                    raw_names=("_raw_p", lambda s: list(s.unique()[:5])),
                )
                for p_name, row in grp_p.iterrows():
                    p_entry = prod_stats.setdefault(p_name, {
                        "canonical_product": p_name,
                        "canonical_category": row["cat"],
                        "original_names": set(),
                        "total_occurrences": 0,
                        "datasets": set(),
                        "by_dataset": {},
                    })
                    p_entry["total_occurrences"] += int(row["cnt"])
                    p_entry["datasets"].add(ds_id)
                    p_entry["by_dataset"][ds_id] = int(row["cnt"])
                    p_entry["original_names"].update(row["raw_names"])

        # Format categories
        categories_list = []
        for c_name, st in cat_stats.items():
            cov = len(st["datasets"])
            tot = st["total_occurrences"]
            contribs = [
                {
                    "dataset_id": d_id,
                    "occurrences": d_cnt,
                    "contribution_pct": round((d_cnt / max(tot, 1)) * 100.0, 2),
                }
                for d_id, d_cnt in sorted(st["by_dataset"].items(), key=lambda x: x[1], reverse=True)
            ]
            categories_list.append({
                "canonical_category": c_name,
                "total_occurrences": tot,
                "dataset_coverage": cov,
                "total_datasets": total_datasets,
                "coverage_pct": round((cov / max(total_datasets, 1)) * 100.0, 2),
                "share_pct": round((tot / max(total_records, 1)) * 100.0, 2),
                "contributing_datasets": sorted(list(st["datasets"])),
                "contributions": contribs,
                "status": "Established" if cov >= max(2, total_datasets // 2) else "Emerging",
            })
        categories_list.sort(key=lambda x: x["total_occurrences"], reverse=True)

        # Format top products
        products_list = []
        for p_name, st in prod_stats.items():
            cov = len(st["datasets"])
            tot = st["total_occurrences"]
            contribs = [
                {
                    "dataset_id": d_id,
                    "occurrences": d_cnt,
                    "contribution_pct": round((d_cnt / max(tot, 1)) * 100.0, 2),
                }
                for d_id, d_cnt in sorted(st["by_dataset"].items(), key=lambda x: x[1], reverse=True)
            ]
            products_list.append({
                "canonical_product": p_name,
                "canonical_category": st.get("canonical_category", "General Merchandise"),
                "original_names": sorted(list(st.get("original_names", []))),
                "total_occurrences": tot,
                "dataset_coverage": cov,
                "total_datasets": total_datasets,
                "coverage_pct": round((cov / max(total_datasets, 1)) * 100.0, 2),
                "contributing_datasets": sorted(list(st["datasets"])),
                "contributions": contribs,
            })
        products_list.sort(key=lambda x: (x["dataset_coverage"], x["total_occurrences"]), reverse=True)

        # Evidence-backed discoveries ("What the model learned")
        discoveries = []
        if categories_list:
            top_c = categories_list[0]
            discoveries.append({
                "title": f"Dominant Global Category: {top_c['canonical_category']}",
                "statement": f"{top_c['canonical_category']} appeared in {top_c['dataset_coverage']} out of {total_datasets} datasets ({top_c['coverage_pct']}%), accounting for {top_c['total_occurrences']:,} observed records ({top_c['share_pct']}% of total volume).",
                "contributing_datasets": top_c["contributing_datasets"],
                "method": "Canonical Entity Resolution & Cross-Dataset Aggregation",
                "confidence": "High",
            })
        if len(categories_list) > 1:
            emerging = [c for c in categories_list if c["status"] == "Emerging"]
            if emerging:
                em = emerging[0]
                discoveries.append({
                    "title": f"Emerging Niche Category: {em['canonical_category']}",
                    "statement": f"{em['canonical_category']} was discovered in {em['dataset_coverage']}/{total_datasets} datasets with {em['total_occurrences']:,} records, representing specialized e-commerce demand.",
                    "contributing_datasets": em["contributing_datasets"],
                    "method": "Cross-Schema Category Normalization",
                    "confidence": "Medium",
                })
        if products_list:
            tp = products_list[0]
            discoveries.append({
                "title": f"Most Consistent Product: {tp['canonical_product']}",
                "statement": f"{tp['canonical_product']} appeared across {tp['dataset_coverage']} out of {total_datasets} datasets ({tp['coverage_pct']}% coverage) with {tp['total_occurrences']:,} total occurrences.",
                "contributing_datasets": tp["contributing_datasets"],
                "method": "Canonical Product Frequency & Coverage Tracking",
                "confidence": "High",
            })

        report = {
            "overview": {
                "datasets_processed": len(registry),
                "total_records_ingested": total_records,
                "canonical_categories_count": len(categories_list),
                "canonical_products_tracked": len(products_list),
                "training_duration_sec": round(time.time() - t0, 2),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            },
            "registry": registry,
            "categories": categories_list,
            "top_products": products_list[:30],
            "discoveries": discoveries,
        }

        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(self.catalogue_path, "w", encoding="utf-8") as f:
            json.dump(products_list, f, indent=2)

        # Append versioned snapshot for Knowledge Evolution
        snapshots = self.load_snapshots()
        ver_id = f"v{len(snapshots) + 1}.0"
        snapshots.append({
            "version_id": ver_id,
            "timestamp": report["overview"]["generated_at"],
            "datasets_processed": len(registry),
            "dataset_ids": [r["dataset_id"] for r in registry],
            "total_records": total_records,
            "canonical_categories_count": len(categories_list),
            "canonical_products_tracked": len(products_list),
            "categories": {c["canonical_category"]: {"occurrences": c["total_occurrences"], "coverage": c["dataset_coverage"]} for c in categories_list},
        })
        with open(self.snapshots_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, indent=2)

        return report

    def load_snapshots(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.snapshots_path):
            with open(self.snapshots_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def load_report(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.report_path):
            with open(self.report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
