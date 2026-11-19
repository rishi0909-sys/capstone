"""
Hybrid Structured Retrieval & Association Deduplication Service
===============================================================
Provides narrowly scoped, read-only retrieval functions over persisted
collective intelligence, association rules, customer behavior, and predictive
artifacts. Never fabricates metrics or executes arbitrary queries.
"""

import os
import json
from typing import Dict, List, Any, Optional


class StructuredRetriever:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def _load_json(self, rel_path: str, default: Any = None) -> Any:
        path = os.path.join(self.models_dir, rel_path)
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search_products(self, query: str = "", category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        cat = self._load_json("collective/product_catalogue.json", [])
        q = query.strip().lower()
        res = []
        for p in cat:
            if q and q not in p.get("canonical_product", "").lower():
                continue
            if category and category != "All" and p.get("canonical_category", "").lower() != category.lower():
                continue
            res.append({
                "canonical_product": p["canonical_product"],
                "canonical_category": p.get("canonical_category"),
                "total_occurrences": p["total_occurrences"],
                "dataset_coverage": f"{p['dataset_coverage']}/{p['total_datasets']}",
            })
            if len(res) >= limit:
                break
        return res

    def get_product_relationships(self, product_query: str = "", min_lift: float = 1.2, limit: int = 10) -> List[Dict[str, Any]]:
        rules = self._load_json("association_rules.json", [])
        q = product_query.strip().lower()
        seen_pairs = {}
        for r in sorted(rules, key=lambda x: x.get("lift", 0), reverse=True):
            if float(r.get("lift", 0)) < min_lift:
                continue
            ant = r.get("antecedents", [])
            con = r.get("consequents", [])
            if q and not any(q in item.lower() for item in ant + con):
                continue
            pair_key = tuple(sorted([" + ".join(ant), " + ".join(con)]))
            if pair_key not in seen_pairs:
                seen_pairs[pair_key] = {
                    "antecedents": ant,
                    "consequents": con,
                    "support": round(float(r.get("support", 0)), 4),
                    "confidence": round(float(r.get("confidence", 0)), 4),
                    "lift": round(float(r.get("lift", 0)), 2),
                    "bidirectional": False,
                    "reverse_confidence": None,
                }
            else:
                seen_pairs[pair_key]["bidirectional"] = True
                seen_pairs[pair_key]["reverse_confidence"] = round(float(r.get("confidence", 0)), 4)
            if len(seen_pairs) >= limit:
                break
        return list(seen_pairs.values())

    def get_category_performance(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        rep = self._load_json("collective/learning_report.json", {})
        cats = rep.get("categories", [])
        if category and category != "All":
            cats = [c for c in cats if c["canonical_category"].lower() == category.lower()]
        return [
            {
                "canonical_category": c["canonical_category"],
                "total_occurrences": c["total_occurrences"],
                "dataset_coverage": f"{c['dataset_coverage']}/{c['total_datasets']}",
                "share_pct": c.get("share_pct"),
                "status": c.get("status"),
            }
            for c in cats
        ]

    def get_customer_segment_summary(self) -> Dict[str, Any]:
        cb = self._load_json("customer_behavior.json", {})
        return cb.get("summary", {})

    def get_emerging_and_anomalies(self) -> Dict[str, Any]:
        pred = self._load_json("collective/predictive_intelligence.json", {})
        return {
            "emerging_patterns": pred.get("emerging_patterns", [])[:7],
            "anomalies": pred.get("anomalies", [])[:10],
        }

    def build_workspace_insights(self) -> Dict[str, Any]:
        """Builds deduplicated, categorized insights with real supporting evidence."""
        rels = self.get_product_relationships(limit=15)
        cb = self.get_customer_segment_summary()
        pred = self.get_emerging_and_anomalies()

        cards = []
        for idx, r in enumerate(rels):
            ant_str = ", ".join(r["antecedents"])
            con_str = ", ".join(r["consequents"])
            cards.append({
                "id": f"assoc-{idx}",
                "category": "Product Associations",
                "impact": "HIGH" if r["lift"] >= 10.0 else "MEDIUM",
                "title": f"{ant_str} ↔ {con_str}" if r["bidirectional"] else f"{ant_str} → {con_str}",
                "summary": (
                    f"Customers buying {ant_str} frequently also buy {con_str} "
                    f"({r['confidence']*100:.1f}% follow-through, {r['lift']:.2f}x lift)."
                ),
                "metrics": {
                    "support_pct": round(r["support"] * 100, 2),
                    "confidence_pct": round(r["confidence"] * 100, 1),
                    "lift": r["lift"],
                    "bidirectional": r["bidirectional"],
                    "reverse_confidence_pct": round(r["reverse_confidence"] * 100, 1) if r["reverse_confidence"] else None,
                },
                "recommendation": f"Consider co-locating or bundling {ant_str} with {con_str}.",
                "provenance": "dataset_3_clean.csv (FP-Growth, 39,536 baskets)",
                "limitation": "Observed co-occurrence only; does not imply causal demand.",
            })

        for idx, ep in enumerate(pred["emerging_patterns"][:4]):
            cards.append({
                "id": f"emerge-{idx}",
                "category": "Emerging Trends",
                "impact": "HIGH" if ep["growth_rate_pct"] > 25 else "MEDIUM",
                "title": ep["pattern_name"],
                "summary": (
                    f"Recent 3-month average ({ep['recent_3m_avg']:,.0f}/mo) is "
                    f"{ep['growth_rate_pct']:+.1f}% vs historical baseline ({ep['historical_baseline_monthly']:,.0f}/mo)."
                ),
                "metrics": {
                    "growth_rate_pct": ep["growth_rate_pct"],
                    "recent_3m_avg": ep["recent_3m_avg"],
                    "baseline_monthly": ep["historical_baseline_monthly"],
                },
                "recommendation": f"Review inventory buffer for {ep['category']} ahead of peak demand.",
                "provenance": ", ".join(ep.get("supporting_datasets", ["dataset_3"])),
                "limitation": "Reflects historical 2-year seasonal window (2009–2011).",
            })

        for idx, an in enumerate(pred["anomalies"][:4]):
            cards.append({
                "id": f"anom-{idx}",
                "category": "Anomalies",
                "impact": "HIGH" if abs(an["robust_z_score"]) >= 3.5 else "MEDIUM",
                "title": f"{an['category']} {an['direction']} ({an['period']})",
                "summary": an["explanation"],
                "metrics": {
                    "observed_quantity": an["observed_quantity"],
                    "expected_median": an["expected_median"],
                    "robust_z_score": an["robust_z_score"],
                },
                "recommendation": "Verify whether this statistical deviation was driven by bulk wholesale orders or seasonal events.",
                "provenance": "dataset_3_clean.csv (Median/MAD Robust Z-Score)",
                "limitation": "Statistical anomaly only; requires domain verification.",
            })

        if cb:
            cards.append({
                "id": "cust-0",
                "category": "Customer Behavior",
                "impact": "HIGH",
                "title": f"{cb.get('repeat_purchase_rate_percent', 0)}% Repeat Purchase Rate",
                "summary": (
                    f"{cb.get('repeat_customers', 0):,} of {cb.get('total_unique_customers', 0):,} shoppers "
                    f"returned for multiple orders (avg {cb.get('average_units_per_basket', 0)} units/basket)."
                ),
                "metrics": {
                    "total_customers": cb.get("total_unique_customers", 0),
                    "repeat_customers": cb.get("repeat_customers", 0),
                    "repeat_rate_pct": cb.get("repeat_purchase_rate_percent", 0),
                },
                "recommendation": "Target One-Time Buyers with post-purchase retention campaigns.",
                "provenance": "dataset_3_clean.csv (RFM Behavioral Tiers)",
                "limitation": "Guest checkouts are grouped at the invoice level (GUEST_<tx_id>).",
            })

        return {
            "overview": {
                "total_insights": len(cards),
                "high_impact_count": sum(1 for c in cards if c["impact"] == "HIGH"),
                "associations_count": len(rels),
                "emerging_count": len(pred["emerging_patterns"]),
                "anomalies_count": len(pred["anomalies"]),
            },
            "insights": cards,
            "relationships": rels,
        }
