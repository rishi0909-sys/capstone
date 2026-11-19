import os
import sys
import json
import tempfile
import pandas as pd
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.collective.engine import CollectiveIntelligenceEngine, resolve_canonical_category
import main as backend_main


def test_category_normalization_and_fallback():
    assert resolve_canonical_category("Home") == ("Home & Living", 0.95)
    assert resolve_canonical_category("Clothing") == ("Fashion & Accessories", 0.92)
    assert resolve_canonical_category("Unknown Gadget") == ("Unknown Gadget", 0.70)


def test_collective_engine_synthetic_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        d1_path = os.path.join(tmpdir, "d1.csv")
        d2_path = os.path.join(tmpdir, "d2.csv")
        pd.DataFrame({
            "transaction_id": ["1", "2", "3"],
            "customer_id": ["C1", "C2", "C1"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "product_name": ["  travel bag ", "Travel Bag", None],
            "category": ["Home", "Clothing", "Home"],
        }).to_csv(d1_path, index=False)
        pd.DataFrame({
            "transaction_id": ["4", "5"],
            "customer_id": ["C3", "C4"],
            "date": ["2023-02-01", "2023-02-02"],
            "product_name": ["TRAVEL BAG", "Ceramic Mug"],
            "category": ["Home & Living", "Kitchen & Dining"],
        }).to_csv(d2_path, index=False)

        eng = CollectiveIntelligenceEngine(tmpdir, tmpdir)
        rep1 = eng.build_from_datasets([{"id": "d1", "name": "D1", "path": d1_path}])
        rep2 = eng.build_from_datasets([
            {"id": "d1", "name": "D1", "path": d1_path},
            {"id": "d2", "name": "D2", "path": d2_path},
        ])

        assert rep2["overview"]["canonical_products_tracked"] == 2
        with open(eng.catalogue_path) as f:
            cat = json.load(f)
        bag = next(p for p in cat if p["canonical_product"] == "Travel Bag")
        assert bag["total_occurrences"] == 3
        assert bag["dataset_coverage"] == 2
        assert abs(sum(c["contribution_pct"] for c in bag["contributions"]) - 100.0) < 0.1

        snaps = eng.load_snapshots()
        assert len(snaps) == 2
        assert snaps[0]["version_id"] == "v1.0"
        assert snaps[1]["version_id"] == "v2.0"


def test_collective_api_endpoints():
    client = TestClient(backend_main.app)
    r_ov = client.get("/api/collective/overview")
    assert r_ov.status_code == 200
    assert r_ov.json()["overview"]["canonical_products_tracked"] == 5374

    r_pr = client.get("/api/collective/products", params={"search": "bag", "limit": 10})
    assert r_pr.status_code == 200
    assert r_pr.json()["total"] > 0
    assert len(r_pr.json()["products"]) <= 10

    r_ev = client.get("/api/collective/evolution")
    assert r_ev.status_code == 200
    r_gr = client.get("/api/collective/graph")
    assert r_gr.status_code == 200
    r_pd = client.get("/api/collective/predictive")
    assert r_pd.status_code == 200


if __name__ == "__main__":
    test_category_normalization_and_fallback()
    test_collective_engine_synthetic_end_to_end()
    test_collective_api_endpoints()
    print("ALL COLLECTIVE ENGINE TESTS PASSED!")
