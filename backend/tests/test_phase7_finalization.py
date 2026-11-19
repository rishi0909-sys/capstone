import os
import sys
import json
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import main as backend_main


def test_manifest_and_artifact_synchronization():
    manifest = backend_main._validate_dataset_manifest()
    assert manifest is not None
    assert manifest["dataset_id"] == "dataset_3_clean.csv"
    assert len(manifest["sha256_checksum"]) == 64

    with open(os.path.join(BACKEND_DIR, "models", "production", "hgb_overall_daily_metadata.json")) as f:
        d_meta = json.load(f)
    with open(os.path.join(BACKEND_DIR, "models", "production", "hgb_overall_weekly_metadata.json")) as f:
        w_meta = json.load(f)
    assert d_meta["dataset_id"] == manifest["dataset_id"]
    assert w_meta["dataset_id"] == manifest["dataset_id"]

    with open(os.path.join(BACKEND_DIR, "models", "customer_behavior.json")) as f:
        cb = json.load(f)
    assert cb["summary"]["total_unique_customers"] == 8779

    with open(os.path.join(BACKEND_DIR, "models", "association_rules.json")) as f:
        rules = json.load(f)
    assert len(rules) == 141


def test_frozen_api_endpoints():
    client = TestClient(backend_main.app)
    assert client.get("/api/mining/rules", params={"dataset_id": "demo", "limit": 5}).status_code == 200
    assert client.post("/api/demographics/analyze", json={"dataset_id": "demo"}).status_code == 200
    assert client.post("/api/temporal/analyze", json={"dataset_id": "demo"}).status_code == 200
    assert client.post("/api/insights/generate", json={"dataset_id": "demo"}).status_code == 200
    r_fc = client.post("/api/forecast/run", json={"dataset_id": "demo", "granularity": "D", "horizon": 14, "target_product": "Overall"})
    assert r_fc.status_code == 200
    assert len(r_fc.json()["future_preds"]) == 14


if __name__ == "__main__":
    test_manifest_and_artifact_synchronization()
    test_frozen_api_endpoints()
    print("ALL PHASE 7 FINALIZATION TESTS PASSED!")
