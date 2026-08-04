"""
Tests for Problem 049 — Add Request-ID Tracking and Aggregate Stats
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.049_add_request_id_tracking")
    create_tracked_app = mod.create_tracked_app
    get_request_stats = mod.get_request_stats
except (ModuleNotFoundError, AttributeError):
    create_tracked_app = None
    get_request_stats = None

pytestmark = pytest.mark.skipif(
    create_tracked_app is None or get_request_stats is None,
    reason="Solution 049 not implemented yet",
)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    app = create_tracked_app()
    return TestClient(app)


def test_stats_endpoint_exists(client):
    resp = client.get("/stats")
    assert resp.status_code == 200


def test_stats_zero_on_fresh_app():
    from fastapi.testclient import TestClient
    fresh_app = create_tracked_app()
    fresh_client = TestClient(fresh_app)
    stats = fresh_client.get("/stats").json()
    assert stats["total_requests"] == 0


def test_stats_increment_with_requests(client):
    for _ in range(3):
        client.post("/generate", json={"prompt": "hi", "max_tokens": 3})
    stats = client.get("/stats").json()
    assert stats["total_requests"] == 3


def test_stats_fields_present(client):
    client.post("/generate", json={"prompt": "test", "max_tokens": 4})
    stats = client.get("/stats").json()
    required = {"total_requests", "avg_ttft", "avg_total_time", "p50_latency", "p95_latency"}
    assert required.issubset(stats.keys())


def test_p95_gte_p50(client):
    for _ in range(10):
        client.post("/generate", json={"prompt": "hello", "max_tokens": 5})
    stats = client.get("/stats").json()
    assert stats["p95_latency"] >= stats["p50_latency"]


def test_get_request_stats_returns_dict():
    stats = get_request_stats()
    assert isinstance(stats, dict)
    assert "total_requests" in stats
