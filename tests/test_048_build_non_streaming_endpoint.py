"""
Tests for Problem 048 — Build the Non-Streaming Generation Endpoint
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.048_build_non_streaming_endpoint")
    create_app = mod.create_app
except (ModuleNotFoundError, AttributeError):
    create_app = None

pytestmark = pytest.mark.skipif(
    create_app is None,
    reason="Solution 048 not implemented yet",
)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    app = create_app()
    return TestClient(app)


def test_status_200(client):
    resp = client.post("/generate", json={"prompt": "hello world", "max_tokens": 5})
    assert resp.status_code == 200


def test_response_has_required_fields(client):
    resp = client.post("/generate", json={"prompt": "hello world", "max_tokens": 5})
    data = resp.json()
    required = {"generated_text", "tokens_generated", "time_to_first_token", "total_time", "request_id"}
    assert required.issubset(data.keys())


def test_tokens_generated_matches_max_tokens(client):
    resp = client.post("/generate", json={"prompt": "hello world", "max_tokens": 7})
    data = resp.json()
    assert data["tokens_generated"] == 7


def test_request_id_is_non_empty_string(client):
    resp = client.post("/generate", json={"prompt": "hi", "max_tokens": 3})
    data = resp.json()
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0


def test_timing_fields_non_negative_and_ordered(client):
    resp = client.post("/generate", json={"prompt": "test prompt", "max_tokens": 5})
    data = resp.json()
    assert data["time_to_first_token"] >= 0
    assert data["total_time"] >= 0
    assert data["total_time"] >= data["time_to_first_token"]


def test_unique_request_ids(client):
    ids = set()
    for _ in range(5):
        resp = client.post("/generate", json={"prompt": "hi", "max_tokens": 2})
        ids.add(resp.json()["request_id"])
    assert len(ids) == 5
