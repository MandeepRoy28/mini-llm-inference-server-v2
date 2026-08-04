"""
Tests for Problem 047 — Build the SSE Streaming Endpoint
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.047_build_sse_streaming_endpoint")
    create_streaming_app = mod.create_streaming_app
except (ModuleNotFoundError, AttributeError):
    create_streaming_app = None

pytestmark = pytest.mark.skipif(
    create_streaming_app is None,
    reason="Solution 047 not implemented yet",
)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    app = create_streaming_app()
    return TestClient(app)


def test_status_200(client):
    resp = client.post("/generate/stream", json={"prompt": "hello", "max_tokens": 3})
    assert resp.status_code == 200


def test_content_type_is_event_stream(client):
    resp = client.post("/generate/stream", json={"prompt": "hello", "max_tokens": 3})
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_last_sse_event_is_done(client):
    resp = client.post("/generate/stream", json={"prompt": "hello world", "max_tokens": 5})
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) > 0
    assert data_lines[-1] == "data: [DONE]"


def test_sse_events_formatted_correctly(client):
    resp = client.post("/generate/stream", json={"prompt": "abc def", "max_tokens": 4})
    # Every non-empty line should be a data event
    non_empty = [l for l in resp.text.splitlines() if l.strip()]
    for line in non_empty:
        assert line.startswith("data: "), f"Unexpected line: {line!r}"


def test_invalid_max_tokens_returns_422(client):
    resp = client.post("/generate/stream", json={"prompt": "hi", "max_tokens": -5})
    assert resp.status_code == 422
