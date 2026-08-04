"""
Tests for Problem 050 — Test Streaming with curl
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.050_test_streaming_with_curl")
    build_curl_command = mod.build_curl_command
    parse_sse_response = mod.parse_sse_response
except (ModuleNotFoundError, AttributeError):
    build_curl_command = None
    parse_sse_response = None

pytestmark = pytest.mark.skipif(
    build_curl_command is None or parse_sse_response is None,
    reason="Solution 050 not implemented yet",
)


# --- build_curl_command tests ---

def test_curl_command_basic_structure():
    cmd = build_curl_command("localhost", 8000, "Hello world")
    assert cmd.startswith("curl -X POST")
    assert "http://localhost:8000/generate/stream" in cmd
    assert "--no-buffer" in cmd


def test_curl_command_default_max_tokens():
    cmd = build_curl_command("localhost", 8000, "Hello world")
    assert '"max_tokens": 50' in cmd or "'max_tokens': 50" in cmd


def test_curl_command_custom_max_tokens():
    cmd = build_curl_command("192.168.1.10", 9000, "Summarise this", max_tokens=200)
    assert "192.168.1.10:9000" in cmd
    assert "200" in cmd


def test_curl_command_prompt_embedded():
    cmd = build_curl_command("localhost", 8000, "Once upon a time")
    assert "Once upon a time" in cmd


def test_curl_command_content_type_header():
    cmd = build_curl_command("localhost", 8000, "test")
    assert "Content-Type" in cmd
    assert "application/json" in cmd


# --- parse_sse_response tests ---

def test_parse_basic_tokens():
    raw = "data: Hello\n\ndata: world\n\ndata: [DONE]\n\n"
    result = parse_sse_response(raw)
    assert result == ["Hello", "world"]


def test_parse_excludes_done_sentinel():
    raw = "data: foo\n\ndata: [DONE]\n\n"
    result = parse_sse_response(raw)
    assert "[DONE]" not in result
    assert result == ["foo"]


def test_parse_empty_response():
    assert parse_sse_response("") == []


def test_parse_no_done_sentinel():
    raw = "data: foo\n\ndata: bar\n\n"
    result = parse_sse_response(raw)
    assert result == ["foo", "bar"]


def test_parse_ignores_non_data_lines():
    raw = ": keep-alive\n\ndata: hello\n\n\ndata: [DONE]\n\n"
    result = parse_sse_response(raw)
    assert result == ["hello"]
