"""
Tests for Problem 044 — Define the GenerateResponse Schema
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.044_define_generate_response_schema")
    GenerateResponse = mod.GenerateResponse
except (ModuleNotFoundError, AttributeError):
    GenerateResponse = None

pytestmark = pytest.mark.skipif(
    GenerateResponse is None,
    reason="Solution 044 not implemented yet",
)


def _make_response(**kwargs):
    defaults = dict(
        generated_text="The sky is blue.",
        tokens_generated=4,
        time_to_first_token=0.05,
        total_time=0.2,
        request_id="req-001",
    )
    defaults.update(kwargs)
    return GenerateResponse(**defaults)


def test_fields_stored_correctly():
    resp = _make_response()
    assert resp.generated_text == "The sky is blue."
    assert resp.tokens_generated == 4
    assert resp.time_to_first_token == 0.05
    assert resp.total_time == 0.2
    assert resp.request_id == "req-001"


def test_tokens_per_second_basic():
    resp = _make_response(tokens_generated=4, total_time=0.2)
    assert resp.tokens_per_second() == pytest.approx(20.0)


def test_tokens_per_second_single_token():
    resp = _make_response(tokens_generated=1, total_time=0.1)
    assert resp.tokens_per_second() == pytest.approx(10.0)


def test_tokens_per_second_high_throughput():
    resp = _make_response(tokens_generated=100, total_time=1.0)
    assert resp.tokens_per_second() == pytest.approx(100.0)


def test_response_is_serialisable():
    resp = _make_response()
    data = resp.model_dump()
    assert isinstance(data, dict)
    assert "generated_text" in data
    assert "tokens_generated" in data
    assert "request_id" in data
