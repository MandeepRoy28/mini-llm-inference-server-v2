"""Tests for Problem 053 — Run a Single-Request Benchmark."""

import importlib
import time
import pytest

try:
    mod = importlib.import_module("solutions.053_run_single_request_benchmark")
    run_single_request_benchmark = mod.run_single_request_benchmark
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 053 not found — skipping tests.", allow_module_level=True)


def _dummy_engine(req):
    """Minimal synchronous engine that sleeps briefly and returns tokens."""
    time.sleep(0.005)
    return list(range(req["max_new_tokens"]))


def test_return_keys():
    """Result contains all required keys."""
    req = {"request_id": 0, "max_new_tokens": 4}
    result = run_single_request_benchmark(_dummy_engine, req, n_warmup=0)
    assert {"request_id", "ttft", "tpot", "total_time", "tokens_generated"}.issubset(
        result.keys()
    )


def test_request_id_propagated():
    """request_id in result matches the input request."""
    req = {"request_id": 42, "max_new_tokens": 2}
    result = run_single_request_benchmark(_dummy_engine, req, n_warmup=0)
    assert result["request_id"] == 42


def test_tokens_generated_count():
    """tokens_generated equals the number of tokens returned by engine."""
    req = {"request_id": 1, "max_new_tokens": 8}
    result = run_single_request_benchmark(_dummy_engine, req, n_warmup=1)
    assert result["tokens_generated"] == 8


def test_positive_timings():
    """All timing fields are strictly positive."""
    req = {"request_id": 2, "max_new_tokens": 3}
    result = run_single_request_benchmark(_dummy_engine, req, n_warmup=1)
    assert result["total_time"] > 0
    assert result["ttft"] > 0
    assert result["tpot"] > 0


def test_tpot_relation():
    """TPOT equals total_time / tokens_generated (within float tolerance)."""
    req = {"request_id": 3, "max_new_tokens": 5}
    result = run_single_request_benchmark(_dummy_engine, req, n_warmup=0)
    expected_tpot = result["total_time"] / result["tokens_generated"]
    assert abs(result["tpot"] - expected_tpot) < 1e-9
