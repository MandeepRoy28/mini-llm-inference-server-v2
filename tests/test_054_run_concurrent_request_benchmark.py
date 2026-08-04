"""Tests for Problem 054 — Run a Concurrent-Request Benchmark."""

import asyncio
import importlib
import pytest

try:
    mod = importlib.import_module("solutions.054_run_concurrent_request_benchmark")
    run_concurrent_benchmark = mod.run_concurrent_benchmark
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 054 not found — skipping tests.", allow_module_level=True)


async def _async_engine(req):
    """Minimal async engine: sleeps briefly and returns tokens."""
    await asyncio.sleep(0.005)
    return list(range(req["max_new_tokens"]))


def _make_requests(n, max_new_tokens=10):
    return [{"request_id": i, "max_new_tokens": max_new_tokens} for i in range(n)]


def test_return_keys():
    """Result contains all required top-level keys."""
    reqs = _make_requests(4)
    result = run_concurrent_benchmark(_async_engine, reqs, concurrency=2)
    expected = {
        "total_requests", "concurrency", "total_time",
        "throughput_req_per_sec", "throughput_tokens_per_sec", "individual_results",
    }
    assert expected.issubset(result.keys())


def test_total_requests_count():
    """total_requests matches the length of the input list."""
    reqs = _make_requests(6)
    result = run_concurrent_benchmark(_async_engine, reqs, concurrency=3)
    assert result["total_requests"] == 6


def test_concurrency_echoed():
    """concurrency in result matches the argument."""
    reqs = _make_requests(4)
    result = run_concurrent_benchmark(_async_engine, reqs, concurrency=4)
    assert result["concurrency"] == 4


def test_individual_results_length():
    """individual_results contains one entry per request."""
    reqs = _make_requests(5)
    result = run_concurrent_benchmark(_async_engine, reqs, concurrency=2)
    assert len(result["individual_results"]) == 5


def test_throughput_positive():
    """Both throughput figures are strictly positive."""
    reqs = _make_requests(4, max_new_tokens=8)
    result = run_concurrent_benchmark(_async_engine, reqs, concurrency=2)
    assert result["throughput_req_per_sec"] > 0
    assert result["throughput_tokens_per_sec"] > 0
