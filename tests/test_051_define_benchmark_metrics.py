"""Tests for Problem 051 — Define Benchmark Metrics."""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.051_define_benchmark_metrics")
    compute_metrics = mod.compute_metrics
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 051 not found — skipping tests.", allow_module_level=True)


def test_return_keys():
    """All expected keys are present in the returned dict."""
    result = compute_metrics([0.1, 0.2, 0.15], [0.05, 0.06, 0.055], 100, 10.0)
    expected_keys = {
        "ttft_mean", "ttft_p50", "ttft_p95", "ttft_p99",
        "tpot_mean", "tpot_p50", "tpot_p95",
        "throughput_tokens_per_sec", "total_tokens", "total_time",
    }
    assert expected_keys.issubset(result.keys())


def test_ttft_mean_and_percentiles():
    """TTFT mean and percentiles are numerically correct."""
    result = compute_metrics([0.1, 0.2, 0.15], [0.05, 0.06, 0.055], 100, 10.0)
    assert abs(result["ttft_mean"] - 0.15) < 1e-9
    assert abs(result["ttft_p50"] - 0.15) < 1e-6
    assert result["ttft_p95"] > result["ttft_p50"]
    assert result["ttft_p99"] >= result["ttft_p95"]


def test_throughput_calculation():
    """Throughput equals total_tokens / total_time."""
    result = compute_metrics([0.05], [0.01], 50, 5.0)
    assert abs(result["throughput_tokens_per_sec"] - 10.0) < 1e-9


def test_totals_echoed():
    """total_tokens and total_time are echoed back unchanged."""
    result = compute_metrics([0.1], [0.05], 999, 7.5)
    assert result["total_tokens"] == 999
    assert abs(result["total_time"] - 7.5) < 1e-9


def test_single_sample():
    """Works with a single-element list (no percentile edge-case errors)."""
    result = compute_metrics([0.08], [0.02], 10, 1.0)
    # For a single value, mean == p50 == p95 == p99
    assert abs(result["ttft_mean"] - 0.08) < 1e-9
    assert abs(result["ttft_p50"] - 0.08) < 1e-6
    assert abs(result["ttft_p99"] - 0.08) < 1e-6
