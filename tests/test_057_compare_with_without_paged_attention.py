"""Tests for Problem 057 — Compare Paged vs Naive Attention Benchmarks."""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.057_compare_with_without_paged_attention")
    compare_paged_vs_naive = mod.compare_paged_vs_naive
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 057 not found — skipping tests.", allow_module_level=True)

NAIVE = {
    "memory_used_gb": 8.0,
    "throughput_tokens_per_sec": 100.0,
    "ttft_mean": 0.20,
}
PAGED = {
    "memory_used_gb": 4.0,
    "throughput_tokens_per_sec": 180.0,
    "ttft_mean": 0.10,
}


def test_memory_reduction_factor():
    """memory_reduction_factor equals naive_memory / paged_memory."""
    cmp = compare_paged_vs_naive(NAIVE, PAGED)
    assert abs(cmp["memory_reduction_factor"] - 2.0) < 1e-9


def test_throughput_improvement_factor():
    """throughput_improvement_factor equals paged_tps / naive_tps."""
    cmp = compare_paged_vs_naive(NAIVE, PAGED)
    assert abs(cmp["throughput_improvement_factor"] - 1.8) < 1e-9


def test_latency_improvement_factor():
    """latency_improvement_factor equals naive_ttft / paged_ttft."""
    cmp = compare_paged_vs_naive(NAIVE, PAGED)
    assert abs(cmp["latency_improvement_factor"] - 2.0) < 1e-9


def test_summary_table_structure():
    """summary_table has at least 3 rows, each with the required keys."""
    cmp = compare_paged_vs_naive(NAIVE, PAGED)
    table = cmp["summary_table"]
    assert len(table) >= 3
    for row in table:
        assert {"metric", "naive", "paged", "improvement"}.issubset(row.keys())


def test_no_improvement_when_equal():
    """All improvement factors are 1.0 when naive and paged results are identical."""
    same = {"memory_used_gb": 4.0, "throughput_tokens_per_sec": 150.0, "ttft_mean": 0.1}
    cmp = compare_paged_vs_naive(same, same)
    assert abs(cmp["memory_reduction_factor"] - 1.0) < 1e-9
    assert abs(cmp["throughput_improvement_factor"] - 1.0) < 1e-9
    assert abs(cmp["latency_improvement_factor"] - 1.0) < 1e-9
