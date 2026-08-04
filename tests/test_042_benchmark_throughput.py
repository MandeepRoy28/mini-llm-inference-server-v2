"""
Tests for Problem 042 — Benchmark Throughput
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.042_benchmark_throughput")
    benchmark_throughput = mod.benchmark_throughput
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 042 not found — skipping tests.",
        allow_module_level=True,
    )


def test_docstring_example():
    seqs = [{"tokens_generated": 10}, {"tokens_generated": 20}, {"tokens_generated": 30}]
    result = benchmark_throughput(seqs, total_time=2.0)
    assert result["total_requests"] == 3
    assert result["total_tokens_generated"] == 60
    assert result["requests_per_sec"] == 1.5
    assert result["tokens_per_sec"] == 30.0
    assert result["avg_tokens_per_request"] == 20.0


def test_empty_sequences():
    result = benchmark_throughput([], total_time=1.0)
    assert result["total_requests"] == 0
    assert result["total_tokens_generated"] == 0
    assert result["avg_tokens_per_request"] == 0.0
    assert result["tokens_per_sec"] == 0.0


def test_single_sequence():
    seqs = [{"tokens_generated": 7}]
    result = benchmark_throughput(seqs, total_time=3.5)
    assert result["total_requests"] == 1
    assert result["total_tokens_generated"] == 7
    assert result["tokens_per_sec"] == round(7 / 3.5, 4)


def test_all_required_keys_present():
    seqs = [{"tokens_generated": 5}, {"tokens_generated": 5}]
    result = benchmark_throughput(seqs, total_time=1.0)
    for key in ("total_requests", "total_tokens_generated", "requests_per_sec", "tokens_per_sec", "avg_tokens_per_request"):
        assert key in result, f"Missing key: {key}"


def test_values_are_correct_types():
    seqs = [{"tokens_generated": 10}]
    result = benchmark_throughput(seqs, total_time=2.0)
    assert isinstance(result["total_requests"], int)
    assert isinstance(result["total_tokens_generated"], int)
    assert isinstance(result["requests_per_sec"], float)
    assert isinstance(result["tokens_per_sec"], float)
    assert isinstance(result["avg_tokens_per_request"], float)
