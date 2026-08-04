"""
Tests for Problem 033 — Benchmark Memory Utilization: Naive vs Paged
"""

import importlib
import math
import pytest

try:
    mod = importlib.import_module("solutions.033_benchmark_memory_utilization")
    benchmark_memory_utilization = mod.benchmark_memory_utilization
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 033_benchmark_memory_utilization not found — write your solution first.",
        allow_module_level=True,
    )


def test_required_keys_present():
    result = benchmark_memory_utilization([10, 20], page_size=16, max_seq_len=64)
    assert set(result.keys()) >= {
        "naive_total_allocated",
        "paged_total_allocated",
        "naive_waste_pct",
        "paged_waste_pct",
        "memory_reduction_factor",
    }


def test_perfect_utilisation():
    # Sequences exactly fill pages and max_seq_len
    result = benchmark_memory_utilization([16, 16], page_size=16, max_seq_len=16)
    assert result["naive_waste_pct"] == 0.0
    assert result["paged_waste_pct"] == 0.0
    assert result["memory_reduction_factor"] == pytest.approx(1.0)


def test_paged_always_better_or_equal_than_naive():
    result = benchmark_memory_utilization([10, 5, 8], page_size=16, max_seq_len=64)
    assert result["paged_total_allocated"] <= result["naive_total_allocated"]


def test_memory_reduction_factor_for_short_sequences():
    result = benchmark_memory_utilization([3], page_size=16, max_seq_len=512)
    # Naive = 512, paged = 16 → factor = 32
    assert result["memory_reduction_factor"] == pytest.approx(32.0)


def test_naive_total_allocated():
    seqs = [10, 20, 30]
    result = benchmark_memory_utilization(seqs, page_size=16, max_seq_len=100)
    assert result["naive_total_allocated"] == len(seqs) * 100


def test_paged_waste_pct_range():
    result = benchmark_memory_utilization([7, 9, 13], page_size=16, max_seq_len=64)
    assert 0.0 <= result["paged_waste_pct"] < 100.0
