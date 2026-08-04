"""Tests for Problem 024 — Benchmark KV Cache Speedup."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.024_benchmark_kv_cache_speedup")
    benchmark_kv_cache_speedup = mod.benchmark_kv_cache_speedup
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 024 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 16


def _make_forwards():
    def fwd(ids):
        B, T = ids.shape
        logits = torch.zeros(B, T, VOCAB)
        logits[:, :, 1] = 10.0
        return logits

    def fwd_cache(ids, cache):
        return fwd(ids), cache

    return fwd, fwd_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "no_cache_time",
    "with_cache_time",
    "speedup_factor",
    "tokens_per_sec_no_cache",
    "tokens_per_sec_with_cache",
}


def test_return_type_and_keys():
    """Result must be a dict with exactly the five expected keys."""
    fwd, fwd_cache = _make_forwards()
    result = benchmark_kv_cache_speedup(
        fwd, fwd_cache, torch.zeros(1, 4, dtype=torch.long), n_new_tokens=3
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == EXPECTED_KEYS


def test_all_values_positive():
    """All timing and throughput values must be strictly positive."""
    fwd, fwd_cache = _make_forwards()
    result = benchmark_kv_cache_speedup(
        fwd, fwd_cache, torch.zeros(1, 4, dtype=torch.long), n_new_tokens=3
    )
    for key in EXPECTED_KEYS:
        assert result[key] > 0, f"Expected {key} > 0, got {result[key]}"


def test_speedup_factor_consistency():
    """speedup_factor must equal no_cache_time / with_cache_time."""
    fwd, fwd_cache = _make_forwards()
    result = benchmark_kv_cache_speedup(
        fwd, fwd_cache, torch.zeros(1, 4, dtype=torch.long), n_new_tokens=4
    )
    expected = result["no_cache_time"] / result["with_cache_time"]
    assert abs(result["speedup_factor"] - expected) < 1e-6


def test_tokens_per_sec_consistency():
    """Throughput values must equal n_new_tokens / respective times."""
    n = 5
    fwd, fwd_cache = _make_forwards()
    result = benchmark_kv_cache_speedup(
        fwd, fwd_cache, torch.zeros(1, 4, dtype=torch.long), n_new_tokens=n
    )
    assert abs(result["tokens_per_sec_no_cache"] - n / result["no_cache_time"]) < 1e-6
    assert abs(result["tokens_per_sec_with_cache"] - n / result["with_cache_time"]) < 1e-6


def test_both_paths_generate_n_tokens():
    """Both generation paths must run for exactly n_new_tokens steps."""
    no_cache_calls = {"n": 0}
    cache_calls = {"n": 0}
    n = 4

    def fwd_counting(ids):
        no_cache_calls["n"] += 1
        B, T = ids.shape
        return torch.zeros(B, T, VOCAB)

    def fwd_cache_counting(ids, cache):
        cache_calls["n"] += 1
        B, T = ids.shape
        return torch.zeros(B, T, VOCAB), cache

    benchmark_kv_cache_speedup(
        fwd_counting, fwd_cache_counting, torch.zeros(1, 3, dtype=torch.long), n_new_tokens=n
    )
    # No-cache: exactly n calls (one per step)
    assert no_cache_calls["n"] == n
    # Cached: 1 prefill + n decode calls = n+1 total
    assert cache_calls["n"] == n + 1
