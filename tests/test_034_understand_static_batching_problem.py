"""
Tests for Problem 034 — Understand Static Batching
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.034_understand_static_batching_problem")
    simulate_static_batching = mod.simulate_static_batching
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 034 not found — skipping tests.",
        allow_module_level=True,
    )


def test_example_from_docstring():
    result = simulate_static_batching([5, 10, 3], max_seq_len=20)
    assert result == {
        "batch_size": 3,
        "max_seq_in_batch": 10,
        "total_steps": 10,
        "wasted_compute_pct": 46.67,
    }


def test_uniform_lengths_zero_waste():
    result = simulate_static_batching([8, 8, 8], max_seq_len=16)
    assert result["wasted_compute_pct"] == 0.0
    assert result["batch_size"] == 3
    assert result["total_steps"] == 8


def test_single_sequence():
    result = simulate_static_batching([7], max_seq_len=20)
    assert result["batch_size"] == 1
    assert result["max_seq_in_batch"] == 7
    assert result["total_steps"] == 7
    assert result["wasted_compute_pct"] == 0.0


def test_extreme_disparity():
    result = simulate_static_batching([1, 20], max_seq_len=20)
    assert result["batch_size"] == 2
    assert result["max_seq_in_batch"] == 20
    assert result["total_steps"] == 20
    assert result["wasted_compute_pct"] == 47.5


def test_return_keys_present():
    result = simulate_static_batching([4, 6, 2], max_seq_len=10)
    for key in ("batch_size", "max_seq_in_batch", "total_steps", "wasted_compute_pct"):
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["wasted_compute_pct"], float)
