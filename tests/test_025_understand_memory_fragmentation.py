"""
Tests for Problem 025 — Understand Memory Fragmentation in KV Cache
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.025_understand_memory_fragmentation")
    simulate_naive_allocation = mod.simulate_naive_allocation
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 025_understand_memory_fragmentation not found — write your solution first.",
        allow_module_level=True,
    )


def test_basic_example():
    result = simulate_naive_allocation([10, 5, 8], max_seq_len=20)
    assert result["total_slots_allocated"] == 60
    assert result["slots_actually_used"] == 23
    assert result["wasted_slots"] == 37
    assert result["waste_percentage"] == 61.67


def test_no_waste_when_sequences_fill_max():
    result = simulate_naive_allocation([100, 100], max_seq_len=100)
    assert result["total_slots_allocated"] == 200
    assert result["slots_actually_used"] == 200
    assert result["wasted_slots"] == 0
    assert result["waste_percentage"] == 0.0


def test_single_short_sequence():
    result = simulate_naive_allocation([1], max_seq_len=512)
    assert result["total_slots_allocated"] == 512
    assert result["slots_actually_used"] == 1
    assert result["wasted_slots"] == 511
    assert result["waste_percentage"] == pytest.approx(99.80, abs=0.01)


def test_return_keys_present():
    result = simulate_naive_allocation([5, 10], max_seq_len=30)
    assert set(result.keys()) == {
        "total_slots_allocated",
        "slots_actually_used",
        "wasted_slots",
        "waste_percentage",
    }


def test_waste_percentage_is_rounded():
    result = simulate_naive_allocation([7, 13], max_seq_len=25)
    # total_allocated = 50, used = 20, wasted = 30, pct = 60.0
    assert isinstance(result["waste_percentage"], float)
    assert result["waste_percentage"] == round(result["waste_percentage"], 2)
