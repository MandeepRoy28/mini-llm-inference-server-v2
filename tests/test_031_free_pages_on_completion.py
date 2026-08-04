"""
Tests for Problem 031 — Free Pages on Sequence Completion
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.031_free_pages_on_completion")
    free_pages_on_completion = mod.free_pages_on_completion
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 031_free_pages_on_completion not found — write your solution first.",
        allow_module_level=True,
    )


def test_pages_returned_to_pool():
    pool = {"free_pages": [2, 3]}
    bt = {0: [0, 1]}
    free_pages_on_completion(pool, bt, seq_id=0)
    assert 0 in pool["free_pages"]
    assert 1 in pool["free_pages"]


def test_block_table_cleared():
    pool = {"free_pages": []}
    bt = {0: [0, 1, 2]}
    free_pages_on_completion(pool, bt, seq_id=0)
    assert bt[0] == []


def test_returns_correct_count():
    pool = {"free_pages": []}
    bt = {0: [10, 11, 12]}
    count = free_pages_on_completion(pool, bt, seq_id=0)
    assert count == 3


def test_freeing_empty_sequence():
    pool = {"free_pages": [0]}
    bt = {5: []}
    count = free_pages_on_completion(pool, bt, seq_id=5)
    assert count == 0
    assert pool["free_pages"] == [0]


def test_only_target_sequence_affected():
    pool = {"free_pages": []}
    bt = {0: [0, 1], 1: [4]}
    free_pages_on_completion(pool, bt, seq_id=0)
    assert bt[1] == [4]  # other sequence untouched
