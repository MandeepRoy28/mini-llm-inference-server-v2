"""
Tests for Problem 028 — Assign Page to Sequence
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.028_assign_page_to_sequence")
    assign_page_to_sequence = mod.assign_page_to_sequence
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 028_assign_page_to_sequence not found — write your solution first.",
        allow_module_level=True,
    )


def test_page_assigned_and_removed_from_pool():
    pool = {"free_pages": [0, 1, 2]}
    bt = {0: []}
    page_id = assign_page_to_sequence(pool, bt, seq_id=0)
    assert page_id not in pool["free_pages"]
    assert page_id in bt[0]


def test_block_table_updated():
    pool = {"free_pages": [5, 6]}
    bt = {0: [], 1: []}
    assign_page_to_sequence(pool, bt, seq_id=1)
    assert len(bt[1]) == 1
    assert bt[0] == []


def test_multiple_pages_assigned_sequentially():
    pool = {"free_pages": [0, 1]}
    bt = {0: []}
    p0 = assign_page_to_sequence(pool, bt, seq_id=0)
    p1 = assign_page_to_sequence(pool, bt, seq_id=0)
    assert bt[0] == [p0, p1]
    assert pool["free_pages"] == []


def test_raises_on_empty_pool():
    with pytest.raises(ValueError):
        assign_page_to_sequence({"free_pages": []}, {0: []}, seq_id=0)


def test_return_value_is_int():
    pool = {"free_pages": [3]}
    bt = {2: []}
    result = assign_page_to_sequence(pool, bt, seq_id=2)
    assert isinstance(result, int)
