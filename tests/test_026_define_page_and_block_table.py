"""
Tests for Problem 026 — Define Page and Block Table
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.026_define_page_and_block_table")
    get_page_size = mod.get_page_size
    create_block_table = mod.create_block_table
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 026_define_page_and_block_table not found — write your solution first.",
        allow_module_level=True,
    )


def test_page_size_value():
    assert get_page_size() == 16


def test_page_size_return_type():
    assert isinstance(get_page_size(), int)


def test_block_table_three_sequences():
    bt = create_block_table(3)
    assert bt == {0: [], 1: [], 2: []}


def test_block_table_single_sequence():
    bt = create_block_table(1)
    assert bt == {0: []}


def test_block_table_keys_are_zero_indexed():
    n = 5
    bt = create_block_table(n)
    assert len(bt) == n
    assert set(bt.keys()) == set(range(n))
    for v in bt.values():
        assert v == []
