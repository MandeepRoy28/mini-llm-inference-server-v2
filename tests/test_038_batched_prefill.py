"""
Tests for Problem 038 — Batched Prefill
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.038_batched_prefill")
    batched_prefill = mod.batched_prefill
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 038 not found — skipping tests.",
        allow_module_level=True,
    )


def _mock_forward(batch):
    """Returns sum of each prompt as a stand-in for logits."""
    return [sum(ids) for ids in batch["input_ids"]]


def _make_pool(n=100):
    return {"free": list(range(n)), "used": []}


def test_empty_requests_returns_empty_dict():
    result = batched_prefill(_mock_forward, [], _make_pool(), {})
    assert result == {}


def test_single_request_returns_logits():
    pool = _make_pool()
    block_table = {}
    reqs = [{"request_id": 0, "prompt_ids": [1, 2, 3], "max_new_tokens": 5, "temperature": 1.0}]
    result = batched_prefill(_mock_forward, reqs, pool, block_table)
    assert 0 in result
    assert result[0] == 6  # sum([1, 2, 3])


def test_block_table_populated():
    pool = _make_pool()
    block_table = {}
    reqs = [{"request_id": 0, "prompt_ids": list(range(20)), "max_new_tokens": 5, "temperature": 1.0}]
    batched_prefill(_mock_forward, reqs, pool, block_table)
    assert 0 in block_table
    assert len(block_table[0]) >= 1


def test_pages_moved_from_free_to_used():
    pool = _make_pool(50)
    block_table = {}
    reqs = [{"request_id": 0, "prompt_ids": list(range(16)), "max_new_tokens": 5, "temperature": 1.0}]
    batched_prefill(_mock_forward, reqs, pool, block_table)
    assert len(pool["free"]) < 50


def test_multiple_requests_all_in_result():
    pool = _make_pool(200)
    block_table = {}
    reqs = [
        {"request_id": i, "prompt_ids": [i, i + 1], "max_new_tokens": 3, "temperature": 1.0}
        for i in range(4)
    ]
    result = batched_prefill(_mock_forward, reqs, pool, block_table)
    assert len(result) == 4
    for i in range(4):
        assert i in result
