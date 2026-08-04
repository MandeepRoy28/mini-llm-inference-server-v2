"""
Tests for Problem 040 — Handle Sequence Completion
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.040_handle_sequence_completion")
    handle_sequence_completion = mod.handle_sequence_completion
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 040 not found — skipping tests.",
        allow_module_level=True,
    )


def _make_state(rid, token_ids, max_new_tokens, tokens_generated):
    return {
        "request_id": rid,
        "token_ids": list(token_ids),
        "max_new_tokens": max_new_tokens,
        "tokens_generated": tokens_generated,
        "temperature": 1.0,
        "finished": False,
    }


def test_eos_triggers_completion():
    batch = {0: _make_state(0, [1, 2, 3, 99], max_new_tokens=10, tokens_generated=3)}
    pool = {"free": [], "used": [5, 6]}
    block_table = {0: [5, 6]}
    completed = handle_sequence_completion(batch, pool, block_table, eos_token_id=99)
    assert len(completed) == 1
    assert completed[0]["request_id"] == 0
    assert 0 not in batch


def test_max_new_tokens_triggers_completion():
    batch = {0: _make_state(0, [1, 2, 3, 4, 5], max_new_tokens=3, tokens_generated=3)}
    pool = {"free": [], "used": [10]}
    block_table = {0: [10]}
    completed = handle_sequence_completion(batch, pool, block_table, eos_token_id=None)
    assert len(completed) == 1
    assert 0 not in batch


def test_pages_freed_on_completion():
    batch = {0: _make_state(0, [1, 2, 99], max_new_tokens=10, tokens_generated=2)}
    pool = {"free": [], "used": [7, 8]}
    block_table = {0: [7, 8]}
    handle_sequence_completion(batch, pool, block_table, eos_token_id=99)
    assert 7 in pool["free"] and 8 in pool["free"]
    assert pool["used"] == [] or (7 not in pool["used"] and 8 not in pool["used"])


def test_unfinished_sequence_stays_in_batch():
    batch = {0: _make_state(0, [1, 2, 3], max_new_tokens=10, tokens_generated=3)}
    pool = {"free": list(range(10)), "used": []}
    block_table = {0: []}
    completed = handle_sequence_completion(batch, pool, block_table, eos_token_id=99)
    assert len(completed) == 0
    assert 0 in batch


def test_completed_sequence_has_required_keys():
    batch = {0: _make_state(0, [1, 99], max_new_tokens=10, tokens_generated=1)}
    pool = {"free": [], "used": [3]}
    block_table = {0: [3]}
    completed = handle_sequence_completion(batch, pool, block_table, eos_token_id=99)
    seq = completed[0]
    for key in ("request_id", "token_ids", "tokens_generated"):
        assert key in seq, f"Missing key: {key}"
