"""
Tests for Problem 039 — Batched Decode Step
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.039_batched_decode_step")
    batched_decode_step = mod.batched_decode_step
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 039 not found — skipping tests.",
        allow_module_level=True,
    )


def _mock_decode(batch):
    """Always returns token 42 for every sequence."""
    return [42] * len(batch["input_ids"])


def _make_state(rid, token_ids, max_new_tokens=5, tokens_generated=0):
    return {
        "request_id": rid,
        "token_ids": list(token_ids),
        "max_new_tokens": max_new_tokens,
        "tokens_generated": tokens_generated,
        "temperature": 1.0,
        "finished": False,
    }


def test_returns_sampled_token():
    batch = {0: _make_state(0, [1, 2, 3])}
    pool = {"free": list(range(10)), "used": []}
    block_table = {0: [0]}
    result = batched_decode_step(_mock_decode, batch, pool, block_table)
    assert result[0] == 42


def test_token_appended_to_sequence():
    batch = {0: _make_state(0, [1, 2, 3])}
    pool = {"free": list(range(10)), "used": []}
    block_table = {0: [0]}
    batched_decode_step(_mock_decode, batch, pool, block_table)
    assert batch[0]["token_ids"][-1] == 42


def test_tokens_generated_incremented():
    batch = {0: _make_state(0, [5, 6], tokens_generated=3)}
    pool = {"free": list(range(10)), "used": []}
    block_table = {0: [0]}
    batched_decode_step(_mock_decode, batch, pool, block_table)
    assert batch[0]["tokens_generated"] == 4


def test_multiple_sequences_all_decoded():
    batch = {
        0: _make_state(0, [1, 2]),
        1: _make_state(1, [3, 4]),
    }
    pool = {"free": list(range(20)), "used": []}
    block_table = {0: [0], 1: [1]}
    result = batched_decode_step(_mock_decode, batch, pool, block_table)
    assert 0 in result and 1 in result
    assert result[0] == 42 and result[1] == 42


def test_skips_finished_sequences():
    batch = {
        0: _make_state(0, [1, 2]),
        1: {**_make_state(1, [3, 4]), "finished": True},
    }
    pool = {"free": list(range(20)), "used": []}
    block_table = {0: [0], 1: [1]}
    result = batched_decode_step(_mock_decode, batch, pool, block_table)
    assert 0 in result
    assert 1 not in result
