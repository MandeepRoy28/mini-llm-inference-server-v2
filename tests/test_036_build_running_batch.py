"""
Tests for Problem 036 — Build a Running Batch
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.036_build_running_batch")
    build_running_batch = mod.build_running_batch
    add_to_batch = mod.add_to_batch
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 036 not found — skipping tests.",
        allow_module_level=True,
    )


def _make_request(rid, prompt_ids, max_new_tokens=5, temperature=1.0):
    return {
        "request_id": rid,
        "prompt_ids": prompt_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }


def test_build_running_batch_returns_empty_dict():
    batch = build_running_batch()
    assert isinstance(batch, dict)
    assert len(batch) == 0


def test_add_to_batch_inserts_entry():
    batch = build_running_batch()
    req = _make_request(7, [10, 20])
    add_to_batch(batch, req, seq_id=0)
    assert 0 in batch


def test_add_to_batch_state_keys():
    batch = {}
    req = _make_request(3, [1, 2, 3], max_new_tokens=8, temperature=0.9)
    add_to_batch(batch, req, seq_id=1)
    state = batch[1]
    for key in ("request_id", "token_ids", "max_new_tokens", "tokens_generated", "temperature", "finished"):
        assert key in state, f"Missing key: {key}"


def test_add_to_batch_initial_values():
    batch = {}
    req = _make_request(42, [5, 6, 7], max_new_tokens=10, temperature=1.2)
    add_to_batch(batch, req, seq_id=0)
    state = batch[0]
    assert state["request_id"] == 42
    assert state["token_ids"] == [5, 6, 7]
    assert state["tokens_generated"] == 0
    assert state["finished"] is False
    assert state["temperature"] == 1.2


def test_add_multiple_sequences():
    batch = build_running_batch()
    for i in range(3):
        req = _make_request(i, [i, i + 1])
        add_to_batch(batch, req, seq_id=i)
    assert len(batch) == 3
    assert batch[2]["request_id"] == 2
