"""
Tests for Problem 037 — Iteration-Level Scheduler
"""

import importlib
import queue
import pytest

try:
    mod = importlib.import_module("solutions.037_iteration_level_scheduler")
    iteration_level_scheduler = mod.iteration_level_scheduler
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 037 not found — skipping tests.",
        allow_module_level=True,
    )


def _make_queue(*request_ids):
    q = queue.Queue()
    for rid in request_ids:
        q.put({"request_id": rid, "prompt_ids": [1, 2], "max_new_tokens": 4, "temperature": 1.0})
    return q


def test_adds_request_when_capacity_available():
    rq = _make_queue(0)
    batch = {}
    added = iteration_level_scheduler(rq, batch, max_batch_size=2)
    assert len(added) == 1
    assert 0 in batch


def test_empty_queue_returns_empty_list():
    rq = queue.Queue()
    batch = {}
    added = iteration_level_scheduler(rq, batch, max_batch_size=4)
    assert added == []
    assert len(batch) == 0


def test_respects_max_batch_size():
    rq = _make_queue(0, 1, 2, 3)
    batch = {}
    added = iteration_level_scheduler(rq, batch, max_batch_size=2)
    assert len(added) == 2
    assert len(batch) == 2
    assert rq.qsize() == 2  # 2 remaining


def test_does_not_add_when_batch_full():
    rq = _make_queue(5)
    # Pre-fill batch to capacity
    batch = {0: {}, 1: {}}
    added = iteration_level_scheduler(rq, batch, max_batch_size=2)
    assert added == []
    assert rq.qsize() == 1  # request still in queue


def test_sequence_state_populated():
    rq = _make_queue(7)
    batch = {}
    iteration_level_scheduler(rq, batch, max_batch_size=4)
    seq_id = list(batch.keys())[0]
    state = batch[seq_id]
    for key in ("request_id", "token_ids", "max_new_tokens", "tokens_generated", "temperature", "finished"):
        assert key in state, f"Missing key: {key}"
