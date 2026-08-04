"""
Tests for Problem 035 — Build a Request Queue
"""

import importlib
import queue
import pytest

try:
    mod = importlib.import_module("solutions.035_build_request_queue")
    build_request_queue = mod.build_request_queue
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 035 not found — skipping tests.",
        allow_module_level=True,
    )


def _make_request(rid, prompt_ids, max_new_tokens=10, temperature=1.0):
    return {
        "request_id": rid,
        "prompt_ids": prompt_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }


def test_single_request_size():
    reqs = [_make_request(0, [1, 2, 3])]
    q = build_request_queue(reqs)
    assert isinstance(q, queue.Queue)
    assert q.qsize() == 1


def test_empty_input_returns_empty_queue():
    q = build_request_queue([])
    assert q.empty()


def test_fifo_order_preserved():
    reqs = [_make_request(i, [i]) for i in range(4)]
    q = build_request_queue(reqs)
    for expected_id in range(4):
        item = q.get_nowait()
        assert item["request_id"] == expected_id


def test_queue_contains_original_dicts():
    req = _make_request(99, [10, 20, 30], max_new_tokens=5, temperature=0.7)
    q = build_request_queue([req])
    item = q.get_nowait()
    assert item["prompt_ids"] == [10, 20, 30]
    assert item["temperature"] == 0.7


def test_multiple_requests_correct_size():
    reqs = [_make_request(i, list(range(i + 1))) for i in range(7)]
    q = build_request_queue(reqs)
    assert q.qsize() == 7
