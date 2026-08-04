"""
Tests for Problem 041 — Run the Continuous Batching Loop
"""

import importlib
import queue
import pytest

try:
    mod = importlib.import_module("solutions.041_run_continuous_batching_loop")
    run_continuous_batching_loop = mod.run_continuous_batching_loop
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution 041 not found — skipping tests.",
        allow_module_level=True,
    )


def _mock_prefill(batch):
    return [0] * len(batch["input_ids"])


def _mock_decode(batch):
    """Always returns token 99."""
    return [99] * len(batch["input_ids"])


def _make_request(rid, max_new_tokens=2):
    return {
        "request_id": rid,
        "prompt_ids": [1, 2],
        "max_new_tokens": max_new_tokens,
        "temperature": 1.0,
    }


def _make_queue(*request_ids, max_new_tokens=2):
    rq = queue.Queue()
    for rid in request_ids:
        rq.put(_make_request(rid, max_new_tokens=max_new_tokens))
    return rq


def test_single_request_completes():
    rq = _make_queue(0, max_new_tokens=3)
    pool = {"free": list(range(50)), "used": []}
    results = run_continuous_batching_loop(
        _mock_prefill, _mock_decode, rq, max_batch_size=4, pool=pool
    )
    assert len(results) == 1
    assert results[0]["request_id"] == 0
    assert results[0]["tokens_generated"] == 3


def test_multiple_requests_all_complete():
    rq = _make_queue(0, 1, 2, max_new_tokens=2)
    pool = {"free": list(range(200)), "used": []}
    results = run_continuous_batching_loop(
        _mock_prefill, _mock_decode, rq, max_batch_size=4, pool=pool
    )
    assert len(results) == 3
    request_ids = {r["request_id"] for r in results}
    assert request_ids == {0, 1, 2}


def test_empty_queue_returns_empty_list():
    rq = queue.Queue()
    pool = {"free": list(range(50)), "used": []}
    results = run_continuous_batching_loop(
        _mock_prefill, _mock_decode, rq, max_batch_size=4, pool=pool
    )
    assert results == []


def test_eos_stops_generation_early():
    def decode_eos(batch):
        # Always return eos token 5
        return [5] * len(batch["input_ids"])

    rq = _make_queue(0, max_new_tokens=100)
    pool = {"free": list(range(50)), "used": []}
    results = run_continuous_batching_loop(
        _mock_prefill, decode_eos, rq, max_batch_size=4, pool=pool, eos_token_id=5
    )
    assert len(results) == 1
    # Should stop at first EOS (1 token generated), not run all 100
    assert results[0]["tokens_generated"] == 1


def test_completed_sequences_have_token_ids():
    rq = _make_queue(0, max_new_tokens=2)
    pool = {"free": list(range(50)), "used": []}
    results = run_continuous_batching_loop(
        _mock_prefill, _mock_decode, rq, max_batch_size=4, pool=pool
    )
    assert "token_ids" in results[0]
    # token_ids should include prompt tokens plus generated tokens
    assert len(results[0]["token_ids"]) >= 2
