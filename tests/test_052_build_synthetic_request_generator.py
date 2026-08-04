"""Tests for Problem 052 — Build a Synthetic Request Generator."""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.052_build_synthetic_request_generator")
    generate_synthetic_requests = mod.generate_synthetic_requests
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 052 not found — skipping tests.", allow_module_level=True)


def test_correct_length():
    """Returns exactly n requests."""
    reqs = generate_synthetic_requests(10, seed=42)
    assert len(reqs) == 10


def test_required_keys():
    """Every request dict has the three required keys."""
    reqs = generate_synthetic_requests(3, seed=0)
    for req in reqs:
        assert set(req.keys()) == {"request_id", "prompt_ids", "max_new_tokens"}


def test_request_id_sequence():
    """request_id is a zero-based sequential integer."""
    reqs = generate_synthetic_requests(5, seed=1)
    ids = [req["request_id"] for req in reqs]
    assert ids == list(range(5))


def test_prompt_length_bounds():
    """Each prompt length is within [min_prompt_len, max_prompt_len]."""
    min_len, max_len = 15, 50
    reqs = generate_synthetic_requests(20, min_prompt_len=min_len, max_prompt_len=max_len, seed=99)
    for req in reqs:
        assert min_len <= len(req["prompt_ids"]) <= max_len


def test_reproducibility():
    """Same seed produces identical results; different seeds differ."""
    reqs_a = generate_synthetic_requests(5, seed=42)
    reqs_b = generate_synthetic_requests(5, seed=42)
    reqs_c = generate_synthetic_requests(5, seed=7)
    assert reqs_a[0]["prompt_ids"] == reqs_b[0]["prompt_ids"]
    assert reqs_a[0]["prompt_ids"] != reqs_c[0]["prompt_ids"]
