"""Tests for Problem 017 — Understand KV Recompute Cost."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.017_understand_kv_recompute_cost")
    benchmark_no_cache = mod.benchmark_no_cache
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 017 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 10


def _make_dummy_forward(next_token: int = 7):
    """Returns a model_forward that always predicts `next_token`."""
    def forward(input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        logits = torch.zeros(B, T, VOCAB)
        logits[:, :, next_token] = 100.0
        return logits
    return forward


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_return_type_and_keys():
    """Result must be a dict with exactly the three expected keys."""
    result = benchmark_no_cache(
        _make_dummy_forward(), torch.zeros(1, 4, dtype=torch.long), n_steps=3
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"total_time", "time_per_token", "n_tokens"}


def test_n_tokens_matches_n_steps():
    """n_tokens must equal the requested n_steps."""
    n_steps = 5
    result = benchmark_no_cache(
        _make_dummy_forward(), torch.zeros(1, 3, dtype=torch.long), n_steps=n_steps
    )
    assert result["n_tokens"] == n_steps


def test_time_per_token_consistency():
    """time_per_token should be total_time / n_steps."""
    n_steps = 4
    result = benchmark_no_cache(
        _make_dummy_forward(), torch.zeros(1, 2, dtype=torch.long), n_steps=n_steps
    )
    expected = result["total_time"] / n_steps
    assert abs(result["time_per_token"] - expected) < 1e-9


def test_total_time_is_non_negative():
    """Wall-clock time must be >= 0."""
    result = benchmark_no_cache(
        _make_dummy_forward(), torch.zeros(1, 5, dtype=torch.long), n_steps=2
    )
    assert result["total_time"] >= 0.0
    assert result["time_per_token"] >= 0.0


def test_model_called_n_steps_times():
    """The model_forward should be called exactly n_steps times."""
    call_count = {"n": 0}
    n_steps = 6

    def counting_forward(input_ids):
        call_count["n"] += 1
        B, T = input_ids.shape
        logits = torch.zeros(B, T, VOCAB)
        return logits

    benchmark_no_cache(
        counting_forward, torch.zeros(1, 2, dtype=torch.long), n_steps=n_steps
    )
    assert call_count["n"] == n_steps
