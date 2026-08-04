"""Tests for Problem 019 — Write KV to Cache."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.019_write_kv_to_cache")
    write_kv_to_cache = mod.write_kv_to_cache
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 019 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(batch=1, heads=2, max_len=8, d_k=4, layers=2):
    return [
        {
            "k": torch.zeros(batch, heads, max_len, d_k),
            "v": torch.zeros(batch, heads, max_len, d_k),
        }
        for _ in range(layers)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_none():
    """write_kv_to_cache must return None (in-place operation)."""
    cache = _make_cache()
    k = torch.ones(1, 2, 1, 4)
    v = torch.ones(1, 2, 1, 4)
    result = write_kv_to_cache(cache, layer_idx=0, step=0, k=k, v=v)
    assert result is None


def test_write_correct_position():
    """The written slice at step index must match the supplied tensor."""
    cache = _make_cache()
    k = torch.full((1, 2, 1, 4), 5.0)
    v = torch.full((1, 2, 1, 4), 7.0)
    write_kv_to_cache(cache, layer_idx=0, step=3, k=k, v=v)

    assert torch.allclose(cache[0]["k"][:, :, 3, :], k.squeeze(2))
    assert torch.allclose(cache[0]["v"][:, :, 3, :], v.squeeze(2))


def test_other_positions_untouched():
    """Positions other than `step` must remain zero."""
    cache = _make_cache(max_len=6)
    k = torch.ones(1, 2, 1, 4)
    v = torch.ones(1, 2, 1, 4)
    write_kv_to_cache(cache, layer_idx=0, step=2, k=k, v=v)

    for pos in [0, 1, 3, 4, 5]:
        assert cache[0]["k"][:, :, pos, :].sum().item() == 0.0
        assert cache[0]["v"][:, :, pos, :].sum().item() == 0.0


def test_correct_layer_targeted():
    """Only the targeted layer should be modified."""
    cache = _make_cache(layers=3)
    k = torch.ones(1, 2, 1, 4) * 9.0
    v = torch.ones(1, 2, 1, 4) * 9.0
    write_kv_to_cache(cache, layer_idx=1, step=0, k=k, v=v)

    # Layer 0 and 2 must be untouched
    assert cache[0]["k"].sum().item() == 0.0
    assert cache[2]["k"].sum().item() == 0.0
    # Layer 1 must be written
    assert cache[1]["k"][:, :, 0, :].sum().item() != 0.0


def test_sequential_writes_accumulate():
    """Multiple writes to different steps must all be preserved."""
    cache = _make_cache(max_len=4)
    for step in range(3):
        k = torch.full((1, 2, 1, 4), float(step + 1))
        v = torch.full((1, 2, 1, 4), float(step + 1))
        write_kv_to_cache(cache, layer_idx=0, step=step, k=k, v=v)

    for step in range(3):
        expected_val = float(step + 1)
        assert torch.allclose(
            cache[0]["k"][:, :, step, :],
            torch.full((1, 2, 4), expected_val),
        )
