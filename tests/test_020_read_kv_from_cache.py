"""Tests for Problem 020 — Read KV from Cache."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.020_read_kv_from_cache")
    read_kv_from_cache = mod.read_kv_from_cache
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 020 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(batch=1, heads=2, max_len=8, d_k=4, n_layers=2):
    return [
        {
            "k": torch.arange(batch * heads * max_len * d_k, dtype=torch.float32).reshape(
                batch, heads, max_len, d_k
            ),
            "v": torch.zeros(batch, heads, max_len, d_k),
        }
        for _ in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_return_type_is_tuple_of_two_tensors():
    """Function must return a (k, v) tuple."""
    cache = _make_cache()
    result = read_kv_from_cache(cache, layer_idx=0, current_len=3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)


def test_output_shape_matches_current_len():
    """Returned tensors must have shape (B, H, current_len, d_k)."""
    B, H, max_len, dk = 1, 2, 8, 4
    cache = _make_cache(batch=B, heads=H, max_len=max_len, d_k=dk)
    for clen in [1, 4, 8]:
        k, v = read_kv_from_cache(cache, layer_idx=0, current_len=clen)
        assert k.shape == torch.Size([B, H, clen, dk])
        assert v.shape == torch.Size([B, H, clen, dk])


def test_correct_layer_read():
    """Reading different layers should yield different data."""
    cache = [
        {"k": torch.ones(1, 2, 6, 4) * i, "v": torch.ones(1, 2, 6, 4) * i}
        for i in range(3)
    ]
    k0, _ = read_kv_from_cache(cache, layer_idx=0, current_len=3)
    k1, _ = read_kv_from_cache(cache, layer_idx=1, current_len=3)
    assert k0.sum().item() == pytest.approx(0.0)
    assert k1.sum().item() != 0.0


def test_values_are_correct_slice():
    """The returned k tensor must be a slice of the cached k data."""
    cache = _make_cache(batch=1, heads=1, max_len=6, d_k=2)
    # Manually fill known values
    cache[0]["k"] = torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2)
    k, _ = read_kv_from_cache(cache, layer_idx=0, current_len=3)
    # First 3 positions: [[0,1],[2,3],[4,5]]
    expected = torch.tensor([[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]])
    assert torch.allclose(k, expected)
