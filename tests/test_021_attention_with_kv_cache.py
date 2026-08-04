"""Tests for Problem 021 — Attention with KV Cache."""

import importlib
import math
import pytest
import torch
import torch.nn.functional as F

try:
    mod = importlib.import_module("solutions.021_attention_with_kv_cache")
    attention_with_kv_cache = mod.attention_with_kv_cache
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 021 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(B=1, H=2, max_len=8, dk=4, n_layers=1, fill_value=None):
    if fill_value is not None:
        return [
            {
                "k": torch.full((B, H, max_len, dk), fill_value),
                "v": torch.full((B, H, max_len, dk), fill_value),
            }
            for _ in range(n_layers)
        ]
    return [
        {
            "k": torch.randn(B, H, max_len, dk),
            "v": torch.randn(B, H, max_len, dk),
        }
        for _ in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape():
    """Output must have shape (B, H, 1, d_k)."""
    B, H, dk = 1, 2, 4
    cache = _make_cache(B=B, H=H, dk=dk)
    q = torch.randn(B, H, 1, dk)
    out = attention_with_kv_cache(q, cache, layer_idx=0, current_len=5)
    assert out.shape == torch.Size([B, H, 1, dk])


def test_output_is_finite():
    """All output values must be finite (no NaN or Inf)."""
    cache = _make_cache()
    q = torch.randn(1, 2, 1, 4)
    out = attention_with_kv_cache(q, cache, layer_idx=0, current_len=4)
    assert torch.isfinite(out).all().item()


def test_single_position_returns_value_vector():
    """With one cached position, output must equal that value vector."""
    dk = 4
    val = 3.0
    cache = [
        {
            "k": torch.ones(1, 1, 10, dk),   # uniform keys
            "v": torch.full((1, 1, 10, dk), val),
        }
    ]
    q = torch.ones(1, 1, 1, dk)
    out = attention_with_kv_cache(q, cache, layer_idx=0, current_len=1)
    assert torch.allclose(out, torch.full_like(out, val), atol=1e-5)


def test_attention_is_scaled():
    """Verify scaling by d_k: attention score should use 1/sqrt(d_k)."""
    # With two keys equal to q, both positions should get weight 0.5 each.
    B, H, dk = 1, 1, 4
    q = torch.ones(B, H, 1, dk)
    cache = [
        {
            "k": torch.ones(B, H, 4, dk),  # all keys = q
            "v": torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1).expand(B, H, 4, dk).clone(),
        }
    ]
    out = attention_with_kv_cache(q, cache, layer_idx=0, current_len=2)
    # Values at pos 0 and 1 are 0 and 1 respectively across dk dim;
    # with equal weights the result should be 0.5 everywhere
    expected = torch.full((B, H, 1, dk), 0.5)
    assert torch.allclose(out, expected, atol=1e-5)


def test_batch_size_preserved():
    """Larger batch sizes must produce output with correct batch dimension."""
    B, H, dk = 3, 4, 8
    cache = _make_cache(B=B, H=H, dk=dk)
    q = torch.randn(B, H, 1, dk)
    out = attention_with_kv_cache(q, cache, layer_idx=0, current_len=5)
    assert out.shape[0] == B
