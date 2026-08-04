"""Tests for Problem 018 — Allocate KV Cache Buffers."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.018_allocate_kv_cache_buffers")
    allocate_kv_cache_buffers = mod.allocate_kv_cache_buffers
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 018 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_list_length_equals_n_layers():
    """The returned list must have exactly n_layers entries."""
    cache = allocate_kv_cache_buffers(1, 4, 128, 64, 6)
    assert len(cache) == 6


def test_tensor_shapes():
    """Each layer dict must have 'k' and 'v' with the correct shape."""
    B, H, T, dk, L = 2, 8, 64, 32, 4
    cache = allocate_kv_cache_buffers(B, H, T, dk, L)
    expected_shape = torch.Size([B, H, T, dk])
    for layer in cache:
        assert layer["k"].shape == expected_shape
        assert layer["v"].shape == expected_shape


def test_tensors_initialised_to_zero():
    """All allocated tensors must be filled with zeros."""
    cache = allocate_kv_cache_buffers(1, 2, 16, 8, 3)
    for layer in cache:
        assert layer["k"].sum().item() == 0.0
        assert layer["v"].sum().item() == 0.0


def test_dtype_propagation():
    """dtype argument must be respected for all allocated tensors."""
    cache = allocate_kv_cache_buffers(1, 4, 32, 16, 2, dtype=torch.float16)
    for layer in cache:
        assert layer["k"].dtype == torch.float16
        assert layer["v"].dtype == torch.float16


def test_dict_keys_present():
    """Each element must be a dict with exactly 'k' and 'v' keys."""
    cache = allocate_kv_cache_buffers(1, 1, 10, 4, 1)
    assert isinstance(cache[0], dict)
    assert "k" in cache[0] and "v" in cache[0]
