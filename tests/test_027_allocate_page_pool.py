"""
Tests for Problem 027 — Allocate Page Pool
"""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.027_allocate_page_pool")
    allocate_page_pool = mod.allocate_page_pool
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 027_allocate_page_pool not found — write your solution first.",
        allow_module_level=True,
    )


def test_tensor_shapes():
    pool = allocate_page_pool(n_pages=10, page_size=16, n_heads=4, d_k=64, n_layers=6)
    expected = torch.Size([6, 10, 16, 4, 64])
    assert pool["k_pages"].shape == expected
    assert pool["v_pages"].shape == expected


def test_free_pages_list():
    pool = allocate_page_pool(n_pages=8, page_size=16, n_heads=2, d_k=32, n_layers=2)
    assert pool["free_pages"] == list(range(8))


def test_tensors_initialised_to_zero():
    pool = allocate_page_pool(n_pages=4, page_size=16, n_heads=2, d_k=16, n_layers=2)
    assert torch.all(pool["k_pages"] == 0)
    assert torch.all(pool["v_pages"] == 0)


def test_dtype_respected():
    pool = allocate_page_pool(4, 16, 2, 16, 2, dtype=torch.float16)
    assert pool["k_pages"].dtype == torch.float16
    assert pool["v_pages"].dtype == torch.float16


def test_return_keys_present():
    pool = allocate_page_pool(2, 8, 1, 8, 1)
    assert set(pool.keys()) == {"k_pages", "v_pages", "free_pages"}
