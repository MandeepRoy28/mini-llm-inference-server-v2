"""
Tests for Problem 029 — Write KV Vectors to a Page
"""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.029_write_kv_to_page")
    write_kv_to_page = mod.write_kv_to_page
    PAGE_SIZE = mod.PAGE_SIZE
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution module 029_write_kv_to_page not found — write your solution first.",
        allow_module_level=True,
    )

N_HEADS, D_K, N_LAYERS, N_PAGES = 2, 8, 3, 4


def _pool():
    return {
        "k_pages": torch.zeros(N_LAYERS, N_PAGES, PAGE_SIZE, N_HEADS, D_K),
        "v_pages": torch.zeros(N_LAYERS, N_PAGES, PAGE_SIZE, N_HEADS, D_K),
    }


def _bt():
    # Sequence 0 owns pages [0, 1]; sequence 1 owns page [2]
    return {0: [0, 1], 1: [2]}


def test_write_token_pos_zero():
    pool, bt = _pool(), _bt()
    k = torch.ones(N_HEADS, D_K)
    v = torch.ones(N_HEADS, D_K) * 2
    write_kv_to_page(pool, bt, seq_id=0, token_pos=0, layer_idx=0, k=k, v=v)
    assert torch.allclose(pool["k_pages"][0, 0, 0], k)
    assert torch.allclose(pool["v_pages"][0, 0, 0], v)


def test_write_crosses_page_boundary():
    pool, bt = _pool(), _bt()
    k = torch.full((N_HEADS, D_K), 3.0)
    v = torch.full((N_HEADS, D_K), 4.0)
    # token_pos == PAGE_SIZE means page_idx=1, slot=0
    write_kv_to_page(pool, bt, seq_id=0, token_pos=PAGE_SIZE, layer_idx=1, k=k, v=v)
    page_id = bt[0][1]  # second page assigned to seq 0
    assert torch.allclose(pool["k_pages"][1, page_id, 0], k)
    assert torch.allclose(pool["v_pages"][1, page_id, 0], v)


def test_write_different_layer():
    pool, bt = _pool(), _bt()
    k = torch.rand(N_HEADS, D_K)
    v = torch.rand(N_HEADS, D_K)
    write_kv_to_page(pool, bt, seq_id=0, token_pos=1, layer_idx=2, k=k, v=v)
    # layer 0 and layer 1 should be untouched
    assert torch.all(pool["k_pages"][0] == 0)
    assert torch.all(pool["k_pages"][1] == 0)


def test_returns_none():
    pool, bt = _pool(), _bt()
    k = torch.zeros(N_HEADS, D_K)
    v = torch.zeros(N_HEADS, D_K)
    result = write_kv_to_page(pool, bt, seq_id=1, token_pos=0, layer_idx=0, k=k, v=v)
    assert result is None
