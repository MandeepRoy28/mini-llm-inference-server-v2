"""
Tests for Problem 030 — Read KV Cache via Block Table
"""

import importlib
import pytest
import torch

try:
    write_mod = importlib.import_module("solutions.029_write_kv_to_page")
    read_mod = importlib.import_module("solutions.030_read_kv_via_block_table")
    write_kv_to_page = write_mod.write_kv_to_page
    read_kv_via_block_table = read_mod.read_kv_via_block_table
    PAGE_SIZE = write_mod.PAGE_SIZE
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution modules 029/030 not found — write your solutions first.",
        allow_module_level=True,
    )

N_HEADS, D_K, N_LAYERS, N_PAGES = 2, 8, 2, 4


def _pool():
    return {
        "k_pages": torch.zeros(N_LAYERS, N_PAGES, PAGE_SIZE, N_HEADS, D_K),
        "v_pages": torch.zeros(N_LAYERS, N_PAGES, PAGE_SIZE, N_HEADS, D_K),
    }


def test_output_shape():
    pool = _pool()
    bt = {0: [0]}
    k_out, v_out = read_kv_via_block_table(pool, bt, seq_id=0, seq_len=3, layer_idx=0)
    assert k_out.shape == torch.Size([3, N_HEADS, D_K])
    assert v_out.shape == torch.Size([3, N_HEADS, D_K])


def test_round_trip_single_token():
    pool = _pool()
    bt = {0: [0]}
    k0 = torch.rand(N_HEADS, D_K)
    v0 = torch.rand(N_HEADS, D_K)
    write_kv_to_page(pool, bt, seq_id=0, token_pos=0, layer_idx=0, k=k0, v=v0)
    k_out, v_out = read_kv_via_block_table(pool, bt, seq_id=0, seq_len=1, layer_idx=0)
    assert torch.allclose(k_out[0], k0)
    assert torch.allclose(v_out[0], v0)


def test_round_trip_across_two_pages():
    pool = _pool()
    bt = {0: [0, 1]}
    # Write PAGE_SIZE + 1 tokens
    ks = [torch.rand(N_HEADS, D_K) for _ in range(PAGE_SIZE + 1)]
    vs = [torch.rand(N_HEADS, D_K) for _ in range(PAGE_SIZE + 1)]
    for pos, (k, v) in enumerate(zip(ks, vs)):
        write_kv_to_page(pool, bt, seq_id=0, token_pos=pos, layer_idx=0, k=k, v=v)
    k_out, v_out = read_kv_via_block_table(
        pool, bt, seq_id=0, seq_len=PAGE_SIZE + 1, layer_idx=0
    )
    for pos in range(PAGE_SIZE + 1):
        assert torch.allclose(k_out[pos], ks[pos])
        assert torch.allclose(v_out[pos], vs[pos])


def test_returns_tuple():
    pool = _pool()
    bt = {0: [0]}
    result = read_kv_via_block_table(pool, bt, seq_id=0, seq_len=1, layer_idx=0)
    assert isinstance(result, tuple)
    assert len(result) == 2
