"""
Tests for Problem 032 — Attention with Paged KV Cache
"""

import importlib
import math
import pytest
import torch

try:
    write_mod = importlib.import_module("solutions.029_write_kv_to_page")
    attn_mod = importlib.import_module("solutions.032_attention_with_paged_kv")
    write_kv_to_page = write_mod.write_kv_to_page
    attention_with_paged_kv = attn_mod.attention_with_paged_kv
    PAGE_SIZE = write_mod.PAGE_SIZE
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "Solution modules 029/032 not found — write your solutions first.",
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
    # Write one token so seq_len=1 is valid
    k0 = torch.rand(N_HEADS, D_K)
    v0 = torch.rand(N_HEADS, D_K)
    write_kv_to_page(pool, bt, seq_id=0, token_pos=0, layer_idx=0, k=k0, v=v0)
    q = torch.rand(N_HEADS, D_K)
    out = attention_with_paged_kv(q, pool, bt, seq_id=0, seq_len=1, layer_idx=0)
    assert out.shape == torch.Size([N_HEADS, D_K])


def test_output_dtype_matches_pool():
    pool = _pool()
    bt = {0: [0]}
    write_kv_to_page(
        pool, bt, 0, 0, 0,
        torch.rand(N_HEADS, D_K),
        torch.rand(N_HEADS, D_K),
    )
    q = torch.rand(N_HEADS, D_K)
    out = attention_with_paged_kv(q, pool, bt, seq_id=0, seq_len=1, layer_idx=0)
    assert out.dtype == pool["k_pages"].dtype


def test_single_token_output_equals_stored_value():
    """With one KV entry, softmax weight is 1.0 so output == stored V."""
    pool = _pool()
    bt = {0: [0]}
    k0 = torch.rand(N_HEADS, D_K)
    v0 = torch.rand(N_HEADS, D_K)
    write_kv_to_page(pool, bt, seq_id=0, token_pos=0, layer_idx=0, k=k0, v=v0)
    q = torch.rand(N_HEADS, D_K)
    out = attention_with_paged_kv(q, pool, bt, seq_id=0, seq_len=1, layer_idx=0)
    assert torch.allclose(out, v0, atol=1e-5)


def test_output_is_weighted_combination_of_values():
    """Verify attention output is a convex combination of the stored values."""
    pool = _pool()
    bt = {0: [0]}
    seq_len = 3
    vs = [torch.rand(N_HEADS, D_K) for _ in range(seq_len)]
    for pos, v in enumerate(vs):
        write_kv_to_page(
            pool, bt, seq_id=0, token_pos=pos, layer_idx=0,
            k=torch.rand(N_HEADS, D_K), v=v
        )
    q = torch.rand(N_HEADS, D_K)
    out = attention_with_paged_kv(q, pool, bt, seq_id=0, seq_len=seq_len, layer_idx=0)
    # Output must lie within the convex hull of stored values (element-wise min/max)
    v_stack = torch.stack(vs)  # (seq_len, N_HEADS, D_K)
    assert torch.all(out >= v_stack.min(dim=0).values - 1e-5)
    assert torch.all(out <= v_stack.max(dim=0).values + 1e-5)
