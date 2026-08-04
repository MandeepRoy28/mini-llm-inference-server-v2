"""
Problem 032 — Attention with Paged KV Cache
Part 4: Paged Attention
"""

import math

import torch


def attention_with_paged_kv(
    q: torch.Tensor,
    pool: dict,
    block_table: dict,
    seq_id: int,
    seq_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention using K and V gathered from the page pool.

    This function is the core inference step in Paged Attention. Rather than
    maintaining a single contiguous KV buffer per sequence, it reads K and V
    on-the-fly from potentially non-contiguous physical pages via the block
    table, then performs standard scaled dot-product attention.

    Steps:
        1. Gather K, V from the paged pool: shape (seq_len, n_heads, d_k).
        2. For each head h, compute:
               scores[h] = q[h] @ K[:, h, :].T  / sqrt(d_k)  # (seq_len,)
               weights[h] = softmax(scores[h])               # (seq_len,)
               out[h]     = weights[h] @ V[:, h, :]           # (d_k,)
        3. Return out of shape (n_heads, d_k).

    Args:
        q (torch.Tensor): Query vector for the current token, shape (n_heads, d_k).
        pool (dict): Page pool with 'k_pages' and 'v_pages' tensors of shape
            (n_layers, n_pages, page_size, n_heads, d_k).
        block_table (dict): Mapping sequence_id -> list of physical page IDs.
        seq_id (int): ID of the sequence being decoded.
        seq_len (int): Number of tokens in the KV cache to attend over.
        layer_idx (int): Transformer layer index.

    Returns:
        torch.Tensor: Attended output of shape (n_heads, d_k).

    Examples:
        >>> out = attention_with_paged_kv(q, pool, bt, seq_id=0, seq_len=5, layer_idx=0)
        >>> out.shape
        torch.Size([n_heads, d_k])

        >>> # Output dtype should match pool dtype
        >>> out.dtype == pool['k_pages'].dtype
        True

        >>> # With seq_len=1 and q matching the single stored key, weights are 1.0
        >>> out = attention_with_paged_kv(q, pool, bt, seq_id=0, seq_len=1, layer_idx=0)
        >>> torch.allclose(out, v_stored)
        True
    """
    raise NotImplementedError
