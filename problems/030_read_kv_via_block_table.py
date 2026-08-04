"""
Problem 030 — Read KV Cache via Block Table
Part 4: Paged Attention
"""

import torch


def read_kv_via_block_table(
    pool: dict,
    block_table: dict,
    seq_id: int,
    seq_len: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather the full key and value tensors for a sequence from the paged pool.

    Sequences in Paged Attention may span multiple non-contiguous physical
    pages. This function reconstructs a contiguous (seq_len, n_heads, d_k)
    view by iterating over every token position 0..seq_len-1, computing its
    page and slot, and copying the stored vector into the output tensor.

    Algorithm sketch:
        For token_pos in range(seq_len):
            page_idx  = token_pos // PAGE_SIZE
            slot      = token_pos  % PAGE_SIZE
            page_id   = block_table[seq_id][page_idx]
            k_out[token_pos] = pool['k_pages'][layer_idx, page_id, slot]
            v_out[token_pos] = pool['v_pages'][layer_idx, page_id, slot]

    Args:
        pool (dict): Page pool with 'k_pages' and 'v_pages' tensors of shape
            (n_layers, n_pages, page_size, n_heads, d_k).
        block_table (dict): Mapping sequence_id -> list of physical page IDs.
        seq_id (int): ID of the sequence to read.
        seq_len (int): Number of tokens to read (must be <= total allocated
            slots for this sequence).
        layer_idx (int): Transformer layer index.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A pair (k, v) where each tensor has
            shape (seq_len, n_heads, d_k).

    Examples:
        >>> k, v = read_kv_via_block_table(pool, bt, seq_id=0, seq_len=3, layer_idx=0)
        >>> k.shape
        torch.Size([3, n_heads, d_k])

        >>> # Verify round-trip: write then read back
        >>> write_kv_to_page(pool, bt, 0, 0, 0, k0, v0)
        >>> k_out, _ = read_kv_via_block_table(pool, bt, 0, 1, 0)
        >>> torch.allclose(k_out[0], k0)
        True

        >>> # Reading a sequence spanning two pages
        >>> k, v = read_kv_via_block_table(pool, bt, seq_id=1, seq_len=20, layer_idx=1)
        >>> k.shape
        torch.Size([20, n_heads, d_k])
    """
    raise NotImplementedError
