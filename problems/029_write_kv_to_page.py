"""
Problem 029 — Write KV Vectors to a Page
Part 4: Paged Attention
"""

import torch

PAGE_SIZE = 16  # Must match the page size used when the pool was allocated


def write_kv_to_page(
    pool: dict,
    block_table: dict,
    seq_id: int,
    token_pos: int,
    layer_idx: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """
    Write the key and value vectors for one token into the paged KV cache.

    Given a global token position `token_pos` within a sequence, determine
    which physical page stores that position and which slot within the page
    corresponds to it:

        page_idx = token_pos // PAGE_SIZE   (index into block_table[seq_id])
        slot     = token_pos  % PAGE_SIZE   (slot within the page)

    The page ID is then looked up via block_table[seq_id][page_idx], and the
    tensors are written into pool['k_pages'] and pool['v_pages'] at the
    appropriate location. Modifies the pool tensors **in-place**.

    Args:
        pool (dict): Page pool with 'k_pages' and 'v_pages' tensors of shape
            (n_layers, n_pages, page_size, n_heads, d_k).
        block_table (dict): Mapping sequence_id -> list of physical page IDs.
        seq_id (int): ID of the sequence being updated.
        token_pos (int): 0-based position of the token within the sequence.
        layer_idx (int): Transformer layer index (selects axis 0 in pool).
        k (torch.Tensor): Key vector of shape (n_heads, d_k).
        v (torch.Tensor): Value vector of shape (n_heads, d_k).

    Returns:
        None. The pool tensors are updated in-place.

    Examples:
        >>> # Write token at position 0 (page 0, slot 0)
        >>> write_kv_to_page(pool, bt, seq_id=0, token_pos=0, layer_idx=0, k=k0, v=v0)

        >>> # Write token at position 16 (page 1, slot 0 — crosses page boundary)
        >>> write_kv_to_page(pool, bt, seq_id=0, token_pos=16, layer_idx=0, k=k16, v=v16)

        >>> # Write token at position 17 (page 1, slot 1)
        >>> write_kv_to_page(pool, bt, seq_id=0, token_pos=17, layer_idx=2, k=k17, v=v17)
    """
    raise NotImplementedError
