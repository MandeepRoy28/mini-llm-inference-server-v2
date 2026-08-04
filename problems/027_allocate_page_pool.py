"""
Problem 027 — Allocate Page Pool
Part 4: Paged Attention
"""

import torch


def allocate_page_pool(
    n_pages: int,
    page_size: int,
    n_heads: int,
    d_k: int,
    n_layers: int,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Pre-allocate a contiguous pool of KV-cache pages for all transformer layers.

    Instead of growing memory on demand, Paged Attention reserves a fixed pool
    of pages upfront. Each page can hold `page_size` token positions. Pages are
    shared across all layers, and the pool tracks which pages are currently
    available via the `free_pages` list.

    Tensor layout — both k_pages and v_pages have shape:
        (n_layers, n_pages, page_size, n_heads, d_k)

    This layout groups all layers in the outermost dimension so a single index
    along axis 0 selects the entire KV pool for one layer.

    Args:
        n_pages (int): Total number of pages in the pool.
        page_size (int): Number of token slots per page (e.g. 16).
        n_heads (int): Number of attention heads.
        d_k (int): Dimension of each head's key/value vectors.
        n_layers (int): Number of transformer layers.
        dtype (torch.dtype, optional): Data type of the tensors.
            Defaults to torch.float32.

    Returns:
        dict: A dictionary with the following keys:
            - 'k_pages' (torch.Tensor): Key cache tensor of shape
              (n_layers, n_pages, page_size, n_heads, d_k), initialised to zeros.
            - 'v_pages' (torch.Tensor): Value cache tensor with the same shape,
              initialised to zeros.
            - 'free_pages' (list[int]): List of all available page IDs,
              i.e. list(range(n_pages)).

    Examples:
        >>> pool = allocate_page_pool(10, 16, 4, 64, 6)
        >>> pool['k_pages'].shape
        torch.Size([6, 10, 16, 4, 64])

        >>> pool['v_pages'].shape
        torch.Size([6, 10, 16, 4, 64])

        >>> len(pool['free_pages'])
        10
    """
    raise NotImplementedError
