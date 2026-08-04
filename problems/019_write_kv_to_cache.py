"""
Problem 019 — Write KV to Cache
Part of the DeepML-style LLM Inference Server project.
"""

import torch


def write_kv_to_cache(
    cache: list[dict],
    layer_idx: int,
    step: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Write one step's K and V tensors into the pre-allocated KV cache.

    After the model computes the key and value projections for the current
    token, those results must be stored in the cache so that future decode
    steps can attend over the full history without recomputing earlier tokens.

    This function modifies ``cache`` **in-place** — it returns ``None``.

    Args:
        cache: List of per-layer dicts as produced by
            ``allocate_kv_cache_buffers``.  Each dict has keys ``'k'`` and
            ``'v'`` with tensors of shape
            ``(batch_size, n_heads, max_seq_len, d_k)``.
        layer_idx: Index of the transformer layer whose K/V to update
            (0-based).
        step: Position index in the sequence at which to write (0-based).
            ``cache[layer_idx]['k'][:, :, step, :]`` is the target slice.
        k: Key tensor for the current step, shape
            ``(batch_size, n_heads, 1, d_k)``.
        v: Value tensor for the current step, shape
            ``(batch_size, n_heads, 1, d_k)``.

    Returns:
        None.  The cache is mutated in-place.

    Examples:
        >>> import torch
        >>> from problems.018_allocate_kv_cache_buffers import allocate_kv_cache_buffers
        >>> cache = allocate_kv_cache_buffers(1, 2, 10, 4, 1)
        >>> k = torch.ones(1, 2, 1, 4)
        >>> v = torch.full((1, 2, 1, 4), 2.0)
        >>> write_kv_to_cache(cache, layer_idx=0, step=0, k=k, v=v)
        >>> cache[0]['k'][0, 0, 0, :]  # should be all 1.0
        tensor([1., 1., 1., 1.])
        >>> cache[0]['v'][0, 0, 0, :]  # should be all 2.0
        tensor([2., 2., 2., 2.])
        >>> cache[0]['k'][0, 0, 1, :].sum().item()  # step=1 untouched
        0.0
    """
    raise NotImplementedError
