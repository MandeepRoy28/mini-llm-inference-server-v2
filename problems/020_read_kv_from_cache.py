"""
Problem 020 — Read KV from Cache
Part of the DeepML-style LLM Inference Server project.
"""

import torch


def read_kv_from_cache(
    cache: list[dict],
    layer_idx: int,
    current_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the accumulated K and V tensors for a layer up to the current step.

    During the decode phase, attention for the new token must cover the full
    context: all prompt positions plus every previously generated token.  This
    function slices the pre-allocated cache buffer to return only the valid
    (already written) portion.

    Args:
        cache: List of per-layer dicts as produced by
            ``allocate_kv_cache_buffers`` and populated by
            ``write_kv_to_cache``.  Each dict has keys ``'k'`` and ``'v'``
            with tensors of shape
            ``(batch_size, n_heads, max_seq_len, d_k)``.
        layer_idx: Index of the transformer layer to read from (0-based).
        current_len: Number of valid positions already written into the cache
            (i.e. the slice ``[:, :, :current_len, :]`` is returned).

    Returns:
        A tuple ``(k, v)`` where each tensor has shape
        ``(batch_size, n_heads, current_len, d_k)``.

    Examples:
        >>> import torch
        >>> cache = [{'k': torch.arange(24).reshape(1, 2, 6, 2).float(),
        ...           'v': torch.zeros(1, 2, 6, 2)}]
        >>> k, v = read_kv_from_cache(cache, layer_idx=0, current_len=3)
        >>> k.shape
        torch.Size([1, 2, 3, 2])
        >>> k, v = read_kv_from_cache(cache, layer_idx=0, current_len=6)
        >>> k.shape
        torch.Size([1, 2, 6, 2])
        >>> k, v = read_kv_from_cache(cache, layer_idx=0, current_len=1)
        >>> k.shape
        torch.Size([1, 2, 1, 2])
    """
    raise NotImplementedError
