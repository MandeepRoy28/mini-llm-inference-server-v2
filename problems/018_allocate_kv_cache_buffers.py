"""
Problem 018 — Allocate KV Cache Buffers
Part of the DeepML-style LLM Inference Server project.
"""

import torch


def allocate_kv_cache_buffers(
    batch_size: int,
    n_heads: int,
    max_seq_len: int,
    d_k: int,
    n_layers: int,
    dtype=torch.float32,
) -> list[dict]:
    """Pre-allocate zero-filled K and V tensors for every transformer layer.

    During inference it is more efficient to allocate one large buffer upfront
    and write individual token K/V pairs into it at each step than to
    concatenate growing tensors.  This function returns such a buffer
    structure: a list with one dict per layer, each dict holding two tensors
    ('k' and 'v') that span the full maximum sequence length.

    Args:
        batch_size: Number of sequences processed in parallel.
        n_heads: Number of attention heads in the model.
        max_seq_len: Maximum number of tokens (prompt + generated) that the
            cache must accommodate.
        d_k: Per-head key/value dimension (``d_model // n_heads``).
        n_layers: Number of transformer layers (one cache entry per layer).
        dtype: PyTorch dtype for the allocated tensors (default
            ``torch.float32``).

    Returns:
        A list of length ``n_layers``.  Each element is a dict with keys:

        * ``'k'``: zero tensor of shape ``(batch_size, n_heads, max_seq_len, d_k)``.
        * ``'v'``: zero tensor of shape ``(batch_size, n_heads, max_seq_len, d_k)``.

    Examples:
        >>> import torch
        >>> cache = allocate_kv_cache_buffers(1, 4, 128, 64, 6)
        >>> len(cache)
        6
        >>> cache[0]['k'].shape
        torch.Size([1, 4, 128, 64])
        >>> cache[0]['v'].shape
        torch.Size([1, 4, 128, 64])
        >>> cache[0]['k'].sum().item()
        0.0
        >>> cache = allocate_kv_cache_buffers(2, 8, 512, 32, 12, dtype=torch.float16)
        >>> cache[0]['k'].dtype
        torch.float16
    """
    raise NotImplementedError
