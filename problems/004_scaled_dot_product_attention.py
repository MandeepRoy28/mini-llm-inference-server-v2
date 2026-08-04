"""
Problem 004 — Scaled Dot-Product Attention
===========================================
Scaled dot-product attention is the core computational primitive of every
Transformer.  It computes a weighted sum of *values* where the weights are
derived from the compatibility between *queries* and *keys*.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) · V

Difficulty : Medium
Tags        : attention, softmax, transformer, linear algebra
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Given query, key and value tensors, computes

        output = softmax( Q @ K.T / sqrt(d_k) + additive_mask ) @ V

    where the *additive_mask* is ``-1e9`` at positions where ``mask`` is
    ``True`` and ``0`` elsewhere.  The masking is applied **before** softmax
    so that masked positions receive ~0 attention weight.

    Args:
        Q (torch.Tensor): Query tensor of shape ``(batch, seq_q, d_k)``.
        K (torch.Tensor): Key tensor of shape ``(batch, seq_k, d_k)``.
        V (torch.Tensor): Value tensor of shape ``(batch, seq_k, d_v)``.
            Typically ``d_v == d_k``.
        mask (torch.Tensor | None): Optional boolean tensor broadcastable to
            shape ``(batch, seq_q, seq_k)``.  Where ``True``, the logit is
            set to ``-1e9`` before softmax.  ``None`` means no masking.

    Returns:
        torch.Tensor: Output tensor of shape ``(batch, seq_q, d_v)``.

    Examples:
        >>> import torch
        >>> B, S, D = 1, 4, 8
        >>> Q = K = V = torch.randn(B, S, D)
        >>> out = scaled_dot_product_attention(Q, K, V)
        >>> out.shape
        torch.Size([1, 4, 8])

        >>> # With a causal mask the output still has the correct shape
        >>> mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        >>> out_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
        >>> out_masked.shape
        torch.Size([1, 4, 8])

        >>> # Self-attention with d_k=1, no mask: output is weighted mean of V
        >>> Q2 = torch.zeros(1, 2, 1)
        >>> K2 = torch.zeros(1, 2, 1)
        >>> V2 = torch.tensor([[[1.0], [3.0]]])
        >>> scaled_dot_product_attention(Q2, K2, V2)
        tensor([[[2.],
                 [2.]]])
    """
    raise NotImplementedError
