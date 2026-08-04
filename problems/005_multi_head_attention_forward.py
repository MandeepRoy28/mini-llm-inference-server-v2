"""
Problem 005 — Multi-Head Attention Forward Pass
================================================
Multi-head attention (MHA) allows the model to jointly attend to information
from different representation subspaces at different positions.  Each "head"
runs its own scaled dot-product attention on a lower-dimensional projection of
the queries, keys and values.

    MultiHead(x) = Concat(head_1, ..., head_h) W_o
    head_i       = Attention(x W_qi, x W_ki, x W_vi)

Difficulty : Medium-Hard
Tags        : attention, multi-head, projection, transformer
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_head_attention_forward(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the multi-head self-attention output for a batch of sequences.

    Steps:
        1. Project *x* with ``W_q``, ``W_k``, ``W_v`` to get Q, K, V each of
           shape ``(batch, seq, d_model)``.
        2. Split the last dimension into *n_heads* heads, reshaping to
           ``(batch, n_heads, seq, d_head)`` where ``d_head = d_model // n_heads``.
        3. Run :func:`scaled_dot_product_attention` on each head (you may call
           it as a helper or implement inline).
        4. Concatenate heads back to ``(batch, seq, d_model)``.
        5. Project through ``W_o`` and return.

    ``W_q``, ``W_k``, ``W_v``, ``W_o`` are weight matrices of shape
    ``(d_model, d_model)``.  No bias terms are required.

    Args:
        x (torch.Tensor): Input tensor of shape ``(batch, seq, d_model)``.
        W_q (torch.Tensor): Query projection weight, shape ``(d_model, d_model)``.
        W_k (torch.Tensor): Key projection weight, shape ``(d_model, d_model)``.
        W_v (torch.Tensor): Value projection weight, shape ``(d_model, d_model)``.
        W_o (torch.Tensor): Output projection weight, shape ``(d_model, d_model)``.
        n_heads (int): Number of attention heads.  Must evenly divide *d_model*.
        mask (torch.Tensor | None): Optional causal or padding mask broadcastable
            to ``(batch, n_heads, seq, seq)``.  ``True`` means masked out.

    Returns:
        torch.Tensor: Output of shape ``(batch, seq, d_model)``.

    Examples:
        >>> import torch
        >>> B, S, D, H = 2, 5, 8, 2
        >>> x = torch.randn(B, S, D)
        >>> W = torch.eye(D)  # identity projections
        >>> out = multi_head_attention_forward(x, W, W, W, W, n_heads=H)
        >>> out.shape
        torch.Size([2, 5, 8])

        >>> # Output dtype matches input dtype
        >>> out.dtype == x.dtype
        True

        >>> # With n_heads=1 and identity weights, output shape still correct
        >>> out1 = multi_head_attention_forward(x, W, W, W, W, n_heads=1)
        >>> out1.shape
        torch.Size([2, 5, 8])
    """
    raise NotImplementedError
