"""
Problem 007 — GPT-2 Style Transformer Block
============================================
A complete GPT-2 transformer block combines multi-head self-attention and a
feed-forward network, both wrapped with *pre-Layer Normalisation* and residual
connections.

    x = x + MHA( LayerNorm(x) )
    x = x + FFN( LayerNorm(x) )

This *pre-norm* variant (used in GPT-2) is more stable to train than the
original post-norm (used in the vanilla "Attention Is All You Need" paper).

Difficulty : Medium-Hard
Tags        : transformer block, layer norm, residual, GPT-2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def transformer_block(
    x: torch.Tensor,
    attn_params: dict,
    ffn_params: dict,
    n_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one GPT-2-style transformer block (pre-LayerNorm + residual).

    The block performs two sub-layers in sequence:

    **Sub-layer 1 — Multi-Head Self-Attention**::

        x = x + MHA( LayerNorm(x, gamma_1, beta_1),
                      W_q, W_k, W_v, W_o, n_heads, mask )

    **Sub-layer 2 — Feed-Forward Network**::

        x = x + FFN( LayerNorm(x, gamma_2, beta_2),
                      W1, b1, W2, b2 )

    You should reuse your implementations of
    :func:`multi_head_attention_forward` and :func:`feed_forward_block`
    (or implement them inline).

    Args:
        x (torch.Tensor): Input of shape ``(batch, seq, d_model)``.
        attn_params (dict): Dictionary with keys:

            * ``'W_q'``, ``'W_k'``, ``'W_v'``, ``'W_o'`` — weight matrices,
              each of shape ``(d_model, d_model)``.
            * ``'gamma_1'``, ``'beta_1'`` — LayerNorm scale/bias, shape
              ``(d_model,)``.

        ffn_params (dict): Dictionary with keys:

            * ``'W1'``, ``'b1'`` — first linear layer, shapes
              ``(d_model, d_ff)`` and ``(d_ff,)``.
            * ``'W2'``, ``'b2'`` — second linear layer, shapes
              ``(d_ff, d_model)`` and ``(d_model,)``.
            * ``'gamma_2'``, ``'beta_2'`` — LayerNorm scale/bias, shape
              ``(d_model,)``.

        n_heads (int): Number of attention heads.
        mask (torch.Tensor | None): Optional causal mask.

    Returns:
        torch.Tensor: Output of shape ``(batch, seq, d_model)``.

    Examples:
        >>> import torch
        >>> B, S, D, H, D_FF = 1, 4, 8, 2, 32
        >>> x = torch.randn(B, S, D)
        >>> I = torch.eye(D)
        >>> attn_params = {
        ...     'W_q': I, 'W_k': I, 'W_v': I, 'W_o': I,
        ...     'gamma_1': torch.ones(D), 'beta_1': torch.zeros(D),
        ... }
        >>> ffn_params = {
        ...     'W1': torch.zeros(D, D_FF), 'b1': torch.zeros(D_FF),
        ...     'W2': torch.zeros(D_FF, D), 'b2': torch.zeros(D),
        ...     'gamma_2': torch.ones(D), 'beta_2': torch.zeros(D),
        ... }
        >>> out = transformer_block(x, attn_params, ffn_params, n_heads=H)
        >>> out.shape
        torch.Size([1, 4, 8])

        >>> # Residual: with all-zero FFN weights, output == x + attn_output
        >>> # (output shape always matches input shape)
        >>> out.shape == x.shape
        True

        >>> # Output dtype preserved
        >>> out.dtype == x.dtype
        True
    """
    raise NotImplementedError
