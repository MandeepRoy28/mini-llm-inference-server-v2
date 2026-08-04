"""
Problem 006 — Position-wise Feed-Forward Block
===============================================
Every Transformer block contains a two-layer fully-connected network applied
independently to each position.  GPT-2 uses a GELU activation between the
two linear layers.

    FFN(x) = GELU( x W_1 + b_1 ) W_2 + b_2

The hidden dimension is typically 4× the model dimension (e.g. d_model=768,
d_ff=3072 in GPT-2 small).

Difficulty : Easy-Medium
Tags        : feed-forward, GELU, MLP, transformer
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def feed_forward_block(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Apply a two-layer feed-forward network with GELU activation.

    Computes ``GELU(x @ W1 + b1) @ W2 + b2``.  This is the FFN sub-layer of
    a GPT-2 transformer block, **without** the surrounding residual connection
    or layer normalisation — those are handled by the enclosing block.

    Use ``torch.nn.functional.gelu`` for the activation.

    Args:
        x (torch.Tensor): Input of shape ``(batch, seq, d_model)``.
        W1 (torch.Tensor): First linear weight of shape ``(d_model, d_ff)``.
        b1 (torch.Tensor): First linear bias of shape ``(d_ff,)``.
        W2 (torch.Tensor): Second linear weight of shape ``(d_ff, d_model)``.
        b2 (torch.Tensor): Second linear bias of shape ``(d_model,)``.

    Returns:
        torch.Tensor: Output of shape ``(batch, seq, d_model)``.  Same shape
            as the input — the feed-forward block is position-wise and
            dimension-preserving overall (``d_model`` in → ``d_model`` out).

    Examples:
        >>> import torch, torch.nn.functional as F
        >>> x  = torch.ones(1, 3, 4)   # batch=1, seq=3, d_model=4
        >>> W1 = torch.eye(4, 8)       # d_ff=8
        >>> b1 = torch.zeros(8)
        >>> W2 = torch.eye(8, 4)       # back to d_model=4
        >>> b2 = torch.zeros(4)
        >>> out = feed_forward_block(x, W1, b1, W2, b2)
        >>> out.shape
        torch.Size([1, 3, 4])

        >>> # Zero input with zero bias → output equals gelu(0) * W2 columns summed
        >>> x_zero = torch.zeros(1, 1, 4)
        >>> out_zero = feed_forward_block(x_zero, W1, b1, W2, b2)
        >>> torch.allclose(out_zero, torch.zeros(1, 1, 4))
        True

        >>> # Output dtype preserved
        >>> out.dtype == x.dtype
        True
    """
    raise NotImplementedError
