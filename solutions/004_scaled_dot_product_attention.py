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
    
    batch, seq_q, d_k = Q.size()

    S = Q@K.transpose(-2,-1)
    S = S/(math.sqrt(d_k))
    
    if mask is not None :
        S = S.masked_fill(mask, -1e9)

    P = torch.softmax(S, dim=-1)
    O = P@V 

    return O

