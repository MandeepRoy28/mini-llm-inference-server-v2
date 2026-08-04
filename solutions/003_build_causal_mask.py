"""
Problem 003 — Build a Causal (Auto-Regressive) Attention Mask
==============================================================
Decoder-only transformers (GPT-style) must not allow any token to "look ahead"
at future tokens during self-attention.  This constraint is enforced by a
*causal mask*: a boolean matrix that marks which (query, key) pairs are
forbidden.

Difficulty : Easy-Medium
Tags        : attention, masking, transformer, linear algebra
"""
import torch

def build_causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, dtype=bool)

    for i in range(0, seq_len):
        for j in range(0, seq_len):
            if (j>i):
                mask[i][j] = True

    return mask
