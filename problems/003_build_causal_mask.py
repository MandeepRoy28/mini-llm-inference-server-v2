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
    """Return an upper-triangular boolean mask of shape (seq_len, seq_len).

    Entry ``mask[i, j]`` is ``True`` when position *i* is **not allowed** to
    attend to position *j*.  Under causal (left-to-right) masking, position
    *i* may only attend to positions ``j <= i``, so every position strictly
    above the main diagonal is masked.

    The mask is designed to be used as an *additive* mask: before applying
    softmax, add ``-1e9`` (or ``-inf``) where the mask is ``True``.

    Args:
        seq_len (int): Length of the sequence.  Must be a positive integer.

    Returns:
        torch.Tensor: Boolean tensor of shape ``(seq_len, seq_len)``.
            ``mask[i, j] == True`` iff ``j > i`` (upper triangle, diagonal
            excluded).

    Examples:
        >>> build_causal_mask(3)
        tensor([[False,  True,  True],
                [False, False,  True],
                [False, False, False]])

        >>> build_causal_mask(1)
        tensor([[False]])

        >>> build_causal_mask(4).shape
        torch.Size([4, 4])
    """
    raise NotImplementedError
