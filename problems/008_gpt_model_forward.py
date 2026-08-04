"""
Problem 008 — Full GPT Forward Pass
=====================================
Putting it all together: a full GPT-2 forward pass takes a batch of token id
sequences and produces logit distributions over the vocabulary at each
position.

    logits = LayerNorm( Block_N( ... Block_1( wte[ids] + wpe[:T] ) ) ) @ wte.T

The *unembedding* matrix is the **transpose** of the token embedding matrix
(weight tying), matching the original GPT-2 architecture.

Difficulty : Hard
Tags        : GPT-2, full model, forward pass, weight tying, transformer
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gpt_model_forward(
    token_ids: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Run a full forward pass through a GPT-2 style language model.

    The forward pass proceeds as follows:

    1. **Token embedding**: look up each token id in ``params['wte']`` (shape
       ``(vocab_size, d_model)``).
    2. **Positional embedding**: look up positions ``0 … T-1`` in
       ``params['wpe']`` (shape ``(max_seq_len, d_model)``).
    3. **Add** token and positional embeddings → ``h`` of shape
       ``(batch, T, d_model)``.
    4. **N transformer blocks**: for each block *i*, apply
       :func:`transformer_block` with ``params['blocks'][i]`` split into
       ``attn_params`` and ``ffn_params``.
    5. **Final LayerNorm**: apply LayerNorm with ``params['ln_f_gamma']`` and
       ``params['ln_f_beta']``.
    6. **Unembed** (weight tying): compute ``h @ params['wte'].T`` to get
       logits of shape ``(batch, T, vocab_size)``.

    ``params`` layout::

        params = {
            'wte':        Tensor (vocab_size, d_model),   # token embeddings
            'wpe':        Tensor (max_seq, d_model),      # positional embeddings
            'ln_f_gamma': Tensor (d_model,),              # final LN scale
            'ln_f_beta':  Tensor (d_model,),              # final LN bias
            'blocks': [   # list of N block dicts, each containing:
                {
                    # attn sub-layer
                    'W_q', 'W_k', 'W_v', 'W_o',    # (d_model, d_model)
                    'gamma_1', 'beta_1',             # (d_model,)
                    # ffn sub-layer
                    'W1', 'b1',                      # (d_model, d_ff), (d_ff,)
                    'W2', 'b2',                      # (d_ff, d_model), (d_model,)
                    'gamma_2', 'beta_2',             # (d_model,)
                    # number of heads (int)
                    'n_heads',
                },
                ...
            ]
        }

    Args:
        token_ids (torch.Tensor): Long tensor of shape ``(batch, seq_len)``
            containing token ids in ``[0, vocab_size)``.
        params (dict): Model parameters as described above.

    Returns:
        torch.Tensor: Logit tensor of shape ``(batch, seq_len, vocab_size)``.

    Examples:
        >>> import torch
        >>> V, D, S, B = 16, 8, 4, 1   # tiny model
        >>> D_FF, N_HEADS, N_BLOCKS = 32, 2, 2
        >>> I = torch.eye(D)
        >>> block = {
        ...     'W_q': I, 'W_k': I, 'W_v': I, 'W_o': I,
        ...     'gamma_1': torch.ones(D), 'beta_1': torch.zeros(D),
        ...     'W1': torch.zeros(D, D_FF), 'b1': torch.zeros(D_FF),
        ...     'W2': torch.zeros(D_FF, D), 'b2': torch.zeros(D),
        ...     'gamma_2': torch.ones(D), 'beta_2': torch.zeros(D),
        ...     'n_heads': N_HEADS,
        ... }
        >>> params = {
        ...     'wte': torch.randn(V, D),
        ...     'wpe': torch.randn(32, D),
        ...     'ln_f_gamma': torch.ones(D),
        ...     'ln_f_beta': torch.zeros(D),
        ...     'blocks': [block] * N_BLOCKS,
        ... }
        >>> token_ids = torch.randint(0, V, (B, S))
        >>> logits = gpt_model_forward(token_ids, params)
        >>> logits.shape
        torch.Size([1, 4, 16])

        >>> # Logits are real-valued floats, not probabilities
        >>> import math
        >>> logits.dtype == torch.float32
        True

        >>> # Batch dimension is preserved
        >>> logits2 = gpt_model_forward(torch.randint(0, V, (3, S)), params)
        >>> logits2.shape
        torch.Size([3, 4, 16])
    """
    raise NotImplementedError
