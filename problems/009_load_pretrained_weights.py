"""
Problem 009 — Load Pre-Trained GPT-2 Weights from HuggingFace
==============================================================
Real-world inference rarely trains from scratch.  This problem asks you to
pull the pre-trained DistilGPT-2 weights from HuggingFace ``transformers``
and re-map them into the parameter dictionary format used by
:func:`gpt_model_forward`.

DistilGPT-2 has 6 transformer blocks, d_model=768, n_heads=12, d_ff=3072
and a vocabulary of 50257 tokens.

Difficulty : Medium (mostly HuggingFace API familiarity)
Tags        : HuggingFace, transformers, weight loading, GPT-2
"""
from __future__ import annotations

import torch


def load_pretrained_weights(model_name: str = "distilgpt2") -> dict:
    """Load DistilGPT-2 weights and return them in the project's params format.

    Uses ``transformers.GPT2Model.from_pretrained(model_name)`` to download
    the model, then extracts and re-maps the state-dict entries into the
    nested ``params`` dictionary consumed by :func:`gpt_model_forward`.

    Key HuggingFace → project name mappings:

    * ``transformer.wte.weight``           → ``params['wte']``
    * ``transformer.wpe.weight``           → ``params['wpe']``
    * ``transformer.ln_f.weight/bias``     → ``params['ln_f_gamma'/'ln_f_beta']``
    * For block *i* (``transformer.h.{i}``):

      * ``attn.c_attn.weight/bias`` — fused QKV projection (split into 3 equal
        parts along axis 1); weights are transposed vs PyTorch ``nn.Linear``
        convention because GPT-2 uses Conv1D.
      * ``attn.c_proj.weight/bias`` → W_o
      * ``mlp.c_fc.weight/bias``    → W1, b1
      * ``mlp.c_proj.weight/bias``  → W2, b2
      * ``ln_1.weight/bias``        → gamma_1, beta_1
      * ``ln_2.weight/bias``        → gamma_2, beta_2

    Args:
        model_name (str): HuggingFace model identifier.  Defaults to
            ``"distilgpt2"`` (6-layer GPT-2 small variant, ~82 M params).

    Returns:
        dict: Parameter dictionary in the format expected by
            :func:`gpt_model_forward`::

                {
                    'wte':        Tensor (50257, 768),
                    'wpe':        Tensor (1024, 768),
                    'ln_f_gamma': Tensor (768,),
                    'ln_f_beta':  Tensor (768,),
                    'blocks': [
                        {
                            'W_q', 'W_k', 'W_v',      # (768, 768)
                            'W_o',                     # (768, 768)
                            'gamma_1', 'beta_1',       # (768,)
                            'W1', 'b1',                # (768, 3072), (3072,)
                            'W2', 'b2',                # (3072, 768), (768,)
                            'gamma_2', 'beta_2',       # (768,)
                            'n_heads',                 # int = 12
                        },
                        ... (6 blocks total)
                    ]
                }

    Examples:
        >>> params = load_pretrained_weights("distilgpt2")
        >>> params['wte'].shape
        torch.Size([50257, 768])

        >>> params['wpe'].shape
        torch.Size([1024, 768])

        >>> len(params['blocks'])
        6

        >>> params['blocks'][0]['W_q'].shape
        torch.Size([768, 768])
    """
    raise NotImplementedError
