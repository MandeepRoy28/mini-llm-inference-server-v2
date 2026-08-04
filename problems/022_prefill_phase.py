"""
Problem 022 — Prefill Phase
Part of the DeepML-style LLM Inference Server project.
"""

import torch


def prefill_phase(
    model_forward_with_cache,
    input_ids: torch.Tensor,
    cache: list[dict],
) -> tuple[torch.Tensor, list[dict]]:
    """Process the full prompt in a single forward pass and populate the KV cache.

    The prefill phase is the first stage of two-phase LLM inference.  The
    entire prompt is processed at once (using parallelism across all prompt
    positions), and the resulting K/V activations are written into the cache
    so that the subsequent decode phase never has to re-attend over prompt
    tokens.

    Args:
        model_forward_with_cache: Callable with signature
            ``model_forward_with_cache(input_ids, cache) -> (logits, cache)``
            where

            * ``input_ids`` is a LongTensor of shape ``(batch_size, seq_len)``,
            * ``cache`` is the list of per-layer dicts (may be empty or
              pre-allocated), and the function returns updated logits of shape
              ``(batch_size, seq_len, vocab_size)`` and the populated cache.

        input_ids: LongTensor of shape ``(batch_size, prompt_len)``
            representing the tokenised prompt.
        cache: Pre-allocated KV cache structure (list of dicts with ``'k'``
            and ``'v'`` tensors).  Passed through to ``model_forward_with_cache``
            and returned in the updated state.

    Returns:
        A tuple ``(last_logits, updated_cache)`` where

        * ``last_logits`` is the logits tensor for the **last prompt token**,
          shape ``(batch_size, vocab_size)``.
        * ``updated_cache`` is the cache dict list after the prefill forward
          pass.

    Examples:
        >>> import torch
        >>> vocab = 20
        >>> def mock_forward(ids, cache):
        ...     B, T = ids.shape
        ...     logits = torch.zeros(B, T, vocab)
        ...     return logits, cache
        >>> cache = []
        >>> input_ids = torch.zeros(1, 6, dtype=torch.long)
        >>> last_logits, updated_cache = prefill_phase(mock_forward, input_ids, cache)
        >>> last_logits.shape
        torch.Size([1, 20])
        >>> last_logits.shape[0]  # batch dimension preserved
        1
        >>> isinstance(updated_cache, list)
        True
    """
    raise NotImplementedError
