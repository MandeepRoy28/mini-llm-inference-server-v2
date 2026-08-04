"""
Problem 023 — Decode Phase (single step)
Part of the DeepML-style LLM Inference Server project.
"""

import torch


def decode_phase(
    model_forward_with_cache,
    last_token_id: int,
    cache: list[dict],
    step: int,
    temperature: float = 1.0,
) -> tuple[int, list[dict]]:
    """Run a single autoregressive decode step using the KV cache.

    After the prefill phase has processed the full prompt, the decode phase
    generates one token at a time.  Each call runs the model forward on a
    single-token input, updates the KV cache with the new K/V activations, and
    samples the next token from the resulting logits.

    Temperature scaling is applied before sampling:
        ``probs = softmax(logits / temperature)``

    Args:
        model_forward_with_cache: Callable with signature
            ``model_forward_with_cache(input_ids, cache) -> (logits, cache)``
            where ``input_ids`` has shape ``(batch_size, 1)`` and ``logits``
            has shape ``(batch_size, 1, vocab_size)``.
        last_token_id: Integer token id of the most recently generated (or
            last prompt) token.  Will be wrapped in a ``(1, 1)`` LongTensor
            before being passed to the model.
        cache: Current KV cache state (list of per-layer dicts).
        step: Current decode step index (0-based after the prefill).  Passed
            through so that callers can track sequence position externally.
        temperature: Softmax temperature for sampling.  Values < 1 make the
            distribution sharper; values > 1 make it flatter.  Must be > 0.
            Default is 1.0 (no rescaling).

    Returns:
        A tuple ``(next_token_id, updated_cache)`` where

        * ``next_token_id`` is a Python int sampled from the output
          distribution.
        * ``updated_cache`` is the cache after inserting the new K/V pair.

    Examples:
        >>> import torch
        >>> vocab = 10
        >>> def mock_forward(ids, cache):
        ...     logits = torch.zeros(1, 1, vocab)
        ...     logits[0, 0, 3] = 100.0  # always picks token 3
        ...     return logits, cache
        >>> token, cache = decode_phase(mock_forward, last_token_id=0,
        ...                             cache=[], step=0, temperature=1.0)
        >>> token
        3
        >>> isinstance(token, int)
        True
        >>> isinstance(cache, list)
        True
    """
    raise NotImplementedError
