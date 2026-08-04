"""
Problem 017 — Understand KV Recompute Cost
Part of the DeepML-style LLM Inference Server project.
"""

import time
import torch


def benchmark_no_cache(model_forward, input_ids: torch.Tensor, n_steps: int) -> dict:
    """Run n_steps of autoregressive generation WITHOUT a KV cache.

    At each step the full sequence (prompt + all generated tokens so far) is
    passed through model_forward so that every K and V matrix is recomputed
    from scratch. This establishes the baseline cost of recomputation and
    makes it easy to compare against the KV-cache path.

    Args:
        model_forward: Callable with signature ``model_forward(input_ids) ->
            logits`` where logits has shape
            ``(batch_size, seq_len, vocab_size)``.  The function should be
            deterministic in the sense that it returns the same result given
            the same input (no stateful side-effects expected by the caller).
        input_ids: LongTensor of shape ``(batch_size, prompt_len)`` containing
            the tokenised prompt to start generation from.
        n_steps: Number of new tokens to generate (i.e. the number of
            autoregressive decode steps to time).

    Returns:
        A dict with three keys:

        * ``'total_time'`` (float): Wall-clock seconds for all n_steps.
        * ``'time_per_token'`` (float): ``total_time / n_steps``.
        * ``'n_tokens'`` (int): Equal to n_steps (tokens generated).

    Examples:
        >>> import torch
        >>> # Toy "model" that always predicts token id 7
        >>> def dummy_forward(ids):
        ...     B, T = ids.shape
        ...     logits = torch.zeros(B, T, 10)
        ...     logits[:, :, 7] = 100.0
        ...     return logits
        >>> result = benchmark_no_cache(dummy_forward, torch.zeros(1, 4, dtype=torch.long), 5)
        >>> result['n_tokens']
        5
        >>> result['total_time'] >= 0
        True
        >>> abs(result['time_per_token'] - result['total_time'] / 5) < 1e-9
        True
    """
    raise NotImplementedError
