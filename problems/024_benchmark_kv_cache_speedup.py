"""
Problem 024 — Benchmark KV Cache Speedup
Part of the DeepML-style LLM Inference Server project.
"""

import torch


def benchmark_kv_cache_speedup(
    model_forward,
    model_forward_with_cache,
    input_ids: torch.Tensor,
    n_new_tokens: int,
) -> dict:
    """Time autoregressive generation both with and without a KV cache.

    Runs two complete generation trials starting from the same ``input_ids``:

    1. **No-cache baseline** — re-runs the full sequence through
       ``model_forward`` at every step (recomputing all K/V).
    2. **Cached run** — uses ``model_forward_with_cache`` in prefill + decode
       mode so K/V are computed only once per position.

    Both trials generate exactly ``n_new_tokens`` new tokens.  Wall-clock time
    is measured with ``time.perf_counter`` (or equivalent).

    Args:
        model_forward: Callable ``model_forward(input_ids) -> logits`` used
            for the no-cache baseline.  ``logits`` shape:
            ``(batch_size, seq_len, vocab_size)``.
        model_forward_with_cache: Callable
            ``model_forward_with_cache(input_ids, cache) -> (logits, cache)``
            used for the cached run.
        input_ids: LongTensor of shape ``(batch_size, prompt_len)`` — the
            tokenised prompt shared by both trials.
        n_new_tokens: Number of new tokens to generate in each trial.

    Returns:
        A dict with the following keys (all float except noted):

        * ``'no_cache_time'``: Total wall-clock seconds for the no-cache run.
        * ``'with_cache_time'``: Total wall-clock seconds for the cached run.
        * ``'speedup_factor'``: ``no_cache_time / with_cache_time``.
        * ``'tokens_per_sec_no_cache'``: ``n_new_tokens / no_cache_time``.
        * ``'tokens_per_sec_with_cache'``: ``n_new_tokens / with_cache_time``.

    Examples:
        >>> import torch
        >>> vocab = 16
        >>> def fwd(ids):
        ...     B, T = ids.shape
        ...     logits = torch.zeros(B, T, vocab)
        ...     logits[:, :, 1] = 10.0
        ...     return logits
        >>> def fwd_cache(ids, cache):
        ...     return fwd(ids), cache
        >>> result = benchmark_kv_cache_speedup(fwd, fwd_cache,
        ...                                     torch.zeros(1, 4, dtype=torch.long), 3)
        >>> set(result.keys()) == {'no_cache_time', 'with_cache_time',
        ...                        'speedup_factor', 'tokens_per_sec_no_cache',
        ...                        'tokens_per_sec_with_cache'}
        True
        >>> result['speedup_factor'] > 0
        True
        >>> result['tokens_per_sec_no_cache'] > 0
        True
    """
    raise NotImplementedError
