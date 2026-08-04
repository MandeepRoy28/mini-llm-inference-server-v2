"""
Problem 053 — Run a Single-Request Benchmark
=============================================
Wrap an inference engine callable so that a single request can be timed
end-to-end.  Warm-up passes let the engine (and any JIT / caching layers)
reach steady state before the measured run.

Difficulty : Medium
Tags        : benchmarking, latency, TTFT, TPOT, timing, warmup
"""

import time


def run_single_request_benchmark(
    engine_fn,
    request: dict,
    n_warmup: int = 2,
) -> dict:
    """Time a single request end-to-end against an inference engine callable.

    The engine callable ``engine_fn`` accepts a request dict and returns a
    list of generated token IDs.  Before the measured pass, ``n_warmup``
    un-timed calls are made to eliminate cold-start effects.

    Time-to-first-token (TTFT) is measured by wrapping ``engine_fn`` in a
    generator that yields individual tokens; if the engine returns a full list
    at once, TTFT equals the total time.  Average time-per-output-token (TPOT)
    is computed as ``total_time / tokens_generated``.

    Args:
        engine_fn (callable): A callable ``engine_fn(request: dict) ->
            list[int]`` that runs one forward pass and returns generated token
            IDs.
        request (dict): Request dictionary with at least 'request_id' and
            'max_new_tokens' keys (as produced by
            ``generate_synthetic_requests``).
        n_warmup (int): Number of un-timed warm-up iterations.  Defaults to 2.

    Returns:
        dict: Timing results with the following keys:
            - 'request_id'      (int | str): Copied from ``request``.
            - 'ttft'            (float): Time to first token in seconds.
            - 'tpot'            (float): Mean time per output token in seconds.
            - 'total_time'      (float): Total inference time in seconds.
            - 'tokens_generated' (int): Number of tokens returned by engine.

    Examples:
        >>> import time
        >>> def dummy_engine(req):
        ...     time.sleep(0.01)
        ...     return list(range(req['max_new_tokens']))
        >>> result = run_single_request_benchmark(
        ...     dummy_engine, {'request_id': 0, 'max_new_tokens': 5}, n_warmup=1
        ... )
        >>> result['request_id']
        0
        >>> result['tokens_generated']
        5

        >>> result['total_time'] > 0
        True

        >>> result['tpot'] == result['total_time'] / result['tokens_generated']
        True
    """
    raise NotImplementedError
