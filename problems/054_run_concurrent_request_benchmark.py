"""
Problem 054 — Run a Concurrent-Request Benchmark
=================================================
Stress-test an async inference engine by firing multiple requests at once
using ``asyncio.gather``.  Wall-clock time is measured across the whole
batch so that throughput figures reflect real concurrency.

Difficulty : Medium
Tags        : benchmarking, async, concurrency, throughput, asyncio
"""

import asyncio
import time


def run_concurrent_benchmark(
    engine_fn,
    requests: list[dict],
    concurrency: int,
) -> dict:
    """Fire *concurrency* requests simultaneously and measure aggregate throughput.

    Splits ``requests`` into chunks of size ``concurrency`` and runs each
    chunk with ``asyncio.gather``.  The total wall-clock time covers all
    chunks.  Per-request results (request_id, tokens_generated, total_time)
    are collected into ``individual_results``.

    ``engine_fn`` must be an *async* callable with the signature
    ``async engine_fn(request: dict) -> list[int]``.

    Args:
        engine_fn (async callable): Async function that accepts a request dict
            and returns a list of generated token IDs.
        requests (list[dict]): List of request dicts.  Each dict must contain
            at least 'request_id' and 'max_new_tokens'.
        concurrency (int): Number of requests to run simultaneously in each
            batch.

    Returns:
        dict: Aggregate benchmark results with the following keys:
            - 'total_requests'           (int): len(requests).
            - 'concurrency'              (int): The concurrency argument.
            - 'total_time'               (float): Wall-clock seconds for all
              requests.
            - 'throughput_req_per_sec'   (float): total_requests / total_time.
            - 'throughput_tokens_per_sec' (float): total_tokens / total_time.
            - 'individual_results'       (list[dict]): Per-request dicts with
              'request_id', 'tokens_generated', 'total_time'.

    Examples:
        >>> import asyncio
        >>> async def dummy_async_engine(req):
        ...     await asyncio.sleep(0.01)
        ...     return list(range(req['max_new_tokens']))
        >>> reqs = [{'request_id': i, 'max_new_tokens': 10} for i in range(4)]
        >>> result = run_concurrent_benchmark(dummy_async_engine, reqs, concurrency=2)
        >>> result['total_requests']
        4

        >>> result['concurrency']
        2

        >>> len(result['individual_results'])
        4
    """
    raise NotImplementedError
