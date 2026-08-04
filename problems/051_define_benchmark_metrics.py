"""
Problem 051 — Define Benchmark Metrics
=======================================
Compute all key benchmark metrics for an LLM inference server from raw
per-request timing measurements.  The returned dictionary provides a
complete picture of latency (TTFT, TPOT) and throughput in a single call.

Difficulty : Easy
Tags        : benchmarking, latency, throughput, numpy, statistics
"""

import numpy as np


def compute_metrics(
    ttft_list: list[float],
    tpot_list: list[float],
    total_tokens: int,
    total_time: float,
) -> dict:
    """Compute all key benchmark metrics from raw timing measurements.

    Given per-request time-to-first-token (TTFT) and time-per-output-token
    (TPOT) samples together with aggregate totals, return a dictionary that
    covers mean, p50, p95, and p99 latency statistics as well as overall
    token throughput.

    Args:
        ttft_list (list[float]): Per-request time-to-first-token in seconds.
        tpot_list (list[float]): Per-request average time-per-output-token in
            seconds.
        total_tokens (int): Total number of output tokens generated across all
            requests in the benchmark run.
        total_time (float): Total wall-clock time for the benchmark run in
            seconds.

    Returns:
        dict: A dictionary with the following keys:
            - 'ttft_mean'  (float): Mean TTFT across all requests.
            - 'ttft_p50'   (float): 50th-percentile TTFT.
            - 'ttft_p95'   (float): 95th-percentile TTFT.
            - 'ttft_p99'   (float): 99th-percentile TTFT.
            - 'tpot_mean'  (float): Mean TPOT across all requests.
            - 'tpot_p50'   (float): 50th-percentile TPOT.
            - 'tpot_p95'   (float): 95th-percentile TPOT.
            - 'throughput_tokens_per_sec' (float): total_tokens / total_time.
            - 'total_tokens' (int): Echo of the total_tokens argument.
            - 'total_time'   (float): Echo of the total_time argument.

    Examples:
        >>> metrics = compute_metrics([0.1, 0.2, 0.15], [0.05, 0.06, 0.055], 100, 10.0)
        >>> metrics['ttft_mean']
        0.15
        >>> metrics['ttft_p50']
        0.15
        >>> round(metrics['ttft_p95'], 4)
        0.195

        >>> metrics2 = compute_metrics([0.05], [0.01], 50, 5.0)
        >>> metrics2['throughput_tokens_per_sec']
        10.0

        >>> metrics3 = compute_metrics(
        ...     [0.08, 0.12, 0.09, 0.20, 0.11],
        ...     [0.02, 0.03, 0.025, 0.04, 0.022],
        ...     500, 25.0,
        ... )
        >>> metrics3['total_tokens']
        500
        >>> metrics3['total_time']
        25.0
    """
    raise NotImplementedError
