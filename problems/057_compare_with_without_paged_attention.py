"""
Problem 057 — Compare Paged vs Naive Attention Benchmark Results
================================================================
Given two benchmark result dictionaries — one from a naive KV-cache
implementation and one from a paged-attention implementation — compute
improvement factors and produce a human-readable summary table.

Difficulty : Easy
Tags        : benchmarking, paged attention, comparison, analysis
"""


def compare_paged_vs_naive(
    naive_results: dict,
    paged_results: dict,
) -> dict:
    """Compute improvement factors between naive and paged-attention benchmarks.

    Calculates how much paged attention improves memory efficiency, throughput,
    and latency over a naive KV-cache baseline.  All improvement factors are
    expressed as ``naive_value / paged_value`` (> 1 means paged is better) for
    memory/latency, and ``paged_value / naive_value`` for throughput.

    Both input dicts are expected to contain at minimum:
        - 'memory_used_gb'           (float): GPU memory consumed.
        - 'throughput_tokens_per_sec' (float): Token throughput.
        - 'ttft_mean'                 (float): Mean time to first token.

    Args:
        naive_results (dict): Benchmark results from the naive attention run.
        paged_results (dict): Benchmark results from the paged attention run.

    Returns:
        dict: A comparison dictionary with the following keys:
            - 'memory_reduction_factor'    (float): naive_memory / paged_memory.
            - 'throughput_improvement_factor' (float): paged_tps / naive_tps.
            - 'latency_improvement_factor' (float): naive_ttft / paged_ttft.
            - 'summary_table'              (list[dict]): One row per metric.
              Each row dict has:
                  'metric'      (str): Human-readable metric name.
                  'naive'       (float): Naive value.
                  'paged'       (float): Paged value.
                  'improvement' (float): The improvement factor.

    Examples:
        >>> naive = {'memory_used_gb': 8.0, 'throughput_tokens_per_sec': 100.0, 'ttft_mean': 0.2}
        >>> paged = {'memory_used_gb': 4.0, 'throughput_tokens_per_sec': 180.0, 'ttft_mean': 0.1}
        >>> cmp = compare_paged_vs_naive(naive, paged)
        >>> cmp['memory_reduction_factor']
        2.0

        >>> cmp['throughput_improvement_factor']
        1.8

        >>> len(cmp['summary_table'])
        3
    """
    raise NotImplementedError
