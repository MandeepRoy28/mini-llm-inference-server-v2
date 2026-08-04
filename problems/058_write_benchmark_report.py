"""
Problem 058 — Write a Benchmark Report
=======================================
Format a benchmark result dictionary into a readable text report table and
optionally persist it to disk.  The report is the final artefact of a
benchmark run and should be easy to share and diff in version control.

Difficulty : Easy
Tags        : benchmarking, reporting, formatting, file I/O
"""


def write_benchmark_report(
    results: dict,
    output_path: str = None,
) -> str:
    """Generate a formatted plain-text benchmark report and optionally save it.

    Reads configuration, throughput, latency, and memory fields from
    ``results`` and formats them into a fixed-width table.  The report is
    always printed to stdout and returned as a string.  When ``output_path``
    is provided the same string is written to that file.

    Expected keys in ``results`` (missing keys should render as ``'N/A'``):
        - 'model'               (str): Model name, e.g. ``'distilgpt2'``.
        - 'batch_size'          (int): Batch size used.
        - 'page_size'           (int): Page size in tokens.
        - 'throughput_req_per_sec'    (float): Requests per second.
        - 'throughput_tokens_per_sec' (float): Tokens per second.
        - 'ttft_p50'            (float): Median TTFT in seconds.
        - 'ttft_p95'            (float): 95th-pct TTFT in seconds.
        - 'ttft_p99'            (float): 99th-pct TTFT in seconds.
        - 'tpot_p50'            (float): Median TPOT in seconds.
        - 'tpot_p95'            (float): 95th-pct TPOT in seconds.
        - 'tpot_p99'            (float): 99th-pct TPOT in seconds.
        - 'gpu_memory_used_gb'  (float): GPU memory in GB.

    Args:
        results (dict): Benchmark results dictionary (see keys above).
        output_path (str | None): If given, write the report to this file
            path.  Defaults to ``None`` (print only).

    Returns:
        str: The formatted report string (including header and footer lines).

    Examples:
        >>> results = {
        ...     'model': 'distilgpt2', 'batch_size': 8, 'page_size': 16,
        ...     'throughput_req_per_sec': 14.2,
        ...     'throughput_tokens_per_sec': 142.3,
        ...     'ttft_p50': 0.043, 'ttft_p95': 0.089, 'ttft_p99': 0.121,
        ...     'tpot_p50': 0.012, 'tpot_p95': 0.019, 'tpot_p99': 0.024,
        ...     'gpu_memory_used_gb': 3.7,
        ... }
        >>> report = write_benchmark_report(results)
        >>> '======' in report
        True

        >>> 'distilgpt2' in report
        True

        >>> '142.3' in report
        True
    """
    raise NotImplementedError
