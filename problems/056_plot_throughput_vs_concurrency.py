"""
Problem 056 — Plot Throughput vs Concurrency
=============================================
Visualise how token throughput scales with the number of concurrent users
and highlight the saturation point — the concurrency level beyond which
adding more users yields diminishing returns.

Difficulty : Medium
Tags        : benchmarking, matplotlib, visualisation, throughput, concurrency
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_throughput_vs_concurrency(
    results: list[dict],
    save_path: str = None,
) -> None:
    """Plot throughput (tokens/s) vs concurrent users and mark saturation.

    Draws a single line chart of token throughput against concurrency level.
    The *saturation point* — defined as the concurrency at which the second
    derivative of throughput is maximised (i.e. where throughput begins to
    plateau) — is highlighted with a vertical dashed line and an annotation.

    Args:
        results (list[dict]): Each dict must contain:
            - 'concurrency'              (int): Number of concurrent users.
            - 'throughput_tokens_per_sec' (float): Tokens generated per second.
            The list should be sorted by concurrency in ascending order.
        save_path (str | None): File path to save the figure.  Pass ``None``
            to display interactively.  Defaults to ``None``.

    Returns:
        None

    Examples:
        >>> results = [
        ...     {'concurrency': 1,  'throughput_tokens_per_sec': 50},
        ...     {'concurrency': 2,  'throughput_tokens_per_sec': 95},
        ...     {'concurrency': 4,  'throughput_tokens_per_sec': 160},
        ...     {'concurrency': 8,  'throughput_tokens_per_sec': 190},
        ...     {'concurrency': 16, 'throughput_tokens_per_sec': 195},
        ... ]
        >>> plot_throughput_vs_concurrency(results, save_path='/tmp/throughput.png')

        >>> # Single data point should not raise
        >>> plot_throughput_vs_concurrency(
        ...     [{'concurrency': 1, 'throughput_tokens_per_sec': 50}],
        ...     save_path='/tmp/single.png',
        ... )

        >>> # Saturation annotation appears when there are enough points
        >>> plot_throughput_vs_concurrency(results, save_path='/tmp/t2.png')
    """
    raise NotImplementedError
