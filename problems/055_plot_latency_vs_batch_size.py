"""
Problem 055 — Plot Latency vs Batch Size
=========================================
Visualise how TTFT and TPOT change as the inference batch size grows.
Two subplots on a single figure make it easy to spot the trade-off between
batching efficiency and latency.

Difficulty : Easy
Tags        : benchmarking, matplotlib, visualisation, latency, batch_size
"""

import matplotlib.pyplot as plt


def plot_latency_vs_batch_size(
    results: list[dict],
    save_path: str = None,
) -> None:
    """Plot TTFT and TPOT as a function of batch size.

    Creates a figure with two side-by-side line charts:
      - Left  : Mean TTFT (s) vs batch size.
      - Right : Mean TPOT (s) vs batch size.

    Both axes share the same x-tick labels (the batch sizes present in
    ``results``).  Markers are drawn at each data point.  If ``save_path``
    is provided the figure is saved there; otherwise ``plt.show()`` is called.

    Args:
        results (list[dict]): Each dict must contain:
            - 'batch_size'  (int): Batch size used for this run.
            - 'ttft_mean'   (float): Mean TTFT in seconds.
            - 'tpot_mean'   (float): Mean TPOT in seconds.
            Typically batch sizes are [1, 2, 4, 8, 16].
        save_path (str | None): File path to save the figure (e.g.
            ``'latency_vs_batch.png'``).  Pass ``None`` to display
            interactively.  Defaults to ``None``.

    Returns:
        None

    Examples:
        >>> results = [
        ...     {'batch_size': 1,  'ttft_mean': 0.03, 'tpot_mean': 0.01},
        ...     {'batch_size': 2,  'ttft_mean': 0.04, 'tpot_mean': 0.011},
        ...     {'batch_size': 4,  'ttft_mean': 0.05, 'tpot_mean': 0.013},
        ...     {'batch_size': 8,  'ttft_mean': 0.08, 'tpot_mean': 0.017},
        ...     {'batch_size': 16, 'ttft_mean': 0.14, 'tpot_mean': 0.025},
        ... ]
        >>> plot_latency_vs_batch_size(results, save_path='/tmp/latency.png')

        >>> # Calling with save_path=None triggers plt.show()
        >>> # plot_latency_vs_batch_size(results)

        >>> # Empty results list should not raise
        >>> plot_latency_vs_batch_size([], save_path='/tmp/empty.png')
    """
    raise NotImplementedError
