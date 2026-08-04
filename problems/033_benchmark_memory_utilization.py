"""
Problem 033 — Benchmark Memory Utilization: Naive vs Paged
Part 4: Paged Attention
"""

import math


def benchmark_memory_utilization(
    sequence_lengths: list[int],
    page_size: int = 16,
    max_seq_len: int = 512,
) -> dict:
    """
    Compare memory utilisation between naive pre-allocation and paged allocation.

    Naive allocation:
        Every sequence reserves `max_seq_len` contiguous slots. Waste = slots
        allocated but never written. Waste% = wasted / total_allocated * 100.

    Paged allocation:
        Only complete pages are allocated. A sequence of length L needs
        ceil(L / page_size) pages, each holding page_size slots. The last page
        may have unused slots within it.
        Slots allocated (paged) = ceil(L / page_size) * page_size  (per seq)
        Waste (paged) = slots_allocated_paged - sum(sequence_lengths)
        Waste% (paged) = wasted_paged / total_paged_allocated * 100

    Memory reduction factor:
        How many times less memory paged uses vs naive:
        memory_reduction_factor = total_naive_allocated / total_paged_allocated

    Args:
        sequence_lengths (list[int]): Actual token lengths for each sequence.
        page_size (int, optional): Number of token slots per page. Defaults to 16.
        max_seq_len (int, optional): Maximum sequence length for naive allocation.
            Defaults to 512.

    Returns:
        dict: A dictionary with the following keys:
            - 'naive_total_allocated' (int): Total slots in naive scheme.
            - 'paged_total_allocated' (int): Total slots in paged scheme.
            - 'naive_waste_pct' (float): Waste percentage for naive, rounded to 2dp.
            - 'paged_waste_pct' (float): Waste percentage for paged, rounded to 2dp.
            - 'memory_reduction_factor' (float): naive / paged, rounded to 2dp.

    Examples:
        >>> benchmark_memory_utilization([10, 5, 8], page_size=16, max_seq_len=64)
        {
            'naive_total_allocated': 192,
            'paged_total_allocated': 48,
            'naive_waste_pct': 88.02,
            'paged_waste_pct': 52.08,
            'memory_reduction_factor': 4.0
        }

        >>> # Perfect utilisation: sequences exactly fill pages/max_len
        >>> benchmark_memory_utilization([16, 16], page_size=16, max_seq_len=16)
        {'naive_waste_pct': 0.0, 'paged_waste_pct': 0.0, 'memory_reduction_factor': 1.0, ...}

        >>> # Single very short sequence against a large max_seq_len
        >>> result = benchmark_memory_utilization([3], page_size=16, max_seq_len=512)
        >>> result['memory_reduction_factor'] > 30
        True
    """
    raise NotImplementedError
