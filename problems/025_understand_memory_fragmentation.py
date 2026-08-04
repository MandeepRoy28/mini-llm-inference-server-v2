"""
Problem 025 — Understand Memory Fragmentation in KV Cache
Part 4: Paged Attention
"""


def simulate_naive_allocation(sequence_lengths: list[int], max_seq_len: int) -> dict:
    """
    Simulate naive KV cache allocation where each sequence pre-allocates
    max_seq_len slots regardless of its actual token length.

    In a naive implementation, every sequence reserves a contiguous block of
    max_seq_len slots in the KV cache when it starts. Most sequences never
    reach max_seq_len, so a large fraction of allocated memory sits unused —
    this is the core memory-fragmentation problem that Paged Attention solves.

    Args:
        sequence_lengths (list[int]): Actual token lengths for each sequence
            currently in the batch.
        max_seq_len (int): Maximum sequence length; every sequence is allocated
            exactly this many KV-cache slots.

    Returns:
        dict: A dictionary with the following keys:
            - 'total_slots_allocated' (int): Total slots reserved across all
              sequences (= len(sequence_lengths) * max_seq_len).
            - 'slots_actually_used' (int): Sum of sequence_lengths — the slots
              that hold real token data.
            - 'wasted_slots' (int): total_slots_allocated - slots_actually_used.
            - 'waste_percentage' (float): wasted_slots / total_slots_allocated
              * 100, rounded to 2 decimal places.

    Examples:
        >>> simulate_naive_allocation([10, 5, 8], max_seq_len=20)
        {'total_slots_allocated': 60, 'slots_actually_used': 23,
         'wasted_slots': 37, 'waste_percentage': 61.67}

        >>> simulate_naive_allocation([100, 100], max_seq_len=100)
        {'total_slots_allocated': 200, 'slots_actually_used': 200,
         'wasted_slots': 0, 'waste_percentage': 0.0}

        >>> simulate_naive_allocation([1], max_seq_len=512)
        {'total_slots_allocated': 512, 'slots_actually_used': 1,
         'wasted_slots': 511, 'waste_percentage': 99.8}
    """
    raise NotImplementedError
