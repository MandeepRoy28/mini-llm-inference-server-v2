"""
Problem 034 — Understand Static Batching
Part 5: Continuous Batching
"""


def simulate_static_batching(sequence_lengths: list[int], max_seq_len: int) -> dict:
    """
    Simulate static batching where all sequences must finish before the batch
    is released.

    In static batching every sequence in a batch runs for exactly
    max(sequence_lengths) decode steps — the length of the longest sequence.
    Shorter sequences finish early but their GPU slots remain occupied (and
    idle) until the slowest sequence completes.  This wasted compute is the
    core inefficiency that Continuous Batching eliminates.

    Args:
        sequence_lengths (list[int]): Number of tokens each sequence needs to
            generate.  Must be non-empty; each value must be > 0 and <=
            max_seq_len.
        max_seq_len (int): Upper bound on sequence length (used only for
            validation purposes; the actual batch runs for
            max(sequence_lengths) steps).

    Returns:
        dict: A dictionary with the following keys:
            - 'batch_size' (int): Number of sequences in the batch
              (= len(sequence_lengths)).
            - 'max_seq_in_batch' (int): Length of the longest sequence
              (= max(sequence_lengths)).
            - 'total_steps' (int): Number of decode steps the batch runs
              (= max_seq_in_batch).
            - 'wasted_compute_pct' (float): Percentage of GPU slot-steps that
              were idle because a shorter sequence had already finished.
              Computed as:
                  wasted_slots = sum(max_seq_in_batch - l for l in sequence_lengths)
                  total_slot_steps = batch_size * total_steps
                  wasted_compute_pct = round(wasted_slots / total_slot_steps * 100, 2)

    Examples:
        >>> simulate_static_batching([5, 10, 3], max_seq_len=20)
        {'batch_size': 3, 'max_seq_in_batch': 10, 'total_steps': 10, 'wasted_compute_pct': 46.67}

        >>> simulate_static_batching([8, 8, 8], max_seq_len=16)
        {'batch_size': 3, 'max_seq_in_batch': 8, 'total_steps': 8, 'wasted_compute_pct': 0.0}

        >>> simulate_static_batching([1, 20], max_seq_len=20)
        {'batch_size': 2, 'max_seq_in_batch': 20, 'total_steps': 20, 'wasted_compute_pct': 47.5}
    """
    raise NotImplementedError
