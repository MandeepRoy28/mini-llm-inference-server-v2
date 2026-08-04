"""
Problem 031 — Free Pages on Sequence Completion
Part 4: Paged Attention
"""


def free_pages_on_completion(pool: dict, block_table: dict, seq_id: int) -> int:
    """
    Return all pages belonging to a completed sequence back to the free pool.

    When a sequence finishes generation (e.g. EOS token produced or max length
    reached), its pages must be reclaimed so they can be reused by new
    sequences. This function:
    1. Reads block_table[seq_id] to find all owned page IDs.
    2. Appends every page ID to pool['free_pages'].
    3. Resets block_table[seq_id] to an empty list [].
    4. Returns the count of pages that were freed.

    Both `pool` and `block_table` are modified **in-place**.

    Args:
        pool (dict): Page pool containing 'free_pages' (list[int]).
        block_table (dict): Mapping sequence_id -> list of physical page IDs.
            The key seq_id must exist.
        seq_id (int): ID of the sequence that has finished.

    Returns:
        int: Number of pages freed (length of the original page list for seq_id).

    Examples:
        >>> pool = {'free_pages': [2, 3]}
        >>> bt = {0: [0, 1], 1: [4]}
        >>> free_pages_on_completion(pool, bt, seq_id=0)
        2
        >>> pool['free_pages']
        [2, 3, 0, 1]
        >>> bt[0]
        []

        >>> # Freeing a sequence with a single page
        >>> free_pages_on_completion(pool, bt, seq_id=1)
        1

        >>> # Freeing a sequence that already has no pages
        >>> bt[5] = []
        >>> free_pages_on_completion(pool, bt, seq_id=5)
        0
    """
    raise NotImplementedError
