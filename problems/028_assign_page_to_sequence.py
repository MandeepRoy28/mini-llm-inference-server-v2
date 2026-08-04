"""
Problem 028 — Assign Page to Sequence
Part 4: Paged Attention
"""


def assign_page_to_sequence(pool: dict, block_table: dict, seq_id: int) -> int:
    """
    Allocate one free page from the pool and assign it to a sequence.

    When a sequence needs more KV-cache space (e.g. it has filled its last page
    or is just starting), this function pops a free page from the pool, records
    the mapping in the block table, and returns the page ID so the caller knows
    where to write the next tokens.

    The function modifies `pool` and `block_table` **in-place**:
    - Removes the allocated page ID from pool['free_pages'].
    - Appends the page ID to block_table[seq_id].

    Args:
        pool (dict): The page pool dict as returned by `allocate_page_pool`.
            Must contain key 'free_pages' (list[int]).
        block_table (dict): Mapping from sequence_id (int) to list of page IDs.
            The sequence seq_id must already exist as a key.
        seq_id (int): The ID of the sequence requesting a new page.

    Returns:
        int: The page ID that was assigned to the sequence.

    Raises:
        ValueError: If pool['free_pages'] is empty (out-of-memory condition).

    Examples:
        >>> pool = {'free_pages': [0, 1, 2]}
        >>> bt = {0: [], 1: []}
        >>> assign_page_to_sequence(pool, bt, seq_id=0)
        0
        >>> bt
        {0: [0], 1: []}
        >>> pool['free_pages']
        [1, 2]

        >>> # Assigning a second page to the same sequence
        >>> assign_page_to_sequence(pool, bt, seq_id=0)
        1
        >>> bt[0]
        [0, 1]

        >>> # Raises ValueError when pool is exhausted
        >>> assign_page_to_sequence({'free_pages': []}, {3: []}, seq_id=3)
        ValueError: No free pages available in the pool.
    """
    raise NotImplementedError
