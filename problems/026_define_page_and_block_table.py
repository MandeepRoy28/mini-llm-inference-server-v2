"""
Problem 026 — Define Page and Block Table
Part 4: Paged Attention
"""

PAGE_SIZE = 16  # Each page holds this many token slots


def get_page_size() -> int:
    """
    Return the fixed page size used throughout the Paged Attention system.

    A "page" is the fundamental unit of KV-cache memory in Paged Attention.
    Each page stores exactly PAGE_SIZE token positions worth of keys and values.
    Keeping this constant centralises the single source of truth so all other
    functions agree on the granularity.

    Returns:
        int: The page size constant (16).

    Examples:
        >>> get_page_size()
        16

        >>> type(get_page_size())
        <class 'int'>

        >>> get_page_size() > 0
        True
    """
    raise NotImplementedError


def create_block_table(n_sequences: int) -> dict:
    """
    Create an empty block table mapping each sequence ID to an empty page list.

    The block table is the central data structure of Paged Attention. It maps
    every active sequence ID (0-indexed integer) to an ordered list of physical
    page IDs that together hold that sequence's KV cache. Starting empty means
    pages are allocated lazily as tokens are generated.

    Args:
        n_sequences (int): Number of sequences to initialise entries for.
            Sequence IDs are 0-indexed: 0, 1, ..., n_sequences-1.

    Returns:
        dict: A dictionary mapping each sequence_id (int) to an empty list [].
            Example for n_sequences=3: {0: [], 1: [], 2: []}

    Examples:
        >>> create_block_table(3)
        {0: [], 1: [], 2: []}

        >>> create_block_table(1)
        {0: []}

        >>> len(create_block_table(5))
        5
    """
    raise NotImplementedError
