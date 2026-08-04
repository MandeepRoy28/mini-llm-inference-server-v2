"""
Problem 040 — Handle Sequence Completion
Part 5: Continuous Batching
"""


def handle_sequence_completion(
    running_batch: dict,
    pool: dict,
    block_table: dict,
    eos_token_id: int = None,
) -> list[dict]:
    """
    Identify finished sequences, free their resources, and return them.

    After each decode step, some sequences may have reached their stopping
    condition.  This function checks every sequence in running_batch and:
    - Marks a sequence as finished if its last generated token equals
      eos_token_id (when eos_token_id is not None) OR if
      tokens_generated >= max_new_tokens.
    - Removes finished sequences from running_batch.
    - Returns all pages used by each finished sequence from pool['used']
      back to pool['free'], and clears their entry in block_table.
    - Returns a list of completed-sequence dicts.

    Each completed-sequence dict in the returned list should contain:
        - 'request_id'       (int)      : Original request ID.
        - 'token_ids'        (list[int]): Full token list (prompt + generated).
        - 'tokens_generated' (int)      : Number of tokens actually generated.

    Args:
        running_batch (dict): Maps seq_id -> sequence state.  Finished
            sequences are removed in place.
        pool          (dict): KV-cache page pool {'free': list[int],
            'used': list[int]}.  Freed pages are moved back to 'free'.
        block_table   (dict): Maps seq_id -> list of page IDs.  Entries for
            finished sequences are deleted.
        eos_token_id  (int, optional): Token ID that signals end-of-sequence.
            If None, only the max_new_tokens condition applies.

    Returns:
        list[dict]: Completed sequence dicts (may be empty if no sequence
            finished this step).

    Examples:
        >>> batch = {0: {'request_id': 0, 'token_ids': [1, 2, 3, 99], 'max_new_tokens': 10, 'tokens_generated': 3, 'temperature': 1.0, 'finished': False}}
        >>> pool = {'free': [], 'used': [5, 6]}
        >>> block_table = {0: [5, 6]}
        >>> completed = handle_sequence_completion(batch, pool, block_table, eos_token_id=99)
        >>> len(completed)
        1

        >>> completed[0]['request_id']
        0

        >>> 0 not in batch
        True
    """
    raise NotImplementedError
