"""
Problem 036 — Build a Running Batch
Part 5: Continuous Batching
"""


def build_running_batch() -> dict:
    """
    Create and return an empty running batch.

    A running batch is a plain dict that maps each active sequence ID (an
    integer) to a sequence-state dict.  Starting empty lets the scheduler
    populate it incrementally via add_to_batch() as requests are dequeued.

    Returns:
        dict: An empty dictionary {}.  Sequence IDs will be inserted as keys
            by add_to_batch().

    Examples:
        >>> batch = build_running_batch()
        >>> batch
        {}

        >>> type(build_running_batch())
        <class 'dict'>

        >>> len(build_running_batch())
        0
    """
    raise NotImplementedError


def add_to_batch(batch: dict, request: dict, seq_id: int) -> None:
    """
    Insert a new sequence into the running batch, initialising its state.

    Each entry in the running batch tracks everything needed to drive one
    sequence through the decode loop.  The function mutates *batch* in place
    and returns nothing.

    The sequence-state dict stored at batch[seq_id] must have exactly these
    keys:
        - 'request_id'       (int)      : Copied from request['request_id'].
        - 'token_ids'        (list[int]): Starts as a copy of
                                          request['prompt_ids'].
        - 'max_new_tokens'   (int)      : Copied from request['max_new_tokens'].
        - 'tokens_generated' (int)      : Initialised to 0.
        - 'temperature'      (float)    : Copied from request['temperature'].
        - 'finished'         (bool)     : Initialised to False.

    Args:
        batch   (dict): The running batch dict to mutate.
        request (dict): A request dict with keys 'request_id', 'prompt_ids',
                        'max_new_tokens', 'temperature'.
        seq_id  (int):  The integer key under which this sequence is stored.

    Returns:
        None

    Examples:
        >>> batch = {}
        >>> req = {'request_id': 7, 'prompt_ids': [10, 20], 'max_new_tokens': 5, 'temperature': 1.0}
        >>> add_to_batch(batch, req, seq_id=0)
        >>> batch[0]['request_id']
        7

        >>> batch[0]['tokens_generated']
        0

        >>> batch[0]['finished']
        False
    """
    raise NotImplementedError
