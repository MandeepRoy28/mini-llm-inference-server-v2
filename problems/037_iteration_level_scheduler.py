"""
Problem 037 — Iteration-Level Scheduler
Part 5: Continuous Batching
"""

import queue


def iteration_level_scheduler(
    request_queue: queue.Queue,
    running_batch: dict,
    max_batch_size: int,
) -> list[dict]:
    """
    Fill available slots in the running batch with requests from the queue.

    The iteration-level scheduler runs at the start of every decode step.
    Unlike static batching, it does not wait for the entire batch to drain —
    it opportunistically pulls new requests from the queue whenever a slot
    opens up.  This is the key mechanism that enables continuous batching.

    The function:
    1. Computes available capacity: max_batch_size - len(running_batch).
    2. Dequeues up to that many requests (non-blocking; stops if the queue
       is empty).
    3. Assigns each dequeued request the next available seq_id (starting from
       max(running_batch.keys()) + 1 if the batch is non-empty, else 0).
    4. Inserts each request into running_batch as a sequence-state dict with
       the same schema as add_to_batch() from problem 036:
           'request_id', 'token_ids', 'max_new_tokens', 'tokens_generated',
           'temperature', 'finished'.
    5. Returns a list of the newly added request dicts (in the order they
       were dequeued).

    Args:
        request_queue (queue.Queue): FIFO queue of pending request dicts.
        running_batch (dict): Current running batch mapping seq_id -> state.
            Modified in place.
        max_batch_size (int): Maximum number of sequences allowed concurrently.

    Returns:
        list[dict]: Newly added request dicts.  Empty list if no capacity or
            queue is empty.

    Examples:
        >>> import queue
        >>> rq = queue.Queue()
        >>> rq.put({'request_id': 0, 'prompt_ids': [1, 2], 'max_new_tokens': 4, 'temperature': 1.0})
        >>> batch = {}
        >>> added = iteration_level_scheduler(rq, batch, max_batch_size=2)
        >>> len(added)
        1

        >>> batch[0]['request_id']
        0

        >>> rq2 = queue.Queue()
        >>> iteration_level_scheduler(rq2, {}, max_batch_size=4)
        []
    """
    raise NotImplementedError
