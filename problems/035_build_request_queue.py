"""
Problem 035 — Build a Request Queue
Part 5: Continuous Batching
"""

import queue


def build_request_queue(requests: list[dict]) -> queue.Queue:
    """
    Given a list of request dicts, build and return a queue.Queue containing
    them in the same order they were provided.

    In a continuous-batching inference server, incoming requests are placed
    into a FIFO queue.  The scheduler dequeues them one at a time as GPU
    capacity becomes available, enabling fine-grained iteration-level
    scheduling instead of waiting for a whole batch to drain.

    Each request dict is expected to have the following keys:
        - 'request_id'    (int)  : Unique identifier for the request.
        - 'prompt_ids'    (list[int]): Tokenised prompt as a list of token IDs.
        - 'max_new_tokens' (int) : Maximum number of tokens to generate.
        - 'temperature'   (float): Sampling temperature (1.0 = no scaling).

    Args:
        requests (list[dict]): Ordered list of request dictionaries.  May be
            empty, in which case an empty queue is returned.

    Returns:
        queue.Queue: A FIFO queue whose elements are the request dicts in the
            same order as the input list.

    Examples:
        >>> reqs = [{'request_id': 0, 'prompt_ids': [1, 2, 3], 'max_new_tokens': 10, 'temperature': 1.0}]
        >>> q = build_request_queue(reqs)
        >>> q.qsize()
        1

        >>> q2 = build_request_queue([])
        >>> q2.empty()
        True

        >>> reqs3 = [
        ...     {'request_id': 0, 'prompt_ids': [1], 'max_new_tokens': 5, 'temperature': 0.8},
        ...     {'request_id': 1, 'prompt_ids': [2, 3], 'max_new_tokens': 8, 'temperature': 1.2},
        ... ]
        >>> q3 = build_request_queue(reqs3)
        >>> q3.get()['request_id']
        0
    """
    raise NotImplementedError
