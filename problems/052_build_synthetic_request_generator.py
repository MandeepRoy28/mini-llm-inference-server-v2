"""
Problem 052 — Build a Synthetic Request Generator
==================================================
Create a reproducible stream of fake inference requests so that benchmark
harnesses can be run without a real dataset.  Each request carries a prompt
represented as a list of token IDs and a max-new-tokens budget.

Difficulty : Easy
Tags        : benchmarking, data generation, numpy, random, reproducibility
"""

import numpy as np


def generate_synthetic_requests(
    n: int,
    min_prompt_len: int = 10,
    max_prompt_len: int = 100,
    min_max_tokens: int = 20,
    max_max_tokens: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Generate *n* synthetic LLM inference requests with random prompts.

    Uses a seeded NumPy RNG so that the same call always returns the same
    sequence of requests, making benchmark results reproducible.  Token IDs
    are sampled from a vocabulary of 32 000 tokens (ids 0–31 999).

    Args:
        n (int): Number of requests to generate.
        min_prompt_len (int): Minimum number of tokens in the prompt.
            Defaults to 10.
        max_prompt_len (int): Maximum number of tokens in the prompt.
            Defaults to 100.
        min_max_tokens (int): Minimum value for the max_new_tokens field.
            Defaults to 20.
        max_max_tokens (int): Maximum value for the max_new_tokens field.
            Defaults to 200.
        seed (int): Seed for the NumPy random number generator.
            Defaults to 42.

    Returns:
        list[dict]: A list of *n* request dictionaries.  Each dict has:
            - 'request_id'    (int): Zero-based index of the request.
            - 'prompt_ids'    (list[int]): Random token IDs for the prompt.
            - 'max_new_tokens' (int): Token budget for generation.

    Examples:
        >>> reqs = generate_synthetic_requests(5, seed=42)
        >>> len(reqs)
        5
        >>> set(reqs[0].keys()) == {'request_id', 'prompt_ids', 'max_new_tokens'}
        True

        >>> reqs[0]['request_id']
        0
        >>> 10 <= len(reqs[0]['prompt_ids']) <= 100
        True

        >>> reqs_a = generate_synthetic_requests(3, seed=7)
        >>> reqs_b = generate_synthetic_requests(3, seed=7)
        >>> reqs_a[0]['prompt_ids'] == reqs_b[0]['prompt_ids']
        True
    """
    raise NotImplementedError
