"""
Problem 045 — Build the Inference Engine Singleton
===================================================
Implement a module-level singleton that holds the inference engine state.
The engine must be explicitly initialised once before any generation can
proceed; subsequent calls to ``initialize_engine`` are no-ops.  Attempting
to retrieve the engine before initialisation raises ``RuntimeError``.

Difficulty : Medium
Tags        : singleton, state management, serving, lifecycle
"""

from __future__ import annotations

_engine: dict | None = None


def initialize_engine(
    model_name: str = "distilgpt2",
    max_batch_size: int = 8,
    page_pool_size: int = 256,
) -> None:
    """Load the model and allocate inference resources exactly once.

    On the first call, builds an ``engine`` dict with keys:
    ``"model_name"``, ``"max_batch_size"``, ``"page_pool_size"``,
    ``"initialized"`` (set to ``True``), and any other state needed to
    simulate generation (e.g. a token counter, a page table, etc.).

    Subsequent calls to ``initialize_engine`` with the *same or different*
    arguments are silently ignored — the singleton is immutable once set.

    Args:
        model_name (str): HuggingFace model identifier or local path.
            Defaults to ``"distilgpt2"``.
        max_batch_size (int): Maximum number of concurrent sequences the
            engine will handle.  Must be >= 1.  Defaults to 8.
        page_pool_size (int): Number of KV-cache pages to pre-allocate.
            Defaults to 256.

    Returns:
        None

    Examples:
        >>> initialize_engine("distilgpt2", max_batch_size=4)
        >>> engine = get_inference_engine()
        >>> engine["model_name"]
        'distilgpt2'
        >>> engine["initialized"]
        True

        >>> # Second call is a no-op
        >>> initialize_engine("gpt2")
        >>> get_inference_engine()["model_name"]
        'distilgpt2'

        >>> # Engine not initialised yet raises RuntimeError
        >>> get_inference_engine()
        # raises RuntimeError (if called before initialize_engine)
    """
    raise NotImplementedError


def get_inference_engine() -> dict:
    """Return the singleton inference engine dict.

    Must be called *after* :func:`initialize_engine`.  Raises
    ``RuntimeError`` if the engine has not been initialised.

    Returns:
        dict: The engine state dictionary created by :func:`initialize_engine`.

    Raises:
        RuntimeError: If ``initialize_engine`` has not been called yet.

    Examples:
        >>> initialize_engine()
        >>> engine = get_inference_engine()
        >>> isinstance(engine, dict)
        True
        >>> engine["initialized"]
        True
    """
    raise NotImplementedError
