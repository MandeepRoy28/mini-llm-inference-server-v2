"""
Problem 046 — Implement the Streaming Token Generator
======================================================
Implement an async generator that simulates token-by-token streaming from
the inference engine.  Each iteration yields one decoded token string.  When
generation is complete (max_tokens reached or EOS), the generator yields a
final ``"[DONE]"`` sentinel and then returns.

Difficulty : Medium
Tags        : async, generator, streaming, inference, serving
"""

from __future__ import annotations

from collections.abc import AsyncGenerator


async def streaming_token_generator(
    engine: dict,
    request: dict,
) -> AsyncGenerator[str, None]:
    """Yield one token string at a time as generation proceeds.

    Simulates autoregressive token generation using the supplied ``engine``
    dict and ``request`` dict.  Generation continues until either
    ``request["max_tokens"]`` tokens have been produced or a special
    end-of-sequence marker is generated.

    The very last value yielded is the sentinel string ``"[DONE]"``, which
    signals to the consumer that the stream has finished.  Consumers should
    strip this sentinel before displaying or concatenating tokens.

    Args:
        engine (dict): The inference engine dict returned by
            ``get_inference_engine()``.  Must contain at least
            ``"model_name"`` and ``"initialized": True``.
        request (dict): Generation request parameters.  Must contain:

            * ``"prompt"`` (str) — the conditioning text.
            * ``"max_tokens"`` (int) — maximum tokens to generate.
            * ``"temperature"`` (float, optional) — sampling temperature.
            * ``"top_k"`` (int, optional) — top-k filter cutoff.
            * ``"top_p"`` (float, optional) — nucleus sampling threshold.

    Yields:
        str: One token string per iteration, followed by ``"[DONE]"`` as
        the final sentinel.

    Examples:
        >>> import asyncio
        >>> async def collect():
        ...     tokens = []
        ...     async for tok in streaming_token_generator(engine, req):
        ...         tokens.append(tok)
        ...     return tokens
        >>> tokens = asyncio.run(collect())
        >>> tokens[-1]
        '[DONE]'

        >>> # Stream and print in real time
        >>> async for token in streaming_token_generator(engine, req):
        ...     print(token, end="", flush=True)

        >>> # Total tokens yielded == max_tokens + 1 (the [DONE] sentinel)
        >>> len(tokens) == req["max_tokens"] + 1
        True
    """
    raise NotImplementedError
