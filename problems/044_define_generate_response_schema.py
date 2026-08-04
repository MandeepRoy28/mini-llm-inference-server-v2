"""
Problem 044 — Define the GenerateResponse Schema
=================================================
Define a Pydantic BaseModel that represents the response returned by the LLM
inference server after a full (non-streaming) generation completes.  The model
captures not just the generated text but also the latency metrics that clients
and monitoring systems need.

Difficulty : Easy
Tags        : pydantic, API, serving, metrics
"""

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    """Pydantic model for a completed LLM generation response.

    Carries the generated text together with timing and throughput metadata
    that allows callers to measure end-to-end latency and tokens-per-second
    throughput without requiring a separate profiling pass.

    Fields:
        generated_text (str): The text produced by the model.
        tokens_generated (int): Number of tokens that were decoded.
        time_to_first_token (float): Wall-clock seconds from request receipt
            to the moment the first token was produced.
        total_time (float): Total wall-clock seconds for the full generation.
        request_id (str): A unique identifier string for this request, used
            for logging and tracing.

    Methods:
        tokens_per_second() -> float:
            Returns ``tokens_generated / total_time``.  Useful for measuring
            decoding throughput.

    Examples:
        >>> resp = GenerateResponse(
        ...     generated_text="The sky is blue.",
        ...     tokens_generated=4,
        ...     time_to_first_token=0.05,
        ...     total_time=0.2,
        ...     request_id="req-001",
        ... )
        >>> resp.tokens_per_second()
        20.0

        >>> resp2 = GenerateResponse(
        ...     generated_text="Hello",
        ...     tokens_generated=1,
        ...     time_to_first_token=0.1,
        ...     total_time=0.1,
        ...     request_id="req-002",
        ... )
        >>> resp2.tokens_per_second()
        10.0

        >>> # total_time of 0 would raise ZeroDivisionError — callers must
        >>> # ensure at least one step was measured before constructing.
    """
    generated_text: str
    tokens_generated: int
    time_to_first_token: float
    total_time: float
    request_id: str

    def tokens_per_second(self) -> float:
        """Return decoding throughput as tokens per second.

        Returns:
            float: ``tokens_generated / total_time``.
        """
        raise NotImplementedError
