"""
Problem 043 — Define the GenerateRequest Schema
================================================
Define a Pydantic BaseModel that validates incoming LLM generation requests.
The model enforces sensible bounds on all generation hyperparameters so that
invalid requests are rejected before they ever reach the inference engine.

Difficulty : Easy
Tags        : pydantic, validation, API, streaming, serving
"""

from pydantic import BaseModel, field_validator


class GenerateRequest(BaseModel):
    """Pydantic model for incoming LLM generation requests.

    Validates and normalises all generation hyperparameters.  Invalid values
    (e.g. negative max_tokens or a temperature of zero) are rejected with a
    ``ValueError`` at construction time via Pydantic field validators.

    Fields:
        prompt (str): The input text to condition generation on.
        max_tokens (int): Maximum number of new tokens to generate.
            Must be >= 1 and <= 2048.  Defaults to 100.
        temperature (float): Softmax temperature applied to logits.
            Must be strictly greater than 0.  Defaults to 1.0.
        top_k (int): If > 0, only the top-k logits are kept before sampling.
            Must be >= 0 (0 disables top-k filtering).  Defaults to 0.
        top_p (float): Nucleus sampling probability mass threshold.
            Must be in (0, 1].  Defaults to 1.0 (disabled).

    Examples:
        >>> req = GenerateRequest(prompt="Hello world", max_tokens=50)
        >>> req.temperature
        1.0
        >>> req.top_k
        0
        >>> req.top_p
        1.0

        >>> req2 = GenerateRequest(
        ...     prompt="Summarise this article",
        ...     max_tokens=256,
        ...     temperature=0.7,
        ...     top_k=40,
        ...     top_p=0.9,
        ... )
        >>> req2.max_tokens
        256

        >>> # Invalid temperature raises ValidationError
        >>> GenerateRequest(prompt="hi", temperature=0.0)
        # raises pydantic.ValidationError
    """
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_positive(cls, v: float) -> float:
        raise NotImplementedError

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_in_range(cls, v: int) -> int:
        raise NotImplementedError

    @field_validator("top_k")
    @classmethod
    def top_k_non_negative(cls, v: int) -> int:
        raise NotImplementedError

    @field_validator("top_p")
    @classmethod
    def top_p_in_range(cls, v: float) -> float:
        raise NotImplementedError
