"""
Problem 048 — Build the Non-Streaming Generation Endpoint
==========================================================
Implement a factory function that creates and returns a FastAPI application
with a ``POST /generate`` route.  Unlike the streaming endpoint, this route
waits for the entire generation to complete and then returns a single JSON
response containing the full generated text plus timing metrics.

Difficulty : Medium
Tags        : fastapi, REST, inference, serving, pydantic
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request schema for non-streaming generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    """Response schema for non-streaming generation."""
    generated_text: str
    tokens_generated: int
    time_to_first_token: float
    total_time: float
    request_id: str


def create_app() -> FastAPI:
    """Create and return a FastAPI app with a non-streaming generation route.

    The returned app exposes a single route:

    ``POST /generate``
        Accepts a JSON body deserialised as ``GenerateRequest``.
        Returns a JSON body serialised from ``GenerateResponse``.

        The handler must:

        1. Record a start timestamp.
        2. Simulate or run token generation (stub: split prompt words,
           repeat/truncate to ``max_tokens`` tokens, join with spaces).
        3. Record ``time_to_first_token`` as the elapsed time after
           producing the first token.
        4. Record ``total_time`` as the elapsed time after all tokens.
        5. Assign a unique ``request_id`` (UUID4 string).
        6. Return ``GenerateResponse`` with all fields populated.

    Returns:
        FastAPI: A configured FastAPI application instance.

    Examples:
        >>> from fastapi.testclient import TestClient
        >>> app = create_app()
        >>> client = TestClient(app)
        >>> resp = client.post("/generate", json={"prompt": "hello world", "max_tokens": 5})
        >>> resp.status_code
        200
        >>> data = resp.json()
        >>> "generated_text" in data
        True
        >>> data["tokens_generated"] == 5
        True

        >>> # request_id is a non-empty string
        >>> bool(data["request_id"])
        True

        >>> # Timing fields are non-negative floats
        >>> data["time_to_first_token"] >= 0
        True
        >>> data["total_time"] >= data["time_to_first_token"]
        True
    """
    raise NotImplementedError
