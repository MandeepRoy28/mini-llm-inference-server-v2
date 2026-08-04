"""
Problem 047 — Build the SSE Streaming Endpoint
===============================================
Implement a factory function that creates and returns a FastAPI application
exposing a ``POST /generate/stream`` endpoint.  The endpoint accepts a
``GenerateRequest`` body and returns a ``StreamingResponse`` that streams
Server-Sent Events (SSE) — one token per event — until generation is done.

Difficulty : Hard
Tags        : fastapi, SSE, streaming, async, serving
"""

from __future__ import annotations

import fastapi
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Minimal request schema used by the streaming endpoint."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


def create_streaming_app() -> FastAPI:
    """Create and return a FastAPI app with SSE streaming generation.

    The returned app exposes a single route:

    ``POST /generate/stream``
        Accepts a JSON body that deserialises into ``GenerateRequest``.
        Returns a ``StreamingResponse`` with
        ``content_type="text/event-stream"``.

        Each Server-Sent Event is formatted as::

            data: <token>\\n\\n

        where ``<token>`` is one decoded token string.  When generation is
        complete the final event is::

            data: [DONE]\\n\\n

    The function must create the FastAPI instance internally — it must NOT
    depend on any global app or engine state.  For testing purposes, stub
    out generation with a fixed sequence of dummy tokens (e.g. the words of
    the prompt split by whitespace, followed by "[DONE]").

    Returns:
        FastAPI: A configured FastAPI application instance.

    Examples:
        >>> from fastapi.testclient import TestClient
        >>> app = create_streaming_app()
        >>> client = TestClient(app)
        >>> resp = client.post(
        ...     "/generate/stream",
        ...     json={"prompt": "hello world", "max_tokens": 3},
        ... )
        >>> resp.status_code
        200
        >>> resp.headers["content-type"]
        'text/event-stream; charset=utf-8'

        >>> # Each line starting with "data: " carries one token
        >>> lines = [l for l in resp.text.splitlines() if l.startswith("data: ")]
        >>> lines[-1]
        'data: [DONE]'

        >>> # Invalid request returns 422
        >>> resp2 = client.post("/generate/stream", json={"prompt": "hi", "max_tokens": -1})
        >>> resp2.status_code
        422
    """
    raise NotImplementedError
