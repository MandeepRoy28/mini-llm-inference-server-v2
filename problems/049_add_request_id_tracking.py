"""
Problem 049 — Add Request-ID Tracking and Aggregate Stats
==========================================================
Extend the generation app with per-request UUID tracking and aggregate
latency statistics.  Every request receives a unique ID; its latency metrics
are stored so that callers can query summary statistics at any time.

Difficulty : Hard
Tags        : fastapi, monitoring, statistics, serving, UUID
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request schema for tracked generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    """Response schema for tracked generation."""
    generated_text: str
    tokens_generated: int
    time_to_first_token: float
    total_time: float
    request_id: str


def create_tracked_app() -> FastAPI:
    """Create and return a FastAPI app with request-ID tracking.

    The returned app exposes two routes:

    ``POST /generate``
        Same contract as Problem 048's ``/generate`` route, with the
        addition that each request's latency metrics are recorded in an
        in-memory log so they can be queried later.

    ``GET /stats``
        Returns a JSON object with aggregate statistics across all
        requests processed since the server started (or since the last
        reset):

        * ``total_requests`` (int) — number of requests handled.
        * ``avg_ttft`` (float) — mean time-to-first-token in seconds.
        * ``avg_total_time`` (float) — mean total generation time.
        * ``p50_latency`` (float) — median total_time.
        * ``p95_latency`` (float) — 95th-percentile total_time.

        Returns zeros / empty values if no requests have been processed.

    The in-memory log must be scoped to the app instance returned by this
    function so that different test calls do not share state.

    Returns:
        FastAPI: A configured FastAPI application instance.

    Examples:
        >>> from fastapi.testclient import TestClient
        >>> app = create_tracked_app()
        >>> client = TestClient(app)
        >>> for _ in range(3):
        ...     client.post("/generate", json={"prompt": "hi", "max_tokens": 5})
        >>> stats = client.get("/stats").json()
        >>> stats["total_requests"]
        3
        >>> stats["avg_ttft"] >= 0
        True

        >>> # p95 >= p50 always
        >>> stats["p95_latency"] >= stats["p50_latency"]
        True

        >>> # Fresh app has zero stats
        >>> fresh = create_tracked_app()
        >>> TestClient(fresh).get("/stats").json()["total_requests"]
        0
    """
    raise NotImplementedError


def get_request_stats() -> dict:
    """Return aggregate stats from the module-level request log (if any).

    This function provides a module-level view of statistics, independent
    of any specific app instance.  It reads from a module-level list of
    recorded latency entries and computes the same fields as the ``/stats``
    route:

    Returns:
        dict: A dictionary with keys ``total_requests`` (int),
        ``avg_ttft`` (float), ``avg_total_time`` (float),
        ``p50_latency`` (float), ``p95_latency`` (float).

    Examples:
        >>> stats = get_request_stats()
        >>> isinstance(stats, dict)
        True
        >>> set(stats.keys()) >= {"total_requests", "avg_ttft", "p50_latency"}
        True

        >>> # When no requests recorded, all numeric values are 0.0
        >>> stats["total_requests"]
        0
    """
    raise NotImplementedError
