"""
Problem 050 — Test Streaming with curl
=======================================
Implement two utility functions for manual end-to-end testing of the SSE
streaming endpoint from a shell.  ``build_curl_command`` constructs the
exact curl invocation needed to hit the endpoint, and ``parse_sse_response``
turns the raw SSE response body into an ordered list of token strings.

Difficulty : Easy
Tags        : curl, SSE, testing, shell, parsing
"""

from __future__ import annotations

import json


def build_curl_command(
    host: str,
    port: int,
    prompt: str,
    max_tokens: int = 50,
) -> str:
    """Return a curl shell command string for testing the SSE streaming endpoint.

    The command targets ``POST http://<host>:<port>/generate/stream`` with a
    JSON body containing the supplied parameters.  The ``--no-buffer`` flag
    is included so that curl prints each SSE chunk as it arrives rather than
    buffering the whole response.

    Args:
        host (str): Hostname or IP address of the inference server
            (e.g. ``"localhost"`` or ``"127.0.0.1"``).
        port (int): TCP port the server is listening on (e.g. ``8000``).
        prompt (str): The generation prompt to embed in the request body.
        max_tokens (int): Maximum tokens to generate.  Defaults to 50.

    Returns:
        str: A single-line shell command string that can be run directly
        in a terminal.  The JSON body uses double quotes internally and the
        outer ``-d`` argument uses single quotes.

    Examples:
        >>> build_curl_command("localhost", 8000, "Hello world")
        "curl -X POST http://localhost:8000/generate/stream -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello world\", \"max_tokens\": 50}' --no-buffer"

        >>> build_curl_command("192.168.1.10", 9000, "Summarise this", max_tokens=200)
        "curl -X POST http://192.168.1.10:9000/generate/stream -H 'Content-Type: application/json' -d '{\"prompt\": \"Summarise this\", \"max_tokens\": 200}' --no-buffer"

        >>> # The prompt is embedded verbatim in the JSON body
        >>> cmd = build_curl_command("localhost", 8000, "Once upon a time")
        >>> '"prompt": "Once upon a time"' in cmd
        True
    """
    raise NotImplementedError


def parse_sse_response(raw_response: str) -> list[str]:
    """Parse a raw SSE response body into an ordered list of token strings.

    Scans ``raw_response`` line by line.  Any line that starts with
    ``"data: "`` is treated as an SSE data event; the prefix is stripped to
    get the token value.  The ``"[DONE]"`` sentinel is excluded from the
    returned list.  All other lines (empty lines, comment lines, etc.) are
    ignored.

    Args:
        raw_response (str): The raw text body of an SSE response, as
            captured by curl or a test client.  Lines are separated by
            ``"\\n"`` and events are typically separated by blank lines.

    Returns:
        list[str]: The token strings in the order they were received,
        with ``"[DONE]"`` excluded.

    Examples:
        >>> parse_sse_response("data: Hello\\n\\ndata: world\\n\\ndata: [DONE]\\n\\n")
        ['Hello', 'world']

        >>> parse_sse_response("data: The\\n\\ndata:  quick\\n\\ndata: [DONE]\\n\\n")
        ['The', ' quick']

        >>> # Empty response returns empty list
        >>> parse_sse_response("")
        []

        >>> # No [DONE] sentinel — still returns all tokens found
        >>> parse_sse_response("data: foo\\n\\ndata: bar\\n\\n")
        ['foo', 'bar']
    """
    raise NotImplementedError
