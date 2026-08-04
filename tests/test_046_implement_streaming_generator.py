"""
Tests for Problem 046 — Implement the Streaming Token Generator
"""

import asyncio
import importlib
import pytest

try:
    mod = importlib.import_module("solutions.046_implement_streaming_generator")
    streaming_token_generator = mod.streaming_token_generator
except (ModuleNotFoundError, AttributeError):
    streaming_token_generator = None

pytestmark = pytest.mark.skipif(
    streaming_token_generator is None,
    reason="Solution 046 not implemented yet",
)


def _fake_engine():
    return {"model_name": "distilgpt2", "initialized": True}


def _collect(engine, request):
    async def _run():
        tokens = []
        async for tok in streaming_token_generator(engine, request):
            tokens.append(tok)
        return tokens

    return asyncio.run(_run())


def test_last_token_is_done():
    tokens = _collect(_fake_engine(), {"prompt": "hello", "max_tokens": 5})
    assert tokens[-1] == "[DONE]"


def test_correct_number_of_tokens():
    max_tokens = 7
    tokens = _collect(_fake_engine(), {"prompt": "hello", "max_tokens": max_tokens})
    # Expect max_tokens content tokens + 1 [DONE] sentinel
    assert len(tokens) == max_tokens + 1


def test_all_intermediate_tokens_are_strings():
    tokens = _collect(_fake_engine(), {"prompt": "test", "max_tokens": 4})
    for tok in tokens[:-1]:
        assert isinstance(tok, str)
        assert tok != "[DONE]"


def test_done_appears_exactly_once():
    tokens = _collect(_fake_engine(), {"prompt": "hello world", "max_tokens": 6})
    assert tokens.count("[DONE]") == 1


def test_generator_is_async():
    import inspect
    engine = _fake_engine()
    request = {"prompt": "hi", "max_tokens": 3}
    gen = streaming_token_generator(engine, request)
    assert inspect.isasyncgen(gen)
