"""
Tests for Problem 045 — Build the Inference Engine Singleton
"""

import importlib
import pytest


try:
    mod = importlib.import_module("solutions.045_build_inference_engine_singleton")
    initialize_engine = mod.initialize_engine
    get_inference_engine = mod.get_inference_engine
except (ModuleNotFoundError, AttributeError):
    initialize_engine = None
    get_inference_engine = None

pytestmark = pytest.mark.skipif(
    initialize_engine is None or get_inference_engine is None,
    reason="Solution 045 not implemented yet",
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    mod._engine = None
    yield
    mod._engine = None


def test_get_before_init_raises():
    with pytest.raises(RuntimeError):
        get_inference_engine()


def test_initialize_and_get():
    initialize_engine("distilgpt2", max_batch_size=4, page_pool_size=128)
    engine = get_inference_engine()
    assert isinstance(engine, dict)
    assert engine.get("initialized") is True
    assert engine.get("model_name") == "distilgpt2"


def test_initialize_is_idempotent():
    initialize_engine("distilgpt2")
    initialize_engine("gpt2")  # second call — must be no-op
    engine = get_inference_engine()
    assert engine["model_name"] == "distilgpt2"


def test_engine_stores_batch_and_pool_size():
    initialize_engine("distilgpt2", max_batch_size=8, page_pool_size=256)
    engine = get_inference_engine()
    assert engine.get("max_batch_size") == 8
    assert engine.get("page_pool_size") == 256


def test_get_returns_same_object():
    initialize_engine()
    e1 = get_inference_engine()
    e2 = get_inference_engine()
    assert e1 is e2
