"""
Tests for Problem 043 — Define the GenerateRequest Schema
"""

import importlib
import pytest

try:
    mod = importlib.import_module("solutions.043_define_generate_request_schema")
    GenerateRequest = mod.GenerateRequest
except (ModuleNotFoundError, AttributeError):
    GenerateRequest = None

pytestmark = pytest.mark.skipif(
    GenerateRequest is None,
    reason="Solution 043 not implemented yet",
)


def test_default_fields():
    req = GenerateRequest(prompt="hello")
    assert req.prompt == "hello"
    assert req.max_tokens == 100
    assert req.temperature == 1.0
    assert req.top_k == 0
    assert req.top_p == 1.0


def test_custom_fields():
    req = GenerateRequest(
        prompt="Summarise this",
        max_tokens=256,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )
    assert req.max_tokens == 256
    assert req.temperature == 0.7
    assert req.top_k == 40
    assert req.top_p == 0.9


def test_temperature_must_be_positive():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", temperature=0.0)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", temperature=-1.0)


def test_max_tokens_bounds():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", max_tokens=0)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", max_tokens=2049)
    # boundary values are valid
    req_min = GenerateRequest(prompt="hi", max_tokens=1)
    assert req_min.max_tokens == 1
    req_max = GenerateRequest(prompt="hi", max_tokens=2048)
    assert req_max.max_tokens == 2048


def test_top_k_non_negative():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", top_k=-1)
    req = GenerateRequest(prompt="hi", top_k=0)
    assert req.top_k == 0


def test_top_p_bounds():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", top_p=0.0)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", top_p=1.1)
    # 1.0 is valid upper bound
    req = GenerateRequest(prompt="hi", top_p=1.0)
    assert req.top_p == 1.0
