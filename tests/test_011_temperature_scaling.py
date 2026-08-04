import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.011_temperature_scaling")
    temperature_scaling = mod.temperature_scaling
except Exception:
    mod = None
    temperature_scaling = None


def _skip_if_missing():
    if temperature_scaling is None:
        pytest.skip("Solution not implemented yet")


def test_temperature_below_one_amplifies():
    _skip_if_missing()
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = temperature_scaling(logits, 0.5)
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(result, expected), f"Got {result}"


def test_temperature_above_one_compresses():
    _skip_if_missing()
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = temperature_scaling(logits, 2.0)
    expected = torch.tensor([0.5, 1.0, 1.5])
    assert torch.allclose(result, expected), f"Got {result}"


def test_temperature_one_is_identity():
    _skip_if_missing()
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = temperature_scaling(logits, 1.0)
    assert torch.allclose(result, logits), f"Got {result}"


def test_output_shape_preserved():
    _skip_if_missing()
    logits = torch.randn(20)
    result = temperature_scaling(logits, 0.7)
    assert result.shape == logits.shape


def test_invalid_temperature_raises():
    _skip_if_missing()
    logits = torch.tensor([1.0, 2.0])
    with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):
        temperature_scaling(logits, 0.0)
