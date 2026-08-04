import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.010_greedy_sample")
    greedy_sample = mod.greedy_sample
except Exception:
    mod = None
    greedy_sample = None


def _skip_if_missing():
    if greedy_sample is None:
        pytest.skip("Solution not implemented yet")


def test_basic_argmax():
    _skip_if_missing()
    logits = torch.tensor([0.1, 0.5, 0.9, 0.2])
    assert greedy_sample(logits) == 2


def test_argmax_at_first_position():
    _skip_if_missing()
    logits = torch.tensor([5.0, 1.0, 2.0, 3.0])
    assert greedy_sample(logits) == 0


def test_argmax_at_last_position():
    _skip_if_missing()
    logits = torch.tensor([-1.0, -2.0, -3.0, 0.5])
    assert greedy_sample(logits) == 3


def test_negative_logits():
    _skip_if_missing()
    logits = torch.tensor([-5.0, -1.0, -3.0])
    assert greedy_sample(logits) == 1


def test_return_type_is_int():
    _skip_if_missing()
    logits = torch.tensor([0.0, 1.0, 0.5])
    result = greedy_sample(logits)
    assert isinstance(result, int), f"Expected int, got {type(result)}"
