import importlib
import math
import pytest
import torch

try:
    mod = importlib.import_module("solutions.012_top_k_filter")
    top_k_filter = mod.top_k_filter
except Exception:
    mod = None
    top_k_filter = None


def _skip_if_missing():
    if top_k_filter is None:
        pytest.skip("Solution not implemented yet")


def test_top2_keeps_correct_positions():
    _skip_if_missing()
    logits = torch.tensor([1.0, 3.0, 0.5, 2.0, 4.0])
    result = top_k_filter(logits, k=2)
    # Indices 1 (3.0) and 4 (4.0) should survive
    assert result[4].item() == pytest.approx(4.0)
    assert result[1].item() == pytest.approx(3.0)
    assert math.isinf(result[0].item()) and result[0].item() < 0
    assert math.isinf(result[2].item()) and result[2].item() < 0
    assert math.isinf(result[3].item()) and result[3].item() < 0


def test_top_k_equals_vocab_keeps_all():
    _skip_if_missing()
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = top_k_filter(logits, k=3)
    assert torch.allclose(result, logits)


def test_top_k_one_keeps_only_max():
    _skip_if_missing()
    logits = torch.tensor([0.5, 2.5, 1.0, 3.0])
    result = top_k_filter(logits, k=1)
    assert result[3].item() == pytest.approx(3.0)
    for i in [0, 1, 2]:
        assert math.isinf(result[i].item()) and result[i].item() < 0


def test_output_shape_preserved():
    _skip_if_missing()
    logits = torch.randn(100)
    result = top_k_filter(logits, k=10)
    assert result.shape == logits.shape


def test_exactly_k_finite_values():
    _skip_if_missing()
    logits = torch.randn(50)
    k = 7
    result = top_k_filter(logits, k=k)
    finite_count = torch.isfinite(result).sum().item()
    assert finite_count == k
