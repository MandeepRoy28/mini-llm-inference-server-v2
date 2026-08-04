import importlib
import math
import pytest
import torch

try:
    mod = importlib.import_module("solutions.013_top_p_nucleus_filter")
    top_p_nucleus_filter = mod.top_p_nucleus_filter
except Exception:
    mod = None
    top_p_nucleus_filter = None


def _skip_if_missing():
    if top_p_nucleus_filter is None:
        pytest.skip("Solution not implemented yet")


def test_p_1_keeps_all_tokens():
    _skip_if_missing()
    logits = torch.tensor([1.0, 2.0, 3.0, 0.5])
    result = top_p_nucleus_filter(logits, p=1.0)
    # No positions should be -inf when p=1.0
    assert not torch.any(result == float("-inf")), "p=1.0 should keep all tokens"


def test_dominant_token_covers_high_p():
    _skip_if_missing()
    # Single very high logit dominates probability mass
    logits = torch.tensor([100.0, 0.0, 0.0, 0.0])
    result = top_p_nucleus_filter(logits, p=0.9)
    # Token 0 has ~100% probability, so only it should survive
    assert torch.isfinite(result[0]), "Top token must survive"
    for i in [1, 2, 3]:
        assert math.isinf(result[i].item()) and result[i].item() < 0


def test_at_least_one_token_always_kept():
    _skip_if_missing()
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    result = top_p_nucleus_filter(logits, p=0.01)  # very small p
    finite_count = torch.isfinite(result).sum().item()
    assert finite_count >= 1, "At least one token must always be retained"


def test_output_shape_preserved():
    _skip_if_missing()
    logits = torch.randn(30)
    result = top_p_nucleus_filter(logits, p=0.8)
    assert result.shape == logits.shape


def test_original_order_preserved():
    _skip_if_missing()
    # Values that survive should equal their originals (not sorted values)
    logits = torch.tensor([0.1, 5.0, 0.2, 4.0])
    result = top_p_nucleus_filter(logits, p=0.9)
    # Token 1 (5.0) should be highest; if it survives, its value must be 5.0
    if torch.isfinite(result[1]):
        assert result[1].item() == pytest.approx(5.0)
