import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.014_sample_next_token")
    sample_next_token = mod.sample_next_token
except Exception:
    mod = None
    sample_next_token = None


def _skip_if_missing():
    if sample_next_token is None:
        pytest.skip("Solution not implemented yet")


def test_return_type_is_int():
    _skip_if_missing()
    logits = torch.tensor([0.1, 0.5, 2.0, 0.3])
    result = sample_next_token(logits)
    assert isinstance(result, int), f"Expected int, got {type(result)}"


def test_output_in_valid_range():
    _skip_if_missing()
    vocab_size = 10
    logits = torch.randn(vocab_size)
    for _ in range(20):
        token = sample_next_token(logits)
        assert 0 <= token < vocab_size


def test_top_k_1_is_deterministic_argmax():
    _skip_if_missing()
    logits = torch.tensor([0.1, 0.2, 10.0, 0.3])
    # top_k=1 forces selection of the single highest logit
    results = {sample_next_token(logits, top_k=1) for _ in range(10)}
    assert results == {2}, f"Expected only token 2, got {results}"


def test_low_temperature_near_greedy():
    _skip_if_missing()
    logits = torch.tensor([0.1, 0.2, 10.0, 0.3])
    results = [sample_next_token(logits, temperature=0.001) for _ in range(20)]
    # With near-zero temperature the argmax (index 2) should dominate
    assert results.count(2) >= 18, f"Expected mostly token 2, got {results}"


def test_top_p_restricts_sampling():
    _skip_if_missing()
    # Single dominant token: with p=0.95 only index 3 should ever be sampled
    logits = torch.tensor([0.0, 0.0, 0.0, 100.0])
    results = {sample_next_token(logits, top_p=0.95) for _ in range(20)}
    assert results == {3}, f"Expected only token 3, got {results}"
