import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.015_autoregressive_generate")
    autoregressive_generate = mod.autoregressive_generate
except Exception:
    mod = None
    autoregressive_generate = None


def _skip_if_missing():
    if autoregressive_generate is None:
        pytest.skip("Solution not implemented yet")


VOCAB_SIZE = 6


def _make_model(fixed_token: int):
    """Return a dummy model that always predicts fixed_token as most likely."""
    def model_forward(ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.shape[0]
        logits = torch.zeros(seq_len, VOCAB_SIZE)
        logits[:, fixed_token] = 10.0
        return logits
    return model_forward


def test_output_length_prompt_plus_new_tokens():
    _skip_if_missing()
    model = _make_model(fixed_token=3)
    prompt = torch.tensor([1, 2])
    out = autoregressive_generate(model, prompt, max_new_tokens=4)
    assert len(out) == 6, f"Expected len 6, got {len(out)}"


def test_prompt_prefix_is_unchanged():
    _skip_if_missing()
    model = _make_model(fixed_token=3)
    prompt = torch.tensor([1, 2, 5])
    out = autoregressive_generate(model, prompt, max_new_tokens=3)
    assert out[:3].tolist() == [1, 2, 5]


def test_eos_stops_generation_early():
    _skip_if_missing()
    model = _make_model(fixed_token=3)
    prompt = torch.tensor([0])
    # Model always generates token 3; set eos=3 → should stop after 1 new token
    out = autoregressive_generate(model, prompt, max_new_tokens=10, eos_token_id=3)
    assert out[-1].item() == 3
    assert len(out) == 2, f"Expected prompt + 1 generated token, got len {len(out)}"


def test_max_new_tokens_zero_returns_prompt():
    _skip_if_missing()
    model = _make_model(fixed_token=3)
    prompt = torch.tensor([4, 5])
    out = autoregressive_generate(model, prompt, max_new_tokens=0)
    assert out.tolist() == [4, 5]


def test_generated_tokens_in_valid_range():
    _skip_if_missing()
    model = _make_model(fixed_token=2)
    prompt = torch.tensor([0])
    out = autoregressive_generate(model, prompt, max_new_tokens=5)
    new_tokens = out[1:].tolist()
    assert all(0 <= t < VOCAB_SIZE for t in new_tokens), f"Out-of-range token: {new_tokens}"
