import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.016_generate_with_prompt")
    generate_with_prompt = mod.generate_with_prompt
except Exception:
    mod = None
    generate_with_prompt = None


def _skip_if_missing():
    if generate_with_prompt is None:
        pytest.skip("Solution not implemented yet")


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #

VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
TOKEN_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_TOKEN = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def _make_model(fixed_token: int):
    """Dummy model that always predicts the same token."""
    def model_forward(ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.shape[0]
        logits = torch.zeros(seq_len, VOCAB_SIZE)
        logits[:, fixed_token] = 10.0
        return logits
    return model_forward


def test_output_is_string():
    _skip_if_missing()
    model = _make_model(fixed_token=TOKEN_TO_ID["a"])
    result = generate_with_prompt("hi", TOKEN_TO_ID, ID_TO_TOKEN, model, max_new_tokens=3)
    assert isinstance(result, str)


def test_prompt_preserved_in_output():
    _skip_if_missing()
    model = _make_model(fixed_token=TOKEN_TO_ID["z"])
    prompt = "hello"
    result = generate_with_prompt(prompt, TOKEN_TO_ID, ID_TO_TOKEN, model, max_new_tokens=5)
    assert result.startswith(prompt), f"Output '{result}' does not start with prompt '{prompt}'"


def test_correct_number_of_generated_chars():
    _skip_if_missing()
    model = _make_model(fixed_token=TOKEN_TO_ID["b"])
    prompt = "hi"
    max_new = 4
    result = generate_with_prompt(prompt, TOKEN_TO_ID, ID_TO_TOKEN, model, max_new_tokens=max_new)
    # prompt length + max_new generated chars
    assert len(result) == len(prompt) + max_new, f"Got '{result}' (len {len(result)})"


def test_unknown_chars_in_prompt_skipped():
    _skip_if_missing()
    model = _make_model(fixed_token=TOKEN_TO_ID["c"])
    # "!" is not in VOCAB and should be silently skipped
    result = generate_with_prompt("hi!", TOKEN_TO_ID, ID_TO_TOKEN, model, max_new_tokens=2)
    # "hi" = 2 valid chars encoded, + 2 generated
    assert len(result) == 4, f"Got '{result}' (len {len(result)})"


def test_empty_prompt_generates_tokens():
    _skip_if_missing()
    model = _make_model(fixed_token=TOKEN_TO_ID["d"])
    result = generate_with_prompt("", TOKEN_TO_ID, ID_TO_TOKEN, model, max_new_tokens=3)
    assert len(result) == 3, f"Expected 3 generated chars from empty prompt, got '{result}'"
