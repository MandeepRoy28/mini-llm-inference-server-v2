"""Tests for Problem 023 — Decode Phase."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.023_decode_phase")
    decode_phase = mod.decode_phase
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 023 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 10


def _greedy_forward(winner: int = 3):
    """Model that always assigns high logit to `winner`."""
    def forward(input_ids, cache):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, VOCAB)
        logits[:, :, winner] = 100.0
        return logits, cache
    return forward


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_tuple_of_int_and_list():
    """decode_phase must return (int, list)."""
    token, cache = decode_phase(
        _greedy_forward(), last_token_id=0, cache=[], step=0
    )
    assert isinstance(token, int)
    assert isinstance(cache, list)


def test_greedy_token_selected():
    """With a near-deterministic distribution the correct token must be picked."""
    winner = 7
    token, _ = decode_phase(
        _greedy_forward(winner), last_token_id=0, cache=[], step=0
    )
    assert token == winner


def test_model_receives_single_token_input():
    """model_forward_with_cache must receive input_ids of shape (1, 1)."""
    received_shapes = []

    def shape_recording_forward(input_ids, cache):
        received_shapes.append(input_ids.shape)
        logits = torch.zeros(1, 1, VOCAB)
        return logits, cache

    decode_phase(shape_recording_forward, last_token_id=5, cache=[], step=0)
    assert len(received_shapes) == 1
    assert received_shapes[0] == torch.Size([1, 1])


def test_last_token_id_used_as_input():
    """The last_token_id must appear as the input token to the model."""
    received_tokens = []

    def token_recording_forward(input_ids, cache):
        received_tokens.append(input_ids.item())
        logits = torch.zeros(1, 1, VOCAB)
        return logits, cache

    expected_token = 4
    decode_phase(token_recording_forward, last_token_id=expected_token, cache=[], step=0)
    assert received_tokens[0] == expected_token


def test_temperature_affects_distribution():
    """With temperature=0.01 (near-greedy), the dominant token must win consistently."""
    winner = 2
    fwd = _greedy_forward(winner)
    # Run 10 times; all should pick the winner
    for _ in range(10):
        token, _ = decode_phase(fwd, last_token_id=0, cache=[], step=0, temperature=0.01)
        assert token == winner
