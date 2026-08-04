"""Tests for Problem 022 — Prefill Phase."""

import importlib
import pytest
import torch

try:
    mod = importlib.import_module("solutions.022_prefill_phase")
    prefill_phase = mod.prefill_phase
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 022 not written yet.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 20


def _make_forward(vocab=VOCAB, record_list=None):
    """Returns a minimal model_forward_with_cache that optionally records calls."""
    def forward(input_ids, cache):
        if record_list is not None:
            record_list.append(input_ids.shape)
        B, T = input_ids.shape
        logits = torch.zeros(B, T, vocab)
        logits[:, :, 5] = 1.0  # token 5 is most likely
        return logits, cache
    return forward


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_tuple_of_two():
    """Result must be a 2-tuple."""
    fwd = _make_forward()
    result = prefill_phase(fwd, torch.zeros(1, 6, dtype=torch.long), cache=[])
    assert isinstance(result, tuple) and len(result) == 2


def test_last_logits_shape():
    """last_logits must have shape (batch_size, vocab_size)."""
    B, T = 2, 8
    fwd = _make_forward()
    last_logits, _ = prefill_phase(
        fwd, torch.zeros(B, T, dtype=torch.long), cache=[]
    )
    assert last_logits.shape == torch.Size([B, VOCAB])


def test_cache_is_returned():
    """The second element must be a list (the cache)."""
    fwd = _make_forward()
    _, cache = prefill_phase(fwd, torch.zeros(1, 4, dtype=torch.long), cache=[])
    assert isinstance(cache, list)


def test_model_called_once_with_full_prompt():
    """model_forward_with_cache must be called exactly once for the full prompt."""
    calls = []
    fwd = _make_forward(record_list=calls)
    T = 10
    prefill_phase(fwd, torch.zeros(1, T, dtype=torch.long), cache=[])
    assert len(calls) == 1
    assert calls[0] == (1, T)


def test_last_logits_are_from_final_position():
    """last_logits must correspond to the last token position in the sequence."""
    vocab = 5
    T = 4
    sentinel = 99.0  # large value placed only at the last position

    def mock_forward(input_ids, cache):
        B, Tl = input_ids.shape
        logits = torch.zeros(B, Tl, vocab)
        logits[:, -1, 2] = sentinel   # mark last position
        return logits, cache

    last_logits, _ = prefill_phase(
        mock_forward, torch.zeros(1, T, dtype=torch.long), cache=[]
    )
    # Token 2 should have the sentinel value
    assert last_logits[0, 2].item() == pytest.approx(sentinel)
