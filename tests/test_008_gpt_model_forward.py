"""
Tests for Problem 008 — Full GPT Forward Pass
"""
import importlib
import pytest
import torch

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.008_gpt_model_forward")
    gpt_model_forward = mod.gpt_model_forward
except ModuleNotFoundError:
    pytest.skip(
        "solutions/008_gpt_model_forward.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_params(
    vocab_size: int = 16,
    d_model: int = 8,
    d_ff: int = 32,
    n_heads: int = 2,
    n_blocks: int = 2,
    max_seq: int = 32,
    seed: int = 0,
) -> dict:
    """Return a minimal but structurally correct params dict."""
    torch.manual_seed(seed)
    I = torch.eye(d_model)
    block = {
        "W_q": I.clone(), "W_k": I.clone(), "W_v": I.clone(), "W_o": I.clone(),
        "gamma_1": torch.ones(d_model),  "beta_1": torch.zeros(d_model),
        "W1": torch.zeros(d_model, d_ff), "b1": torch.zeros(d_ff),
        "W2": torch.zeros(d_ff, d_model), "b2": torch.zeros(d_model),
        "gamma_2": torch.ones(d_model),  "beta_2": torch.zeros(d_model),
        "n_heads": n_heads,
    }
    return {
        "wte":        torch.randn(vocab_size, d_model),
        "wpe":        torch.randn(max_seq, d_model),
        "ln_f_gamma": torch.ones(d_model),
        "ln_f_beta":  torch.zeros(d_model),
        "blocks":     [block] * n_blocks,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGPTModelForward:

    def test_output_shape(self):
        """Output must be (batch, seq_len, vocab_size)."""
        V, D, S, B = 16, 8, 4, 1
        params = make_tiny_params(vocab_size=V, d_model=D)
        token_ids = torch.randint(0, V, (B, S))
        logits = gpt_model_forward(token_ids, params)
        assert logits.shape == torch.Size([B, S, V])

    def test_output_shape_larger_batch(self):
        """Batch dimension must be preserved."""
        V, D, S, B = 16, 8, 4, 3
        params = make_tiny_params(vocab_size=V, d_model=D)
        token_ids = torch.randint(0, V, (B, S))
        logits = gpt_model_forward(token_ids, params)
        assert logits.shape == torch.Size([B, S, V])

    def test_output_dtype_is_float32(self):
        V, D, S, B = 16, 8, 4, 1
        params = make_tiny_params(vocab_size=V, d_model=D)
        token_ids = torch.randint(0, V, (B, S))
        logits = gpt_model_forward(token_ids, params)
        assert logits.dtype == torch.float32

    def test_different_inputs_give_different_logits(self):
        """Two different token sequences should produce different logit tensors."""
        V, D, S, B = 16, 8, 4, 1
        params = make_tiny_params(vocab_size=V, d_model=D, seed=7)
        ids1 = torch.zeros(B, S, dtype=torch.long)
        ids2 = torch.ones(B, S, dtype=torch.long)
        logits1 = gpt_model_forward(ids1, params)
        logits2 = gpt_model_forward(ids2, params)
        assert not torch.allclose(logits1, logits2, atol=1e-4)

    def test_deterministic_given_same_input(self):
        """Same input must produce identical logits (no stochastic ops)."""
        V, D, S, B = 16, 8, 6, 2
        params = make_tiny_params(vocab_size=V, d_model=D)
        token_ids = torch.randint(0, V, (B, S))
        logits1 = gpt_model_forward(token_ids, params)
        logits2 = gpt_model_forward(token_ids, params)
        assert torch.allclose(logits1, logits2, atol=1e-6)

    def test_logits_are_not_probabilities(self):
        """Raw logits should NOT sum to 1 along the vocab dimension."""
        V, D, S, B = 16, 8, 4, 1
        params = make_tiny_params(vocab_size=V, d_model=D, seed=99)
        token_ids = torch.randint(0, V, (B, S))
        logits = gpt_model_forward(token_ids, params)
        vocab_sums = logits.sum(dim=-1)
        # If they were probabilities they would sum to 1.0; logits sum to something else
        assert not torch.allclose(vocab_sums, torch.ones_like(vocab_sums), atol=0.1)
