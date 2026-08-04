"""
Tests for Problem 005 — Multi-Head Attention Forward Pass
"""
import importlib
import pytest
import torch

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.005_multi_head_attention_forward")
    multi_head_attention_forward = mod.multi_head_attention_forward
except ModuleNotFoundError:
    pytest.skip(
        "solutions/005_multi_head_attention_forward.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_identity_weights(d_model: int) -> tuple:
    I = torch.eye(d_model)
    return I, I, I, I  # W_q, W_k, W_v, W_o


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionForward:

    def test_output_shape_multi_head(self):
        """Output shape must be (batch, seq, d_model) for any n_heads."""
        B, S, D, H = 2, 5, 8, 2
        x = torch.randn(B, S, D)
        W_q, W_k, W_v, W_o = make_identity_weights(D)
        out = multi_head_attention_forward(x, W_q, W_k, W_v, W_o, n_heads=H)
        assert out.shape == torch.Size([B, S, D])

    def test_output_shape_single_head(self):
        """n_heads=1 is a valid edge case."""
        B, S, D = 1, 4, 8
        x = torch.randn(B, S, D)
        W_q, W_k, W_v, W_o = make_identity_weights(D)
        out = multi_head_attention_forward(x, W_q, W_k, W_v, W_o, n_heads=1)
        assert out.shape == torch.Size([B, S, D])

    def test_output_dtype_preserved(self):
        B, S, D, H = 1, 3, 8, 2
        x = torch.randn(B, S, D)
        W_q, W_k, W_v, W_o = make_identity_weights(D)
        out = multi_head_attention_forward(x, W_q, W_k, W_v, W_o, n_heads=H)
        assert out.dtype == torch.float32

    def test_causal_mask_applied(self):
        """With a causal mask, outputs at position 0 should only use position 0."""
        B, S, D, H = 1, 4, 8, 2
        x = torch.randn(B, S, D)
        W_q, W_k, W_v, W_o = make_identity_weights(D)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        out_masked = multi_head_attention_forward(x, W_q, W_k, W_v, W_o, n_heads=H, mask=mask)
        out_no_mask = multi_head_attention_forward(x, W_q, W_k, W_v, W_o, n_heads=H, mask=None)
        # With a causal mask outputs should generally differ from no-mask
        # (unless by extreme coincidence), and shape must still be correct
        assert out_masked.shape == torch.Size([B, S, D])
        # They should not be identical (with random input this would be astronomically unlikely)
        assert not torch.allclose(out_masked, out_no_mask, atol=1e-4)

    def test_larger_batch_and_heads(self):
        """Smoke test with a larger, more realistic configuration."""
        B, S, D, H = 4, 16, 64, 8
        x = torch.randn(B, S, D)
        W = torch.randn(D, D) * 0.02
        out = multi_head_attention_forward(x, W, W, W, W, n_heads=H)
        assert out.shape == torch.Size([B, S, D])
