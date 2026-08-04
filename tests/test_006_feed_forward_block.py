"""
Tests for Problem 006 — Feed-Forward Block
"""
import importlib
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.006_feed_forward_block")
    feed_forward_block = mod.feed_forward_block
except ModuleNotFoundError:
    pytest.skip(
        "solutions/006_feed_forward_block.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeedForwardBlock:

    def test_output_shape(self):
        """Output shape must match input shape (batch, seq, d_model)."""
        B, S, D, D_FF = 2, 5, 16, 64
        x  = torch.randn(B, S, D)
        W1 = torch.randn(D, D_FF)
        b1 = torch.zeros(D_FF)
        W2 = torch.randn(D_FF, D)
        b2 = torch.zeros(D)
        out = feed_forward_block(x, W1, b1, W2, b2)
        assert out.shape == torch.Size([B, S, D])

    def test_output_dtype_preserved(self):
        B, S, D, D_FF = 1, 3, 8, 32
        x  = torch.randn(B, S, D)
        W1 = torch.randn(D, D_FF)
        b1 = torch.zeros(D_FF)
        W2 = torch.randn(D_FF, D)
        b2 = torch.zeros(D)
        out = feed_forward_block(x, W1, b1, W2, b2)
        assert out.dtype == torch.float32

    def test_zero_input_zero_bias_gives_zero_output(self):
        """GELU(0) = 0, so zero input with zero bias must yield zero output."""
        B, S, D, D_FF = 1, 2, 4, 16
        x  = torch.zeros(B, S, D)
        W1 = torch.randn(D, D_FF)
        b1 = torch.zeros(D_FF)
        W2 = torch.randn(D_FF, D)
        b2 = torch.zeros(D)
        out = feed_forward_block(x, W1, b1, W2, b2)
        assert torch.allclose(out, torch.zeros(B, S, D), atol=1e-6)

    def test_matches_manual_computation(self):
        """Directly verify the formula: GELU(x @ W1 + b1) @ W2 + b2."""
        torch.manual_seed(42)
        B, S, D, D_FF = 1, 3, 8, 32
        x  = torch.randn(B, S, D)
        W1 = torch.randn(D, D_FF)
        b1 = torch.randn(D_FF)
        W2 = torch.randn(D_FF, D)
        b2 = torch.randn(D)
        out = feed_forward_block(x, W1, b1, W2, b2)
        expected = F.gelu(x @ W1 + b1) @ W2 + b2
        assert torch.allclose(out, expected, atol=1e-5)

    def test_non_zero_bias_propagates(self):
        """A large positive b2 bias should shift every output value upward."""
        B, S, D, D_FF = 1, 2, 4, 16
        x  = torch.zeros(B, S, D)
        W1 = torch.zeros(D, D_FF)
        b1 = torch.zeros(D_FF)
        W2 = torch.zeros(D_FF, D)
        b2 = torch.ones(D) * 5.0   # shift output by 5
        out = feed_forward_block(x, W1, b1, W2, b2)
        # GELU(0)=0 → hidden = 0; 0 @ W2 + b2 = b2
        assert torch.allclose(out, torch.ones(B, S, D) * 5.0, atol=1e-5)
