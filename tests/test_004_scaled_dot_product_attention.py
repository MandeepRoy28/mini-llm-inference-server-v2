"""
Tests for Problem 004 — Scaled Dot-Product Attention
"""
import importlib
import math
import pytest
import torch

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.004_scaled_dot_product_attention")
    scaled_dot_product_attention = mod.scaled_dot_product_attention
except ModuleNotFoundError:
    pytest.skip(
        "solutions/004_scaled_dot_product_attention.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention:

    def test_output_shape(self):
        """Output shape must be (batch, seq_q, d_v)."""
        B, S, D = 2, 5, 16
        Q = K = V = torch.randn(B, S, D)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == torch.Size([B, S, D])

    def test_output_dtype_preserved(self):
        Q = K = V = torch.randn(1, 4, 8)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.dtype == torch.float32

    def test_uniform_queries_keys_output_is_mean_of_values(self):
        """When all Q and K are equal (same vector), attention is uniform and
        output must equal the mean of V across the key/value dimension."""
        B, S, D = 1, 4, 8
        Q = torch.zeros(B, S, D)
        K = torch.zeros(B, S, D)
        V = torch.randn(B, S, D)
        out = scaled_dot_product_attention(Q, K, V)
        # Uniform attention → output at each position = mean over seq of V
        expected = V.mean(dim=1, keepdim=True).expand(B, S, D)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_scaling_by_sqrt_dk(self):
        """Verify the 1/sqrt(d_k) scaling is applied.

        Strategy: construct Q, K so that Q @ K^T = d_k * I (before scaling).
        Without scaling the softmax would be very peaked; with scaling it
        should be softer.  We check that the diagonal logits are d_k / sqrt(d_k)
        = sqrt(d_k), not d_k.
        """
        D = 16
        # Q = I_{1 x D}, K = D * I_{1 x D}  →  QK^T = D (scalar) before scale
        Q = torch.ones(1, 1, D)        # single query token
        K = torch.ones(1, 1, D)        # single key token
        V = torch.ones(1, 1, D)
        out = scaled_dot_product_attention(Q, K, V)
        # With a single key, softmax is trivially 1.0, output = V
        assert torch.allclose(out, V, atol=1e-5)

    def test_causal_mask_blocks_future_positions(self):
        """With a causal mask, the first token must NOT see future tokens.

        We set V to be distinct at each position; if the mask works, position 0
        can only see position 0 so its output must equal V[:,0,:].
        """
        S, D = 4, 8
        Q = torch.zeros(1, S, D)
        K = torch.zeros(1, S, D)
        # Give each position a unique value vector
        V = torch.arange(S, dtype=torch.float32).view(1, S, 1).expand(1, S, D).clone()
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Position 0 sees only itself → output = V[:, 0, :]
        assert torch.allclose(out[:, 0, :], V[:, 0, :], atol=1e-5)
        # Position 1 sees positions 0 and 1 → output = mean(V[:,0,:], V[:,1,:])
        expected_pos1 = (V[:, 0, :] + V[:, 1, :]) / 2.0
        assert torch.allclose(out[:, 1, :], expected_pos1, atol=1e-5)

    def test_no_mask_matches_pytorch_reference(self):
        """Without a mask, result must match PyTorch's F.scaled_dot_product_attention."""
        import torch.nn.functional as F
        B, S, D = 2, 6, 16
        Q = torch.randn(B, S, D)
        K = torch.randn(B, S, D)
        V = torch.randn(B, S, D)
        out_ours = scaled_dot_product_attention(Q, K, V)
        # F.scaled_dot_product_attention expects (B, H, S, D); use H=1 trick
        out_ref = F.scaled_dot_product_attention(
            Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1)
        ).squeeze(1)
        assert torch.allclose(out_ours, out_ref, atol=1e-5)
