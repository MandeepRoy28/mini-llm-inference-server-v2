"""
Tests for Problem 007 — GPT-2 Style Transformer Block
"""
import importlib
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.007_transformer_block")
    transformer_block = mod.transformer_block
except ModuleNotFoundError:
    pytest.skip(
        "solutions/007_transformer_block.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_block_params(d_model: int, d_ff: int, n_heads: int) -> tuple[dict, dict]:
    """Build minimal valid attn_params and ffn_params dicts."""
    I = torch.eye(d_model)
    attn_params = {
        "W_q": I.clone(), "W_k": I.clone(), "W_v": I.clone(), "W_o": I.clone(),
        "gamma_1": torch.ones(d_model), "beta_1": torch.zeros(d_model),
    }
    ffn_params = {
        "W1": torch.zeros(d_model, d_ff), "b1": torch.zeros(d_ff),
        "W2": torch.zeros(d_ff, d_model), "b2": torch.zeros(d_model),
        "gamma_2": torch.ones(d_model), "beta_2": torch.zeros(d_model),
    }
    return attn_params, ffn_params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransformerBlock:

    def test_output_shape(self):
        """Output must have the same shape as the input."""
        B, S, D, D_FF, H = 2, 6, 16, 64, 4
        x = torch.randn(B, S, D)
        attn_params, ffn_params = make_block_params(D, D_FF, H)
        out = transformer_block(x, attn_params, ffn_params, n_heads=H)
        assert out.shape == x.shape

    def test_output_dtype_preserved(self):
        B, S, D, D_FF, H = 1, 4, 8, 32, 2
        x = torch.randn(B, S, D)
        attn_params, ffn_params = make_block_params(D, D_FF, H)
        out = transformer_block(x, attn_params, ffn_params, n_heads=H)
        assert out.dtype == torch.float32

    def test_all_zero_ffn_residual_only(self):
        """With all-zero FFN weights and biases, the FFN sub-layer contributes
        nothing so the output equals x + attn_output(x).  We simply verify
        the shape and that the result is not identically x (the attn branch
        should modify x)."""
        B, S, D, D_FF, H = 1, 4, 8, 32, 2
        x = torch.randn(B, S, D)
        attn_params, ffn_params = make_block_params(D, D_FF, H)
        out = transformer_block(x, attn_params, ffn_params, n_heads=H)
        assert out.shape == x.shape
        # The attention residual should have modified x
        assert not torch.allclose(out, x, atol=1e-4)

    def test_residual_connection_present(self):
        """Zero-out the attention projection (W_o = 0) AND the FFN weights;
        with both sub-layers producing zero output, residual connections
        ensure out == x (after LayerNorm has been applied to a normalised x
        this may not hold exactly, but with gamma=1 and beta=0, LN(x) ≠ x,
        so we just check shape here and that at least one connection is live)."""
        B, S, D, D_FF, H = 1, 3, 8, 32, 2
        x = torch.randn(B, S, D)
        attn_params, ffn_params = make_block_params(D, D_FF, H)
        # Zero out W_o so MHA sub-layer produces zero output
        attn_params["W_o"] = torch.zeros(D, D)
        out = transformer_block(x, attn_params, ffn_params, n_heads=H)
        # FFN is also zero, so out = x + 0 + 0 = x  (both residuals)
        assert torch.allclose(out, x, atol=1e-5), (
            "With zero W_o and zero FFN, both sub-layers emit 0 "
            "so residuals must return x unchanged."
        )

    def test_causal_mask_changes_output(self):
        """Applying a causal mask should change the output vs no mask."""
        B, S, D, D_FF, H = 1, 5, 8, 32, 2
        torch.manual_seed(0)
        x = torch.randn(B, S, D)
        attn_params, ffn_params = make_block_params(D, D_FF, H)
        # Use random (non-identity) W matrices so masking actually matters
        for key in ("W_q", "W_k", "W_v"):
            attn_params[key] = torch.randn(D, D) * 0.1
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        out_masked = transformer_block(x, attn_params, ffn_params, n_heads=H, mask=mask)
        out_no_mask = transformer_block(x, attn_params, ffn_params, n_heads=H, mask=None)
        assert not torch.allclose(out_masked, out_no_mask, atol=1e-4)
