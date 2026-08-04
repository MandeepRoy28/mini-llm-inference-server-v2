"""
Tests for Problem 009 — Load Pre-Trained DistilGPT-2 Weights

NOTE: These tests download DistilGPT-2 from HuggingFace (~330 MB) on first run.
They are automatically skipped when:
  - the solution file doesn't exist, OR
  - the `transformers` library is not installed.
"""
import importlib
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip if transformers is not available (keep CI fast without GPU deps)
# ---------------------------------------------------------------------------
transformers = pytest.importorskip(
    "transformers",
    reason="transformers library not installed — skipping pretrained weight tests",
)

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.009_load_pretrained_weights")
    load_pretrained_weights = mod.load_pretrained_weights
except ModuleNotFoundError:
    pytest.skip(
        "solutions/009_load_pretrained_weights.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# DistilGPT-2 architecture constants
VOCAB_SIZE  = 50257
D_MODEL     = 768
MAX_SEQ     = 1024
N_BLOCKS    = 6
D_FF        = 3072
N_HEADS     = 12


@pytest.fixture(scope="module")
def params():
    """Load weights once for all tests in this module."""
    return load_pretrained_weights("distilgpt2")


class TestLoadPretrainedWeights:

    def test_return_type_is_dict(self, params):
        assert isinstance(params, dict)

    def test_top_level_keys_present(self, params):
        for key in ("wte", "wpe", "ln_f_gamma", "ln_f_beta", "blocks"):
            assert key in params, f"Missing top-level key: '{key}'"

    def test_wte_shape(self, params):
        assert params["wte"].shape == torch.Size([VOCAB_SIZE, D_MODEL]), (
            f"wte shape {params['wte'].shape} != ({VOCAB_SIZE}, {D_MODEL})"
        )

    def test_wpe_shape(self, params):
        assert params["wpe"].shape == torch.Size([MAX_SEQ, D_MODEL])

    def test_ln_f_shapes(self, params):
        assert params["ln_f_gamma"].shape == torch.Size([D_MODEL])
        assert params["ln_f_beta"].shape  == torch.Size([D_MODEL])

    def test_number_of_blocks(self, params):
        assert len(params["blocks"]) == N_BLOCKS

    def test_block_weight_shapes(self, params):
        """Each block must have all required keys with correct shapes."""
        for i, block in enumerate(params["blocks"]):
            for wname in ("W_q", "W_k", "W_v", "W_o"):
                assert block[wname].shape == torch.Size([D_MODEL, D_MODEL]), (
                    f"Block {i} {wname} has wrong shape: {block[wname].shape}"
                )
            assert block["W1"].shape == torch.Size([D_MODEL, D_FF])
            assert block["b1"].shape == torch.Size([D_FF])
            assert block["W2"].shape == torch.Size([D_FF, D_MODEL])
            assert block["b2"].shape == torch.Size([D_MODEL])
            for lnname in ("gamma_1", "beta_1", "gamma_2", "beta_2"):
                assert block[lnname].shape == torch.Size([D_MODEL]), (
                    f"Block {i} {lnname} has wrong shape: {block[lnname].shape}"
                )

    def test_n_heads_in_each_block(self, params):
        for i, block in enumerate(params["blocks"]):
            assert "n_heads" in block, f"Block {i} missing 'n_heads'"
            assert block["n_heads"] == N_HEADS, (
                f"Block {i} n_heads={block['n_heads']}, expected {N_HEADS}"
            )

    def test_weights_are_tensors(self, params):
        """All weight values must be torch.Tensor instances."""
        for block in params["blocks"]:
            for key, val in block.items():
                if key != "n_heads":
                    assert isinstance(val, torch.Tensor), (
                        f"Block key '{key}' is not a Tensor: {type(val)}"
                    )

    def test_weights_are_not_all_zero(self, params):
        """Sanity check: real pretrained weights should not be identically zero."""
        assert params["wte"].abs().sum() > 0
        assert params["wpe"].abs().sum() > 0
        for block in params["blocks"]:
            assert block["W_q"].abs().sum() > 0
