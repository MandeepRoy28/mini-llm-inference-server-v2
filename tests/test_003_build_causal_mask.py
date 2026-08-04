"""
Tests for Problem 003 — Build a Causal Attention Mask
"""
import importlib
import pytest
import torch

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.003_build_causal_mask")
    build_causal_mask = mod.build_causal_mask
except ModuleNotFoundError:
    pytest.skip(
        "solutions/003_build_causal_mask.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildCausalMask:

    def test_shape(self):
        """Returned tensor must be square with side seq_len."""
        for n in [1, 3, 5, 10]:
            mask = build_causal_mask(n)
            assert mask.shape == torch.Size([n, n]), f"Wrong shape for seq_len={n}"

    def test_dtype_is_bool(self):
        """Returned tensor must be boolean."""
        mask = build_causal_mask(4)
        assert mask.dtype == torch.bool

    def test_canonical_3x3(self):
        """Exact values for the 3×3 case given in the problem statement."""
        expected = torch.tensor(
            [[False, True, True],
             [False, False, True],
             [False, False, False]]
        )
        assert torch.equal(build_causal_mask(3), expected)

    def test_diagonal_is_false(self):
        """A token is always allowed to attend to itself (diagonal = False)."""
        mask = build_causal_mask(6)
        for i in range(6):
            assert mask[i, i].item() is False, f"Diagonal [{i},{i}] should be False"

    def test_upper_triangle_is_true(self):
        """Every position above the diagonal (j > i) must be True."""
        mask = build_causal_mask(6)
        for i in range(6):
            for j in range(i + 1, 6):
                assert mask[i, j].item() is True, f"mask[{i},{j}] should be True"

    def test_lower_triangle_is_false(self):
        """Every position at or below the diagonal (j <= i) must be False."""
        mask = build_causal_mask(6)
        for i in range(6):
            for j in range(0, i + 1):
                assert mask[i, j].item() is False, f"mask[{i},{j}] should be False"

    def test_seq_len_1(self):
        """Edge case: single-token sequence."""
        mask = build_causal_mask(1)
        assert mask.shape == torch.Size([1, 1])
        assert mask[0, 0].item() is False
