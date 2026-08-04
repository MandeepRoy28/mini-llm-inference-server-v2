"""
Tests for Problem 001 — Build a Token Vocabulary
"""
import importlib
import pytest

# ---------------------------------------------------------------------------
# Import guard: skip the entire module if the solution doesn't exist yet
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.001_build_token_vocab")
    build_token_vocab = mod.build_token_vocab
except ModuleNotFoundError:
    pytest.skip(
        "solutions/001_build_token_vocab.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildTokenVocab:

    def test_basic_example(self):
        """Canonical example from the problem statement."""
        t2i, i2t = build_token_vocab("hello world hello")
        assert t2i == {"hello": 0, "world": 1}
        assert i2t == {0: "hello", 1: "world"}

    def test_empty_string_returns_empty_dicts(self):
        """Empty input should produce two empty dicts, not raise."""
        t2i, i2t = build_token_vocab("")
        assert t2i == {}
        assert i2t == {}

    def test_return_type_is_tuple_of_dicts(self):
        """Return value must be a tuple containing exactly two dicts."""
        result = build_token_vocab("a b c")
        assert isinstance(result, tuple)
        assert len(result) == 2
        t2i, i2t = result
        assert isinstance(t2i, dict)
        assert isinstance(i2t, dict)

    def test_first_appearance_ordering(self):
        """IDs must reflect order of first appearance, not alphabetical order."""
        t2i, i2t = build_token_vocab("the cat sat on the mat")
        # 'the' appears first → id 0; 'cat' second → id 1, etc.
        assert t2i["the"] == 0
        assert t2i["cat"] == 1
        assert t2i["sat"] == 2
        assert t2i["on"] == 3
        assert t2i["mat"] == 4
        # 'the' is a duplicate and must NOT get a new id
        assert len(t2i) == 5

    def test_mappings_are_inverse_of_each_other(self):
        """id_to_token must be the exact inverse of token_to_id."""
        text = "roses are red violets are blue"
        t2i, i2t = build_token_vocab(text)
        for token, idx in t2i.items():
            assert i2t[idx] == token
        for idx, token in i2t.items():
            assert t2i[token] == idx

    def test_single_token(self):
        """Single-word input: one entry in each dict."""
        t2i, i2t = build_token_vocab("hello")
        assert t2i == {"hello": 0}
        assert i2t == {0: "hello"}

    def test_ids_are_contiguous_from_zero(self):
        """IDs must be 0, 1, 2, ... with no gaps."""
        t2i, _ = build_token_vocab("a b c d e")
        assert sorted(t2i.values()) == list(range(len(t2i)))
