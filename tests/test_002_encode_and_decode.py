"""
Tests for Problem 002 — Encode and Decode Token Sequences
"""
import importlib
import pytest

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
try:
    mod = importlib.import_module("solutions.002_encode_and_decode")
    encode = mod.encode
    decode = mod.decode
except ModuleNotFoundError:
    pytest.skip(
        "solutions/002_encode_and_decode.py not found — skipping",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T2I = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
I2T = {v: k for k, v in T2I.items()}


# ---------------------------------------------------------------------------
# Tests — encode
# ---------------------------------------------------------------------------

class TestEncode:

    def test_basic_encode(self):
        assert encode("hello world", T2I) == [0, 1]

    def test_encode_repeated_token(self):
        assert encode("hello hello world", T2I) == [0, 0, 1]

    def test_encode_single_token(self):
        assert encode("foo", T2I) == [2]

    def test_encode_empty_string_returns_empty_list(self):
        result = encode("", T2I)
        assert result == []

    def test_encode_return_type_is_list_of_int(self):
        result = encode("hello world", T2I)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_encode_preserves_order(self):
        assert encode("bar foo hello world", T2I) == [3, 2, 0, 1]


# ---------------------------------------------------------------------------
# Tests — decode
# ---------------------------------------------------------------------------

class TestDecode:

    def test_basic_decode(self):
        assert decode([0, 1], I2T) == "hello world"

    def test_decode_repeated_id(self):
        assert decode([0, 0, 1], I2T) == "hello hello world"

    def test_decode_single_id(self):
        assert decode([2], I2T) == "foo"

    def test_decode_empty_list_returns_empty_string(self):
        result = decode([], I2T)
        assert result == ""

    def test_decode_return_type_is_str(self):
        assert isinstance(decode([0, 1], I2T), str)


# ---------------------------------------------------------------------------
# Tests — round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_encode_then_decode_is_identity(self):
        """encode followed by decode must recover the original string."""
        text = "hello world foo bar hello"
        assert decode(encode(text, T2I), I2T) == text

    def test_decode_then_encode_is_identity(self):
        """decode followed by encode must recover the original id list."""
        ids = [3, 2, 0, 1, 0]
        assert encode(decode(ids, I2T), T2I) == ids
