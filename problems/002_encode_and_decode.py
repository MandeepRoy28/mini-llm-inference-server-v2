"""
Problem 002 — Encode and Decode Token Sequences
================================================
With a vocabulary in hand, the next step is to *use* it: convert raw text into
integer id sequences (encoding) and convert id sequences back to text
(decoding).

Difficulty : Easy
Tags        : tokenisation, encoding, decoding, NLP fundamentals
"""
from __future__ import annotations


def encode(text: str, token_to_id: dict) -> list[int]:
    """Convert a whitespace-tokenised string into a list of integer token ids.

    Splits *text* on whitespace and replaces each token with its id from
    *token_to_id*.  Every token in *text* is assumed to exist in
    *token_to_id* (no out-of-vocabulary handling required).

    Args:
        text (str): Raw input string.  Tokens are separated by whitespace.
        token_to_id (dict): Mapping from token string to integer id, as
            produced by :func:`build_token_vocab`.

    Returns:
        list[int]: Ordered list of integer ids corresponding to each token in
            *text*.  Returns an empty list when *text* is empty.

    Examples:
        >>> t2i = {'hello': 0, 'world': 1}
        >>> encode("hello world", t2i)
        [0, 1]

        >>> encode("hello hello world", {'hello': 0, 'world': 1})
        [0, 0, 1]

        >>> encode("", {'hello': 0})
        []
    """
    raise NotImplementedError


def decode(ids: list[int], id_to_token: dict) -> str:
    """Convert a list of integer token ids back into a whitespace-separated string.

    Looks up each id in *id_to_token* and joins the resulting tokens with a
    single space.  Every id is assumed to exist in *id_to_token*.

    Args:
        ids (list[int]): Sequence of integer token ids to decode.
        id_to_token (dict): Mapping from integer id to token string, as
            produced by :func:`build_token_vocab`.

    Returns:
        str: Space-joined string of tokens corresponding to *ids*.  Returns an
            empty string when *ids* is empty.

    Examples:
        >>> decode([0, 1], {0: 'hello', 1: 'world'})
        'hello world'

        >>> decode([0, 0, 1], {0: 'hello', 1: 'world'})
        'hello hello world'

        >>> decode([], {0: 'hello'})
        ''
    """
    raise NotImplementedError
