"""
Problem 001 — Build a Token Vocabulary
=======================================
Given a raw text corpus, construct the two lookup dictionaries that every
tokeniser needs: token→id (for encoding) and id→token (for decoding).

Difficulty : Easy
Tags        : tokenisation, vocabulary, NLP fundamentals
"""


def build_token_vocab(text: str) -> tuple[dict, dict]:
    """Build token-to-id and id-to-token mappings from a whitespace-tokenised corpus.

    Splits *text* on whitespace to produce a sequence of tokens, then assigns
    each **unique** token an integer id in the order of its **first appearance**.
    Duplicate tokens do not get a new id.

    Args:
        text (str): A raw string corpus.  Tokens are delimited by one or more
            whitespace characters (spaces, tabs, newlines).  An empty string
            produces empty dictionaries.

    Returns:
        tuple[dict, dict]: A two-element tuple ``(token_to_id, id_to_token)``
            where

            * ``token_to_id``  maps each unique token string to its integer id.
            * ``id_to_token``  maps each integer id back to its token string.

            Ids are contiguous integers starting from 0 and assigned in
            first-appearance order.

    Examples:
        >>> build_token_vocab("hello world hello")
        ({'hello': 0, 'world': 1}, {0: 'hello', 1: 'world'})

        >>> build_token_vocab("the cat sat on the mat")
        ({'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4},
         {0: 'the', 1: 'cat', 2: 'sat', 3: 'on', 4: 'mat'})

        >>> build_token_vocab("")
        ({}, {})
    """
    raise NotImplementedError
