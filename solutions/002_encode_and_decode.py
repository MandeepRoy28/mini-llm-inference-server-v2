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
    encode_list = []

    for s in text.split() :
        encode_list.append(token_to_id[s])

    return encode_list

def decode(ids: list[int], id_to_token: dict) -> str:
    text = ""
    for i in ids :
        text+=id_to_token[i]
        text+=" "
    
    return text[:-1]
