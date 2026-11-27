"""Cutting a text to a token limit without producing text that is not in the original.

The result is always an exact prefix of the input, so offsets and quotes stay valid. The cut falls
at the last sentence end that fits; failing that at the last word; failing that at a character.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)\]]*(?=\s|$)")
_WORD = re.compile(r"\S+")


def _largest_fitting(ends: list[int], text: str, limit: int, count: Callable[[str], int]) -> int:
    """Index (1-based) of the longest prefix ending at one of ``ends`` that fits; 0 if none."""
    low, high = 0, len(ends)
    while low < high:
        middle = (low + high + 1) // 2
        if count(text[: ends[middle - 1]]) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


def truncate_to_tokens(text: str, limit: int, count: Callable[[str], int]) -> str:
    """The longest sensible prefix of ``text`` with at most ``limit`` tokens ('' if none fits)."""
    if limit <= 0:
        return ""
    if count(text) <= limit:
        return text
    sentences = [match.end() for match in _SENTENCE_END.finditer(text)]
    found = _largest_fitting(sentences, text, limit, count)
    if found:
        return text[: sentences[found - 1]]
    words = [match.end() for match in _WORD.finditer(text)]
    found = _largest_fitting(words, text, limit, count)
    if found:
        return text[: words[found - 1]]
    characters = list(range(1, len(text) + 1))
    found = _largest_fitting(characters, text, limit, count)
    return text[:found].rstrip()
