"""Text splitting for the chunkers.

Every function returns pieces whose concatenation is exactly the input (separators stay
attached to the piece before them), so offsets can always be recovered by summing lengths.
"""

from __future__ import annotations

import re
import unicodedata

_ABBREVIATIONS = frozenset(
    {
        # Russian
        *("т.е", "т.д", "т.п", "т.к", "т.н", "т.о", "и.о", "г", "гг", "ул", "пр", "просп", "д"),
        *("кв", "рис", "табл", "см", "стр", "др", "проф", "акад", "им", "млн", "млрд", "тыс"),
        *("руб", "коп", "обл", "р-н", "пос", "с", "ст", "тел", "англ", "рус", "напр", "в"),
        # English
        *("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e", "fig", "no"),
        *("inc", "ltd", "co", "st", "approx", "dept", "cf", "vol", "pp", "al", "u.s", "u.k"),
    }
)
# terminal punctuation, optional closing quotes/brackets, then whitespace
_CANDIDATE = re.compile(r"([.!?…]+)([\"'»”’)\]]*)(\s+)")
_WORD_BEFORE = re.compile(r"([^\W\d_](?:[\w.\-]*[^\W_])?)$")
_NEXT_STARTS_SENTENCE = re.compile(r"[\"'«“‘(\[—\-–]*[^\W_a-zа-яё]", re.UNICODE)
_LOOKBACK = 40  # characters before a full stop that can belong to an abbreviation
_NUMBER_BEFORE = re.compile(r"(\d+)$")
_WORD_PIECES = re.compile(r"\s*\S+\s*|\s+")


def _is_boundary(text: str, match: re.Match[str]) -> bool:
    terminal = match.group(1)
    end = match.end()
    if end >= len(text):
        return True
    starts_sentence = _NEXT_STARTS_SENTENCE.match(text, end) is not None
    if terminal.startswith("\u2026") or terminal.endswith("..."):
        # an ellipsis ends a sentence only when a capital letter or digit follows
        return starts_sentence
    if "." in terminal and not match.group(2):
        window = text[max(0, match.start() - _LOOKBACK) : match.start()]  # bounded lookback
        word = _WORD_BEFORE.search(window)
        if word is not None:
            token = word.group(1).lower()
            letters = token.replace(".", "")
            if token in _ABBREVIATIONS:
                return False
            if len(letters) == 1 and letters.isalpha():
                return False  # an initial: "A. S. Pushkin"
        number = _NUMBER_BEFORE.search(window)
        if number is not None and len(number.group(1)) <= 2:
            before = window[: number.start(1)].rstrip(" \t")
            if not before or before[-1] in ".!?\u2026:;\n":
                return False  # a list marker: "1. First item"
    return starts_sentence


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text``; a boundary needs terminal punctuation, a space and a capital (or
    digit, quote, bracket, dash) after it, and abbreviations and initials do not end one."""
    pieces: list[str] = []
    start = 0
    for match in _CANDIDATE.finditer(text):
        if _is_boundary(text, match):
            pieces.append(text[start : match.end()])
            start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def split_words(text: str) -> list[str]:
    """Words, each with the whitespace that follows it (leading whitespace joins the first)."""
    return _WORD_PIECES.findall(text)


def split_lines(text: str) -> list[str]:
    """Lines with their line breaks."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def split_hard(text: str, max_chars: int) -> list[str]:
    """Pieces of at most ``max_chars`` characters, never separating a combining mark or a joiner
    from its base character."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        while end < len(text) and end > start + 1 and _attaches_to_previous(text[end]):
            end -= 1
        if end < len(text) and end == start + 1 and _attaches_to_previous(text[end]):
            end = min(start + max_chars, len(text))  # nothing better: keep the size bound
        pieces.append(text[start:end])
        start = end
    return pieces


def _attaches_to_previous(char: str) -> bool:
    return unicodedata.category(char) in ("Mn", "Me", "Mc") or char in "‌‍"
