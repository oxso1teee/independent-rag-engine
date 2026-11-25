"""Stable RU/EN Snowball token normalization for FTS5."""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

from rag_engine.domain import ErrorCode, Language, RagError

ANALYZER_ID = "snowball-ru-en/1"
_TOKENS = re.compile(r"\w+", re.UNICODE)
_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")


def analyze(text: str, language: Language = Language.UND) -> str:
    """Normalize and stem tokens; mixed text selects a stemmer per token script."""
    normalized = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    result: list[str] = []
    for match in _TOKENS.finditer(normalized):
        token = match.group()
        if language is Language.RU or (language is Language.MIXED and _has_cyrillic(token)):
            token = _RU.stemWord(token)
        elif language is Language.EN or (language is Language.MIXED and _has_latin(token)):
            token = _EN.stemWord(token)
        result.append(token)
    return " ".join(result)


def _has_cyrillic(token: str) -> bool:
    return any("\u0400" <= char <= "\u052f" for char in token)


def _has_latin(token: str) -> bool:
    return any("a" <= char <= "z" for char in token)


def safe_match(query: str) -> str | None:
    """Only generated quoted terms reach MATCH; raw FTS syntax never does."""
    normalized = unicodedata.normalize("NFC", query).lower().replace("ё", "е")
    terms: list[str] = []
    tokens = _TOKENS.findall(normalized)
    if len(tokens) > 64:
        raise RagError(ErrorCode.E_RESOURCE_GATE, "lexical query exceeds 64 tokens")
    for token in tokens:
        stem = token
        if _has_cyrillic(token):
            stem = _RU.stemWord(token)
        elif _has_latin(token):
            stem = _EN.stemWord(token)
        # An unclassified document keeps the original token.
        terms.extend(('"' + value.replace('"', '""') + '"') for value in (stem, token))
    return " OR ".join(dict.fromkeys(terms)) or None
