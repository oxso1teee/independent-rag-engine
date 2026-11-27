"""Turning raw query text into what the retrieval channels receive."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from rag_engine.domain import ErrorCode, Language, RagError
from rag_engine.domain.language import detect_language
from rag_engine.ports import TokenCounter

MAX_CHARS = 20_000  # refuse before tokenising: a megabyte of text is not a question


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    text: str
    language: Language
    tokens: int


def _clean(text: str) -> str:
    """NFC, control characters to spaces, invisible format characters dropped, spaces collapsed."""
    out: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        category = unicodedata.category(char)
        if category == "Cf":
            continue  # zero-width and byte-order marks: invisible, so removed, not spaced
        out.append(" " if category in {"Cc", "Zs", "Zl", "Zp"} else char)
    return " ".join("".join(out).split())


class QueryProcessor:
    def __init__(self, counter: TokenCounter, *, max_tokens: int) -> None:
        if max_tokens < 1:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "the query token limit must be positive")
        self._counter = counter
        self._max_tokens = max_tokens

    def process(self, text: str, language_hint: Language | None = None) -> ProcessedQuery:
        if len(text) > MAX_CHARS:
            raise RagError(
                ErrorCode.E_QUERY_INVALID,
                "the query is too long",
                context={"chars": len(text), "limit": MAX_CHARS},
            )
        cleaned = _clean(text)
        if not cleaned:
            raise RagError(ErrorCode.E_QUERY_INVALID, "the query has nothing to search for")
        tokens = self._counter.count(cleaned)
        if tokens > self._max_tokens:
            raise RagError(
                ErrorCode.E_QUERY_INVALID,
                "the query exceeds the token limit",
                context={"tokens": tokens, "limit": self._max_tokens},
            )
        return ProcessedQuery(cleaned, language_hint or detect_language(cleaned), tokens)
