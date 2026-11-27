"""Query normalisation, the length limit and language detection."""

from __future__ import annotations

import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ErrorCode, Language, RagError
from rag_engine.retrieval.query import QueryProcessor

pytestmark = pytest.mark.unit


def processor(max_tokens: int = 64) -> QueryProcessor:
    return QueryProcessor(HeuristicTokenCounter(), max_tokens=max_tokens)


def test_text_is_nfc_trimmed_and_whitespace_collapsed() -> None:
    assert processor().process("  école \n\t  de musique  ").text == "école de musique"


def test_control_and_invisible_characters_never_reach_the_index() -> None:
    assert processor().process("a\x00b\x07c").text == "a b c"
    assert processor().process("zero​width﻿").text == "zerowidth"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", "\x00\x00", "​"])
def test_a_query_with_nothing_to_search_for_is_refused(blank: str) -> None:
    with pytest.raises(RagError) as info:
        processor().process(blank)
    assert info.value.code is ErrorCode.E_QUERY_INVALID


def test_a_query_over_the_token_limit_is_refused_not_truncated() -> None:
    long = " ".join(["word"] * 200)
    with pytest.raises(RagError) as info:
        processor(max_tokens=64).process(long)
    assert info.value.code is ErrorCode.E_QUERY_INVALID
    context = dict(info.value.context or {})
    assert context["limit"] == 64 and isinstance(context["tokens"], int) and context["tokens"] > 64


def test_an_absurdly_long_text_is_refused_before_tokenising() -> None:
    with pytest.raises(RagError) as info:
        processor().process("x" * 100_000)
    assert info.value.code is ErrorCode.E_QUERY_INVALID


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Как настроить индексацию документов?", Language.RU),
        ("How do I configure document indexing?", Language.EN),
        ("настроить indexing документов and search", Language.MIXED),
        ("12345 ???", Language.UND),
    ],
)
def test_language_follows_the_script(text: str, expected: Language) -> None:
    assert processor().process(text).language is expected


def test_a_language_hint_wins_over_detection() -> None:
    assert processor().process("12345", language_hint=Language.RU).language is Language.RU


def test_token_count_is_reported() -> None:
    assert processor().process("hello world").tokens >= 2
