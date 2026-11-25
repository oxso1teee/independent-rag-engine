"""Analyzer and MATCH construction contract for RU/EN FTS5."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import ErrorCode, Language, RagError
from rag_engine.lexical.analysis import analyze, safe_match


def test_ru_en_stemming_and_yo_normalization() -> None:
    assert analyze("Машины", Language.RU) == analyze("машина", Language.RU)
    assert analyze("Running", Language.EN) == analyze("runs", Language.EN)
    assert analyze("Ёжики", Language.RU) == analyze("ежики", Language.RU)


@given(st.text(max_size=80))
def test_match_expression_contains_only_generated_quoted_tokens(text: str) -> None:
    expression = safe_match(text)
    if expression is None:
        return
    for term in expression.split(" OR "):
        assert term.startswith('"') and term.endswith('"')
        assert all(char.isalnum() or char == "_" for char in term[1:-1])


def test_query_is_rejected_instead_of_silently_truncated() -> None:
    with pytest.raises(RagError) as error:
        safe_match("word " * 65)
    assert error.value.code is ErrorCode.E_RESOURCE_GATE


def test_fts_syntax_is_escaped() -> None:
    expression = safe_match('" OR * NEAR (secret)')
    assert expression is not None
    assert expression == '"or" OR "near" OR "secret"'
