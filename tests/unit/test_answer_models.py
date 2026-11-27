"""Invariants of the answer-side models."""

from __future__ import annotations

import pytest

from rag_engine.domain import CitationCheck, CitationProblem, ErrorCode, RagError, SentenceCheck

pytestmark = pytest.mark.unit


def test_an_unjudged_sentence_is_never_a_failure() -> None:
    with pytest.raises(RagError) as info:
        SentenceCheck("Text.", ("E1",), None, False)
    assert info.value.code is ErrorCode.E_INTEGRITY


@pytest.mark.parametrize("coverage", [-0.1, 1.1])
def test_coverage_is_a_share(coverage: float) -> None:
    with pytest.raises(RagError):
        SentenceCheck("Text.", ("E1",), coverage, True)


def test_a_checked_sentence_cites_something() -> None:
    with pytest.raises(RagError):
        SentenceCheck("Text.", (), 1.0, True)


def test_a_check_is_ok_exactly_when_it_has_no_problems() -> None:
    clean = CitationCheck(("E1",), (), (), (), ())
    broken = CitationCheck((), (), (), (), (CitationProblem.NO_CITATION,))
    assert clean.ok and not broken.ok
    judged = SentenceCheck("A b c.", ("E1",), 1.0, True)
    unjudged = SentenceCheck("Yes.", ("E1",), None, True)
    assert CitationCheck(("E1",), (), (), (judged, unjudged), ()).unjudged == 1
