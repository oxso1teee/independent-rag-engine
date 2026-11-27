"""The generation metrics count statuses and refusals; they never claim to measure truth."""

from __future__ import annotations

import pytest

from rag_engine.domain import AskStatus
from rag_engine.evaluation.generation import GenerationSample, summarize_generation

pytestmark = pytest.mark.unit


def sample(
    status: AskStatus, *, answerable: bool, called: bool = True, degenerate: bool = False
) -> GenerationSample:
    return GenerationSample("q", answerable, status, called, degenerate, None)


def test_rates_are_computed_per_the_kind_of_question() -> None:
    summary = summarize_generation(
        [
            sample(AskStatus.ANSWERED, answerable=True),
            sample(AskStatus.INSUFFICIENT_EVIDENCE, answerable=True),  # a false refusal
            sample(AskStatus.INSUFFICIENT_EVIDENCE, answerable=False, called=False),
            sample(AskStatus.INSUFFICIENT_EVIDENCE, answerable=False),
            sample(AskStatus.ANSWERED, answerable=False),  # an answer that should not exist
            sample(AskStatus.UNVERIFIED, answerable=True),
            sample(AskStatus.ERROR, answerable=True, degenerate=True),
        ]
    )
    assert summary.questions == 7 and summary.answerable == 4 and summary.unanswerable == 3
    assert summary.by_status == {
        "answered": 2,
        "insufficient_evidence": 3,
        "unverified": 1,
        "error": 1,
    }
    assert summary.citation_validity == pytest.approx(2 / 3)
    assert summary.false_refusal_rate == pytest.approx(1 / 4)
    assert summary.correct_refusal_rate == pytest.approx(2 / 3)
    assert summary.unanswerable_answered == 1 and summary.llm_calls_avoided == 1
    assert summary.degenerate_rate == pytest.approx(1 / 7)


def test_missing_groups_give_none_not_zero() -> None:
    summary = summarize_generation([sample(AskStatus.INSUFFICIENT_EVIDENCE, answerable=True)])
    assert summary.citation_validity is None and summary.correct_refusal_rate is None
    assert summarize_generation([]).degenerate_rate == 0.0
