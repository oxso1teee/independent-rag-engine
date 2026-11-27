"""Minimal generation metrics: citation validity, degenerate answers and refusal behaviour.

These measure what the pipeline guarantees mechanically and how it treats questions that have
no answer. They do **not** measure whether an answer is true: faithfulness, hallucination rate
and answer correctness need labelled data and a judge, and belong to the evaluation phase. As
with retrieval, results over unreviewed labels are exploratory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_engine.domain import AskStatus, ErrorCode
from rag_engine.evaluation.dataset import Question
from rag_engine.pipeline import PipelineAnswer


@dataclass(frozen=True, slots=True)
class GenerationSample:
    question_id: str
    answerable: bool
    status: AskStatus
    llm_called: bool
    degenerate: bool
    error_code: str | None


def score_answer(question: Question, answer: PipelineAnswer) -> GenerationSample:
    code = None if answer.error is None else answer.error.code.value
    return GenerationSample(
        question_id=question.id,
        answerable=question.answerable,
        status=answer.status,
        llm_called=answer.diagnostics.llm_called,
        degenerate=code == ErrorCode.E_DEGENERATE_ANSWER.value,
        error_code=code,
    )


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    questions: int
    answerable: int
    unanswerable: int
    by_status: dict[str, int]
    citation_validity: float | None
    """Of the replies that had text, the share whose citations held up (answered / (answered +
    unverified)); ``None`` when no reply had text."""
    degenerate_rate: float
    correct_refusal_rate: float | None
    """Of the unanswerable questions, the share that ended as insufficient evidence."""
    false_refusal_rate: float | None
    """Of the answerable questions, the share that ended as insufficient evidence."""
    unanswerable_answered: int
    """Unanswerable questions that nevertheless got an ``answered`` result (to be read by hand)."""
    llm_calls_avoided: int
    """Questions the gate or an empty result settled without calling the model."""


def summarize_generation(samples: Sequence[GenerationSample]) -> GenerationSummary:
    n = len(samples)
    by_status = {status.value: sum(s.status is status for s in samples) for status in AskStatus}
    with_text = by_status["answered"] + by_status["unverified"]
    yes = [s for s in samples if s.answerable]
    no = [s for s in samples if not s.answerable]

    def refused(group: Sequence[GenerationSample]) -> int:
        return sum(s.status is AskStatus.INSUFFICIENT_EVIDENCE for s in group)

    return GenerationSummary(
        questions=n,
        answerable=len(yes),
        unanswerable=len(no),
        by_status=by_status,
        citation_validity=by_status["answered"] / with_text if with_text else None,
        degenerate_rate=sum(s.degenerate for s in samples) / n if n else 0.0,
        correct_refusal_rate=refused(no) / len(no) if no else None,
        false_refusal_rate=refused(yes) / len(yes) if yes else None,
        unanswerable_answered=sum(s.status is AskStatus.ANSWERED for s in no),
        llm_calls_avoided=sum(
            s.status is AskStatus.INSUFFICIENT_EVIDENCE and not s.llm_called for s in samples
        ),
    )
