"""Generation evaluation of a running pipeline: the measurements a real model has to be judged by.

``evaluate_answers`` asks every question through a pipeline, times each answer, and for every
answered reply checks its claims against the evidence it cites with a ``ClaimJudge``.
``summarize_records`` turns the records into the metrics of the evaluation protocol: answer
correctness and completeness, faithfulness, citation precision and recall, hallucination rate, the
answerable-by-refused matrix and latency. The functions work with any pipeline, so they are tested
with a scripted model and run for real with a local one; which judge was used is part of the result,
because a lexical judge and a language-model judge are not the same instrument.
"""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag_engine.domain import AskStatus, Evidence, RetrievalQuery
from rag_engine.evaluation.dataset import Question
from rag_engine.evaluation.generation_metrics import (
    citation_score,
    correctness,
    faithfulness,
    hallucination_rate,
    no_answer_matrix,
)
from rag_engine.evaluation.judge import ClaimJudge, Verdict
from rag_engine.generation.citations import extract_markers, split_sentences
from rag_engine.pipeline import PipelineAnswer

_MARKER = re.compile(r"\[E[1-9][0-9]*\]")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_WORD = re.compile(r"\w+")


class Asker(Protocol):
    """What the evaluation needs from a pipeline: answer one retrieval query."""

    async def ask(self, query: RetrievalQuery) -> PipelineAnswer: ...


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    question_id: str
    answerable: bool
    status: AskStatus
    text: str
    seconds: float
    cited_documents: tuple[str, ...]
    gold_documents: tuple[str, ...]
    claim_verdicts: tuple[bool, ...]
    claim_supported_by_citation: tuple[bool, ...]
    fact_coverage: float | None
    exact: bool | None
    prompt_tokens: int
    completion_tokens: int
    error_code: str | None = None
    markers_total: int = 0
    markers_valid: int = 0


def fact_present(answer: str, fact: str) -> bool:
    """A fact is present when its numbers are all in the answer, or most of its words are."""
    body = _MARKER.sub(" ", answer).lower()
    numbers = {n.rstrip(".,:") for n in _NUMBER.findall(fact)}
    if numbers:
        return numbers <= {n.rstrip(".,:") for n in _NUMBER.findall(body)}
    wanted = {w for w in _WORD.findall(fact.lower()) if len(w) > 3}
    return bool(wanted) and len(wanted & set(_WORD.findall(body))) / len(wanted) >= 0.6


def claims_with_evidence(text: str, sources: Sequence[Evidence]) -> list[tuple[str, list[str]]]:
    """Each sentence of an answer without its markers, with the texts of the evidence it cites."""
    by_id = {item.evidence_id: item.text for item in sources}
    claims = []
    for sentence in split_sentences(text):
        plain = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_MARKER.sub("", sentence).split()))
        if any(ch.isalnum() for ch in plain):
            cited = [by_id[m] for m in extract_markers(sentence) if m in by_id]
            claims.append((plain, cited))
    return claims


async def evaluate_answers(
    pipeline: Asker,
    questions: Sequence[Question],
    judge: ClaimJudge,
    make_query: Callable[[Question], RetrievalQuery],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> list[AnswerRecord]:
    records: list[AnswerRecord] = []
    for question in questions:
        began = clock()
        answer = await pipeline.ask(make_query(question))
        seconds = clock() - began
        verdicts: list[bool] = []
        cited_ok: list[bool] = []
        if answer.status is AskStatus.ANSWERED:
            for claim, evidence in claims_with_evidence(answer.text, answer.sources):
                support = await judge.judge(claim, evidence) if evidence else Verdict.UNSUPPORTED
                verdicts.append(support is Verdict.SUPPORTED)
                cited_ok.append(support is Verdict.SUPPORTED and bool(evidence))
        facts = question.reference_facts if question.answerable else ()
        match = correctness(answer.text, question.reference_answer, ()) if answer.text else None
        usage = answer.diagnostics.usage
        records.append(
            AnswerRecord(
                question_id=question.id,
                answerable=question.answerable,
                status=answer.status,
                text=answer.text,
                seconds=seconds,
                cited_documents=tuple(sorted({c.locator.source_key for c in answer.citations})),
                gold_documents=tuple(sorted({ref.doc for ref in question.gold})),
                claim_verdicts=tuple(verdicts),
                claim_supported_by_citation=tuple(cited_ok),
                fact_coverage=(
                    sum(fact_present(answer.text, f) for f in facts) / len(facts)
                    if facts and answer.status is AskStatus.ANSWERED
                    else None
                ),
                exact=None if match is None else match.exact,
                prompt_tokens=0 if usage is None else usage.prompt_tokens,
                completion_tokens=0 if usage is None else usage.completion_tokens,
                error_code=None if answer.error is None else answer.error.code.value,
                markers_total=len(extract_markers(answer.raw_text or answer.text)),
                markers_valid=len(answer.citations),
            )
        )
    return records


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


@dataclass(frozen=True, slots=True)
class RunSummary:
    questions: int
    by_status: Mapping[str, int]
    answer_correctness: float | None
    completeness: float | None
    faithfulness: float | None
    citation_precision: float | None
    citation_recall: float | None
    hallucination_rate: float | None
    no_answer: Mapping[str, float | int | None]
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_max_s: float | None
    prompt_tokens: int
    completion_tokens: int
    notes: tuple[str, ...] = field(default=())


def summarize_records(records: Sequence[AnswerRecord], *, judge_id: str) -> RunSummary:
    """Metrics of a run. Each ``None`` means there was nothing to compute it from."""
    answered = [r for r in records if r.status is AskStatus.ANSWERED]
    answerable_answered = [r for r in answered if r.answerable and r.fact_coverage is not None]
    settled = [
        r for r in records if r.status in (AskStatus.ANSWERED, AskStatus.INSUFFICIENT_EVIDENCE)
    ]  # an error or an unverified reply is neither an answer nor a refusal
    matrix = no_answer_matrix(
        [(r.answerable, r.status is AskStatus.INSUFFICIENT_EVIDENCE) for r in settled]
    )
    per_answer = [faithfulness(list(r.claim_verdicts)) for r in answered if r.claim_verdicts]
    citation = [
        citation_score(
            list(r.claim_supported_by_citation), r.cited_documents, r.gold_documents,
            valid_markers=r.markers_valid, total_markers=r.markers_total,
        )
        for r in answered
    ]  # fmt: skip
    recall = [
        c.recall
        for c, r in zip(citation, answered, strict=True)
        if r.answerable and c.recall is not None
    ]
    precision = [c.precision for c in citation if c.precision is not None]
    seconds = [r.seconds for r in records]
    return RunSummary(
        questions=len(records),
        by_status={s.value: sum(r.status is s for r in records) for s in AskStatus},
        answer_correctness=_mean(
            [1.0 if (r.fact_coverage or 0.0) >= 1.0 else 0.0 for r in answerable_answered]
        ),
        completeness=_mean([r.fact_coverage or 0.0 for r in answerable_answered]),
        faithfulness=_mean([f for f in per_answer if f is not None]),
        citation_precision=_mean(precision),
        citation_recall=_mean(recall),
        hallucination_rate=hallucination_rate([list(r.claim_verdicts) for r in answered]),
        no_answer={
            "false_refusal_rate": matrix.false_refusal_rate,
            "false_answer_rate": matrix.false_answer_rate,
            "refusal_precision": matrix.refusal_precision,
            "refusal_recall": matrix.refusal_recall,
            "answerable": matrix.answered_answerable + matrix.refused_answerable,
            "unanswerable": matrix.answered_unanswerable + matrix.refused_unanswerable,
        },
        latency_p50_s=_percentile(seconds, 0.5),
        latency_p95_s=_percentile(seconds, 0.95),
        latency_max_s=max(seconds, default=None),
        prompt_tokens=sum(r.prompt_tokens for r in records),
        completion_tokens=sum(r.completion_tokens for r in records),
        notes=(
            f"claims judged by {judge_id}",
            "labels are unreviewed unless the dataset says otherwise",
            f"{len(records) - len(settled)} error or unverified replies are not in the matrix",
            "faithfulness, hallucination and citation precision depend on the judge",
        ),
    )
