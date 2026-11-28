"""Generation evaluation of a running pipeline: records, claims, facts and the summary."""

from __future__ import annotations

import asyncio

from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.domain import AskStatus, RetrievalQuery
from rag_engine.evaluation.dataset import GoldRef, Provenance, Question, Review
from rag_engine.evaluation.judge import LexicalJudge
from rag_engine.evaluation.llm_run import (
    AnswerRecord,
    claims_with_evidence,
    evaluate_answers,
    fact_present,
    summarize_records,
)
from rag_engine.testing import Reply
from tests.pipeline_support import ANSWER, QUESTION, query, rig


def make_question(identifier: str, *, answerable: bool, facts: tuple[str, ...] = ()) -> Question:
    return Question(
        id=identifier,
        split="dev",
        type="keyword" if answerable else "no_answer",
        language="en",
        question=QUESTION,
        answerable=answerable,
        gold=(GoldRef("doc.md", "SQLite"),) if answerable else (),
        reference_answer="Chunks are stored in SQLite" if answerable else None,
        reference_facts=facts,
        absence_terms=(),
        provenance=Provenance("an AI language model", "draft", "2026-11-01"),
        review=Review("unreviewed", None, None, ""),
    )


def record(**changes: object) -> AnswerRecord:
    base: dict[str, object] = {
        "question_id": "q",
        "answerable": True,
        "status": AskStatus.ANSWERED,
        "text": "an answer",
        "seconds": 1.0,
        "cited_documents": ("doc.md",),
        "gold_documents": ("doc.md",),
        "claim_verdicts": (True,),
        "claim_supported_by_citation": (True,),
        "fact_coverage": 1.0,
        "exact": True,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "markers_total": 1,
        "markers_valid": 1,
    }
    return AnswerRecord(**{**base, **changes})  # type: ignore[arg-type]


def test_numbers_must_all_appear_and_words_mostly() -> None:
    assert fact_present("Backups run at 02:30 UTC [E1].", "02:30")
    assert not fact_present("Backups run at 03:30 UTC.", "02:30 and 14 days")
    assert fact_present("Chunks are stored in the SQLite database", "chunks stored SQLite database")
    assert not fact_present("The weather is fine", "chunks stored SQLite database")
    assert not fact_present("anything", "a b")  # no word long enough to count


def test_claims_come_with_the_text_of_the_evidence_they_cite() -> None:
    result = rig(Reply(ANSWER)).pipeline
    answer = asyncio.run(result.ask(_query()))
    claims = claims_with_evidence(answer.text, answer.sources)
    assert claims and claims[0][0].startswith("Chunks are stored")
    assert claims[0][1], "the cited evidence text is attached"


def _query() -> RetrievalQuery:
    return query()


def test_a_grounded_answer_is_scored_and_timed() -> None:
    pipeline = rig(Reply(ANSWER)).pipeline
    ticks = iter([0.0, 2.5])
    records = asyncio.run(
        evaluate_answers(
            pipeline,
            [make_question("q1", answerable=True, facts=("SQLite",))],
            LexicalJudge(),
            lambda _q: _query(),
            clock=lambda: next(ticks),
        )
    )
    (one,) = records
    assert one.status is AskStatus.ANSWERED and one.seconds == 2.5
    assert one.fact_coverage == 1.0 and one.claim_verdicts == (True,)


def test_a_refusal_is_recorded_without_claims() -> None:
    pipeline = rig(Reply(REFUSAL_SENTINEL)).pipeline
    (one,) = asyncio.run(
        evaluate_answers(
            pipeline,
            [make_question("q2", answerable=False)],
            LexicalJudge(),
            lambda _q: _query(),
        )
    )
    assert one.status is AskStatus.INSUFFICIENT_EVIDENCE
    assert one.claim_verdicts == () and one.fact_coverage is None


def test_the_summary_reports_each_metric_and_marks_what_it_could_not_compute() -> None:
    records = [
        record(question_id="a"),
        record(question_id="b", claim_verdicts=(True, False),
               claim_supported_by_citation=(True, False)),
        record(question_id="c", answerable=False, status=AskStatus.INSUFFICIENT_EVIDENCE, text="",
               claim_verdicts=(), claim_supported_by_citation=(), fact_coverage=None,
               cited_documents=(), gold_documents=(), exact=None, markers_total=0, markers_valid=0),
        record(question_id="d", answerable=False, claim_verdicts=(False,),
               claim_supported_by_citation=(False,), fact_coverage=None, gold_documents=()),
        record(question_id="e", status=AskStatus.UNVERIFIED, text="", claim_verdicts=(),
               claim_supported_by_citation=(), fact_coverage=None),
    ]  # fmt: skip
    summary = summarize_records(records, judge_id="lexical/1")
    assert summary.questions == 5 and summary.by_status["unverified"] == 1
    assert summary.answer_correctness == 1.0 and summary.completeness == 1.0
    assert summary.faithfulness is not None and 0.0 < summary.faithfulness < 1.0
    assert summary.hallucination_rate is not None and summary.hallucination_rate > 0.0
    assert summary.no_answer["false_answer_rate"] == 0.5
    assert summary.no_answer["false_refusal_rate"] == 0.0
    assert summary.latency_p50_s == 1.0 and summary.latency_max_s == 1.0
    assert summary.prompt_tokens == 50
    assert any("lexical/1" in note for note in summary.notes)
    assert any("1 error or unverified" in note for note in summary.notes)


def test_an_empty_run_has_no_metrics_rather_than_zero() -> None:
    summary = summarize_records([], judge_id="lexical/1")
    assert summary.questions == 0
    assert summary.answer_correctness is None and summary.faithfulness is None
    assert summary.latency_p50_s is None and summary.latency_max_s is None
