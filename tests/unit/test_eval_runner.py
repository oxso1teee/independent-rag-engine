"""Scoring of one ranking, summaries with intervals, and paired comparison, by hand."""

from __future__ import annotations

import math

import pytest

from rag_engine.evaluation.dataset import GoldRef, Provenance, Question, Review
from rag_engine.evaluation.runner import (
    RankedChunk,
    compare,
    score_question,
    summarize,
)

pytestmark = pytest.mark.unit


def make(
    identifier: str = "q",
    *,
    kind: str = "keyword",
    language: str = "en",
    gold: tuple[GoldRef, ...] = (GoldRef("a.md", "alpha beta"),),
    answerable: bool = True,
) -> Question:
    return Question(
        id=identifier,
        split="dev",
        type=kind,
        language=language,
        question="text",
        answerable=answerable,
        gold=gold,
        reference_answer=None,
        reference_facts=(),
        absence_terms=() if answerable else ("x",),
        provenance=Provenance("human", "test", "2026-01-01"),
        review=Review("accepted", "Ada", "2026-01-02", ""),
    )


def test_a_ranking_is_scored_at_document_and_chunk_level() -> None:
    ranked = [
        RankedChunk("b.md", "nothing relevant"),
        RankedChunk("a.md", "foo ALPHA   beta bar"),  # covers the anchor
        RankedChunk("a.md", "another chunk of the same document"),
    ]
    metrics = score_question(make(), ranked).metrics
    assert (metrics["hit@1"], metrics["hit@3"], metrics["hit@10"]) == (0.0, 1.0, 1.0)
    assert metrics["mrr@10"] == pytest.approx(1 / 2)  # the first relevant document is rank 2
    assert metrics["recall@5"] == 1.0 and metrics["precision@5"] == pytest.approx(1 / 5)
    # chunk gains 0, 2 (covers the anchor), 1 (same document); ideal is one grade-2 chunk
    assert metrics["ndcg@10"] == pytest.approx((2 / math.log2(3) + 1 / 2) / 2)
    assert metrics["evidence@5"] == 1.0 and metrics["all_gold@5"] == 1.0
    assert metrics["diversity@5"] == pytest.approx(2 / 3)  # two documents among three chunks


def test_an_anchor_is_credited_once_even_if_two_chunks_contain_it() -> None:
    ranked = [RankedChunk("a.md", "alpha beta here"), RankedChunk("a.md", "alpha beta again")]
    metrics = score_question(make(), ranked).metrics
    assert metrics["ndcg@10"] == pytest.approx((2 + 1 / math.log2(3)) / 2)  # 2.0 and then 1.0


def test_multi_doc_needs_every_gold_document_for_all_gold() -> None:
    gold = (GoldRef("a.md", "alpha beta"), GoldRef("b.md", "gamma delta"))
    question = make(kind="multi_doc", gold=gold)
    only_a = score_question(question, [RankedChunk("a.md", "alpha beta")]).metrics
    assert only_a["recall@5"] == 0.5 and only_a["all_gold@5"] == 0.0
    assert only_a["evidence@5"] == 0.5 and only_a["hit@1"] == 1.0
    both = score_question(
        question, [RankedChunk("a.md", "alpha beta"), RankedChunk("b.md", "gamma delta")]
    ).metrics
    assert both["recall@5"] == 1.0 and both["all_gold@5"] == 1.0 and both["ndcg@10"] == 1.0


def test_an_empty_result_scores_zero_without_dividing_by_zero() -> None:
    metrics = score_question(make(), []).metrics
    assert set(metrics.values()) == {0.0}


def test_unanswerable_questions_have_no_retrieval_score() -> None:
    with pytest.raises(ValueError, match="answerable"):
        score_question(make(answerable=False, gold=()), [])


def perfect(count: int) -> list[float]:
    return [1.0] * count


def test_summaries_carry_intervals_group_by_type_and_language_and_flag_saturation() -> None:
    scores = [
        score_question(
            make(str(i), kind="keyword" if i % 2 else "paraphrase", language="ru"),
            [RankedChunk("a.md", "alpha beta")],
        )
        for i in range(40)
    ]
    summary = summarize(scores, seed=1, resamples=200)
    assert set(summary) >= {"all", "type:keyword", "type:paraphrase", "language:ru"}
    top = summary["all"]["hit@1"]
    assert (top.n, top.mean, top.low, top.high) == (40, 1.0, 1.0, 1.0)
    assert top.saturated and not top.anecdotal
    small = summarize(scores[:5], seed=1, resamples=200)["all"]["hit@1"]
    assert small.anecdotal
    assert summarize(scores, seed=1, resamples=200) == summary  # deterministic


def test_comparison_is_paired_over_the_same_questions() -> None:
    good = [score_question(make(str(i)), [RankedChunk("a.md", "alpha beta")]) for i in range(20)]
    bad = [score_question(make(str(i)), [RankedChunk("b.md", "nothing")]) for i in range(20)]
    gain = compare(good, bad, "hit@1", seed=3, resamples=300)
    assert gain.mean == 1.0 and gain.excludes_zero
    same = compare(good, good, "mrr@10", seed=3, resamples=300)
    assert same.mean == 0.0 and not same.excludes_zero
    with pytest.raises(ValueError, match="same questions"):
        compare(good, bad[:5], "hit@1", seed=3)
