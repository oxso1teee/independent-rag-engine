"""Scoring retrieval runs: per-question metrics, summaries with intervals, paired comparison.

A run is one ranked list of chunks per answerable question. Relevance is judged at two levels:
documents (is the gold document in the list) and chunks, where a chunk that contains a gold
anchor earns grade 2 once per anchor and any other chunk of a gold document earns grade 1.
Unanswerable questions have no retrieval score; they belong to the no-answer evaluation.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_engine.evaluation.bootstrap import Interval, bootstrap_mean, paired_bootstrap
from rag_engine.evaluation.dataset import Question
from rag_engine.evaluation.metrics import (
    hit_at_k,
    ndcg_from_gains,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

METRICS = (
    "hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "precision@5",
    "mrr@10", "ndcg@10", "evidence@5", "all_gold@5", "diversity@5",
)  # fmt: skip
SATURATION = 0.98
ANECDOTAL = 30
EVIDENCE_DEPTH = 5


@dataclass(frozen=True, slots=True)
class RankedChunk:
    document: str
    text: str


@dataclass(frozen=True, slots=True)
class QuestionScore:
    question_id: str
    type: str
    language: str
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    n: int
    mean: float
    low: float
    high: float

    @property
    def saturated(self) -> bool:
        """Near the ceiling with an interval that touches it: gains cannot be measured."""
        return self.mean >= SATURATION and self.high >= 1.0

    @property
    def anecdotal(self) -> bool:
        return self.n < ANECDOTAL


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def score_question(question: Question, ranked: Sequence[RankedChunk]) -> QuestionScore:
    if not question.answerable or not question.gold:
        raise ValueError("only an answerable question has a retrieval score")
    documents = list(dict.fromkeys(chunk.document for chunk in ranked))
    wanted = {ref.doc for ref in question.gold}
    relevant = dict.fromkeys(wanted, 1)

    uncovered = [(ref.doc, _fold(ref.anchor)) for ref in question.gold]
    gains: list[float] = []
    for chunk in ranked[:10]:
        text = _fold(chunk.text)
        hit = next(
            (item for item in uncovered if item[0] == chunk.document and item[1] in text), None
        )
        if hit is not None:
            uncovered.remove(hit)
            gains.append(2.0)
        else:
            gains.append(1.0 if chunk.document in wanted else 0.0)

    top = ranked[:EVIDENCE_DEPTH]
    covered = len(question.gold) - sum(
        1
        for ref in question.gold
        if not any(c.document == ref.doc and _fold(ref.anchor) in _fold(c.text) for c in top)
    )
    evidence = covered / len(question.gold)
    metrics = {
        "hit@1": hit_at_k(documents, relevant, 1),
        "hit@3": hit_at_k(documents, relevant, 3),
        "hit@5": hit_at_k(documents, relevant, 5),
        "hit@10": hit_at_k(documents, relevant, 10),
        "recall@5": recall_at_k(documents, relevant, 5),
        "precision@5": precision_at_k(documents, relevant, 5),
        "mrr@10": reciprocal_rank(documents, relevant, 10),
        "ndcg@10": ndcg_from_gains(gains, [2.0] * len(question.gold), 10),
        "evidence@5": evidence,
        "all_gold@5": 1.0 if evidence == 1.0 else 0.0,
        "diversity@5": len({c.document for c in top}) / len(top) if top else 0.0,
    }
    return QuestionScore(question.id, question.type, question.language, metrics)


def summarize(
    scores: Sequence[QuestionScore], *, seed: int, resamples: int = 2000
) -> dict[str, dict[str, MetricSummary]]:
    """Mean and 95 % interval of every metric for all questions and for each type and language."""
    groups: dict[str, list[QuestionScore]] = {"all": list(scores)}
    for item in scores:
        groups.setdefault(f"type:{item.type}", []).append(item)
        groups.setdefault(f"language:{item.language}", []).append(item)
    summary: dict[str, dict[str, MetricSummary]] = {}
    for name, members in sorted(groups.items()):
        summary[name] = {}
        for metric in METRICS:
            interval = bootstrap_mean(
                [item.metrics[metric] for item in members], seed=seed, resamples=resamples
            )
            summary[name][metric] = MetricSummary(
                len(members), interval.mean, interval.low, interval.high
            )
    return summary


def compare(
    first: Sequence[QuestionScore],
    second: Sequence[QuestionScore],
    metric: str,
    *,
    seed: int,
    resamples: int = 2000,
) -> Interval:
    """Paired interval of ``first - second`` over the same questions."""
    one = {item.question_id: item.metrics[metric] for item in first}
    two = {item.question_id: item.metrics[metric] for item in second}
    if one.keys() != two.keys():
        raise ValueError("a paired comparison needs scores for the same questions")
    ids = sorted(one)
    return paired_bootstrap(
        [one[key] for key in ids], [two[key] for key in ids], seed=seed, resamples=resamples
    )
