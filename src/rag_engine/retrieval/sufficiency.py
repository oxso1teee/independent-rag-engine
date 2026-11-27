"""The sufficiency gate: refuse before generation when the evidence is plainly too weak.

Retrieval scores are not probabilities, so the gate compares signals with thresholds that were
calibrated on questions that have an answer and questions that do not, for one embedding model.
Policy: the best dense cosine (A) or the best reranker logit (B) must reach its threshold, either
one is enough; with neither available the lexical coverage of the top passage (C) decides. The
gate never blocks on what it cannot judge: with no calibration, a calibration made for another
model, or no signal, it lets the question through and says why, and ``doctor`` warns about a stale
calibration. Fusion scores are deliberately not a signal: they are not comparable between queries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_engine.domain import (
    ContextValue,
    ErrorCode,
    Language,
    RagError,
    RetrievalHit,
)
from rag_engine.lexical.analysis import analyze


@dataclass(frozen=True, slots=True)
class Signals:
    dense: float | None
    rerank: float | None
    lexical_coverage: float | None


@dataclass(frozen=True, slots=True)
class Thresholds:
    model_id: str
    dense: float | None = None
    rerank: float | None = None
    lexical: float | None = None
    dataset: str = ""


@dataclass(frozen=True, slots=True)
class GateDecision:
    sufficient: bool
    calibrated: bool
    reasons: tuple[str, ...] = ()


class SufficiencyGate:
    def __init__(self, thresholds: Thresholds | None, active_model_id: str | None) -> None:
        self._thresholds = thresholds
        self._active = active_model_id

    def decide(self, signals: Signals) -> GateDecision:
        limits = self._thresholds
        if limits is None:
            return GateDecision(True, False, ("uncalibrated",))
        if limits.model_id != self._active:
            return GateDecision(True, False, ("calibration_invalid",))
        checks: list[tuple[str, bool]] = []
        if limits.dense is not None and signals.dense is not None:
            checks.append(("dense_below_threshold", signals.dense >= limits.dense))
        if limits.rerank is not None and signals.rerank is not None:
            checks.append(("rerank_below_threshold", signals.rerank >= limits.rerank))
        if not checks and limits.lexical is not None and signals.lexical_coverage is not None:
            checks.append(("lexical_below_threshold", signals.lexical_coverage >= limits.lexical))
        if not checks:
            return GateDecision(True, False, ("no_signal",))
        if any(passed for _, passed in checks):
            return GateDecision(True, True)
        return GateDecision(False, True, tuple(name for name, _ in checks))


def lexical_coverage(query: str, text: str, language: Language) -> float | None:
    """Share of the query's analysed terms that occur in ``text``; ``None`` for an empty query."""
    wanted = set(analyze(query, language).split())
    if not wanted:
        return None
    return len(wanted & set(analyze(text, language).split())) / len(wanted)


def signals_from(
    hits: Sequence[RetrievalHit], query: str, top_text: str | None, language: Language
) -> Signals:
    """The best dense and rerank scores among the hits and the coverage of the top passage."""
    dense = [h.scores.dense for h in hits if h.scores.dense is not None]
    rerank = [h.scores.rerank for h in hits if h.scores.rerank is not None]
    coverage = None if top_text is None else lexical_coverage(query, top_text, language)
    return Signals(max(dense) if dense else None, max(rerank) if rerank else None, coverage)


@dataclass(frozen=True, slots=True)
class Calibration:
    threshold: float
    false_refusal_rate: float
    correct_refusal_rate: float
    positives: int
    negatives: int
    meets_target: bool


def calibrate(
    answerable: Sequence[float],
    unanswerable: Sequence[float],
    *,
    max_false_refusal: float = 0.10,
    min_correct_refusal: float = 0.90,
) -> Calibration:
    """Pick the threshold that refuses the most unanswerable questions within the allowance.

    A question is refused when its score is below the threshold. Among thresholds whose false
    refusal rate stays within ``max_false_refusal`` the one with the highest correct refusal rate
    wins; the smallest such threshold is taken, placed midway to the next lower score.
    """
    if not answerable or not unanswerable:
        raise ValueError("calibration needs both answerable and unanswerable questions")
    scores = sorted({*answerable, *unanswerable})
    candidates = [*scores, scores[-1] + 1.0]

    def rates(threshold: float) -> tuple[float, float]:
        return (
            sum(s < threshold for s in answerable) / len(answerable),
            sum(s < threshold for s in unanswerable) / len(unanswerable),
        )

    best: tuple[float, float, float] | None = None  # (correct refusal, -threshold, threshold)
    for candidate in candidates:
        false_rate, correct_rate = rates(candidate)
        if false_rate <= max_false_refusal:
            key = (correct_rate, -candidate, candidate)
            if best is None or key > best:
                best = key
    assert best is not None  # noqa: S101 - the lowest candidate refuses nothing, so one is feasible
    chosen = best[2]
    lower = [s for s in scores if s < chosen]
    threshold = (lower[-1] + chosen) / 2 if lower else chosen
    false_rate, correct_rate = rates(threshold)
    return Calibration(
        threshold,
        false_rate,
        correct_rate,
        len(answerable),
        len(unanswerable),
        false_rate <= max_false_refusal and correct_rate >= min_correct_refusal,
    )


def thresholds_to_metadata(thresholds: Thresholds) -> dict[str, ContextValue]:
    """The form stored in ``index_state.calibration``."""
    return {
        "sufficiency": {
            "model_id": thresholds.model_id,
            "dense": thresholds.dense,
            "rerank": thresholds.rerank,
            "lexical": thresholds.lexical,
            "dataset": thresholds.dataset,
        }
    }


def thresholds_from_metadata(calibration: Mapping[str, ContextValue]) -> Thresholds | None:
    stored = calibration.get("sufficiency")
    if stored is None:
        return None
    if not isinstance(stored, Mapping) or not isinstance(stored.get("model_id"), str):
        raise RagError(ErrorCode.E_INTEGRITY, "the stored sufficiency calibration is malformed")

    def number(key: str) -> float | None:
        value = stored.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RagError(ErrorCode.E_INTEGRITY, "the stored sufficiency calibration is malformed")
        return float(value)

    dataset = stored.get("dataset", "")
    return Thresholds(
        model_id=str(stored["model_id"]),
        dense=number("dense"),
        rerank=number("rerank"),
        lexical=number("lexical"),
        dataset=dataset if isinstance(dataset, str) else "",
    )
