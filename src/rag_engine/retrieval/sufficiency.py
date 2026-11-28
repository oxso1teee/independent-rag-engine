"""The sufficiency gate: refuse before generation when the evidence is plainly too weak.

Retrieval scores are not probabilities, so the gate compares signals with thresholds that were
calibrated on questions that have an answer and questions that do not, for one embedding model.
Policy: the best reranker logit decides when the reranker ran and has a threshold: on an
independent held-out set it refused 72.5 % of the unanswerable questions against 32.5 % for the
best dense cosine, at similar false refusals. Without it the best dense cosine decides, and
without either the lexical coverage of the top passage. The gate never blocks on what it cannot
judge: with no calibration, a calibration made for another model, or no signal, it lets the
question through and says why, and ``doctor`` warns about a stale calibration. Fusion scores are
deliberately not a signal: they are not comparable between queries.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_engine.domain import (
    ContextValue,
    ErrorCode,
    Language,
    RagError,
    RetrievalHit,
)
from rag_engine.domain.models import SUFFICIENCY_ALGORITHM
from rag_engine.lexical.analysis import analyze

GATE_ALGORITHM = SUFFICIENCY_ALGORITHM
"""Version of the decision rule. 1 was "dense or rerank"; 2 is the first signal that has a score
and a threshold (ADR 0020). A calibration made for another version is not applied."""

TARGET_CORRECT_REFUSAL = 0.90
TARGET_FALSE_REFUSAL = 0.10


@dataclass(frozen=True, slots=True)
class RetrievalShape:
    """How one query was retrieved: what decides which scores the gate is looking at."""

    mode: str
    top_k: int
    candidate_k: int
    reranked: bool


@dataclass(frozen=True, slots=True)
class Signals:
    dense: float | None
    rerank: float | None
    lexical_coverage: float | None
    shape: RetrievalShape | None = None


def gate_environment(
    chunker: str, analyzer: str, rerank_revision: str | None, rerank_pool: int | None
) -> str:
    """The parts of the setup, besides the embedding model, that change the scores."""
    return json.dumps([chunker, analyzer, rerank_revision, rerank_pool])


def binding_fingerprint(environment: str, shape: RetrievalShape) -> str:
    """A short digest of the setup and the retrieval a calibration was made for."""
    payload = json.dumps(
        [GATE_ALGORITHM, environment, shape.mode, shape.top_k, shape.candidate_k, shape.reranked]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Thresholds:
    model_id: str
    dense: float | None = None
    rerank: float | None = None
    lexical: float | None = None
    dataset: str = ""
    algorithm: int = GATE_ALGORITHM
    binding: str = ""
    """Digest of the setup and retrieval it was made for; empty means "not checked"."""
    below_target: bool = False
    """The cross-validated estimate at calibration time missed the targets."""
    details: Mapping[str, ContextValue] | None = None
    """Signal, counts, estimates and the date; for display and ``doctor``, never for deciding."""


@dataclass(frozen=True, slots=True)
class GateDecision:
    sufficient: bool
    calibrated: bool
    reasons: tuple[str, ...] = ()
    below_target: bool = False


class SufficiencyGate:
    def __init__(
        self,
        thresholds: Thresholds | None,
        active_model_id: str | None,
        environment: str = "",
    ) -> None:
        self._thresholds = thresholds
        self._active = active_model_id
        self._environment = environment

    def decide(self, signals: Signals) -> GateDecision:
        limits = self._thresholds
        if limits is None:
            return GateDecision(True, False, ("uncalibrated",))
        if limits.model_id != self._active:
            return GateDecision(True, False, ("calibration_invalid",))
        if limits.algorithm != GATE_ALGORITHM:
            return GateDecision(True, False, ("calibration_outdated",))
        if (
            limits.binding
            and signals.shape is not None
            and binding_fingerprint(self._environment, signals.shape) != limits.binding
        ):
            return GateDecision(True, False, ("calibration_stale",))
        # the first signal that has both a score and a threshold decides; a weaker signal never
        # overrules a stronger one (a lenient dense score used to rescue a poor rerank logit)
        if limits.rerank is not None and signals.rerank is not None:
            check = ("rerank_below_threshold", signals.rerank >= limits.rerank)
        elif limits.dense is not None and signals.dense is not None:
            check = ("dense_below_threshold", signals.dense >= limits.dense)
        elif limits.lexical is not None and signals.lexical_coverage is not None:
            check = ("lexical_below_threshold", signals.lexical_coverage >= limits.lexical)
        else:
            return GateDecision(True, False, ("no_signal",))
        name, passed = check
        if passed:
            return GateDecision(True, True, below_target=limits.below_target)
        return GateDecision(False, True, (name,), below_target=limits.below_target)


_GATE_WARNINGS = {
    "uncalibrated": (
        "the sufficiency gate is not calibrated for this index, so weak evidence is not refused "
        "before generation; run `rag calibrate`"
    ),
    "calibration_invalid": (
        "the gate calibration was made for another embedding model and is ignored; "
        "run `rag calibrate`"
    ),
    "calibration_outdated": (
        "the gate calibration was made for an older decision rule and is ignored; "
        "run `rag calibrate`"
    ),
    "calibration_stale": (
        "the gate calibration was made for another retrieval setup (mode, depth, reranker or "
        "chunking) and is ignored for this query; run `rag calibrate` or ask with the same settings"
    ),
    "no_signal": (
        "the score the gate was calibrated on is not available for this query (for example the "
        "reranker did not run), so the gate let it through"
    ),
}


def gate_warnings(decision: GateDecision | None) -> tuple[str, ...]:
    """What a person should be told about a gate that was open, stale or below its targets."""
    if decision is None:
        return ()
    out = [_GATE_WARNINGS[r] for r in decision.reasons if r in _GATE_WARNINGS]
    if decision.below_target:
        out.append(
            "the gate calibration estimated fewer correct refusals or more false refusals than "
            "the targets (see `rag calibrate show`)"
        )
    return tuple(out)


def lexical_coverage(query: str, text: str, language: Language) -> float | None:
    """Share of the query's analysed terms that occur in ``text``; ``None`` for an empty query."""
    wanted = set(analyze(query, language).split())
    if not wanted:
        return None
    return len(wanted & set(analyze(text, language).split())) / len(wanted)


def signals_from(
    hits: Sequence[RetrievalHit],
    query: str,
    top_text: str | None,
    language: Language,
    shape: RetrievalShape | None = None,
) -> Signals:
    """The best dense and rerank scores among the hits and the coverage of the top passage."""
    dense = [h.scores.dense for h in hits if h.scores.dense is not None]
    rerank = [h.scores.rerank for h in hits if h.scores.rerank is not None]
    coverage = None if top_text is None else lexical_coverage(query, top_text, language)
    return Signals(max(dense) if dense else None, max(rerank) if rerank else None, coverage, shape)


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
    record: dict[str, ContextValue] = {
        "model_id": thresholds.model_id,
        "dense": thresholds.dense,
        "rerank": thresholds.rerank,
        "lexical": thresholds.lexical,
        "dataset": thresholds.dataset,
        "algorithm": thresholds.algorithm,
        "binding": thresholds.binding,
        "below_target": thresholds.below_target,
    }
    if thresholds.details is not None:
        record["details"] = thresholds.details
    return {"sufficiency": record}


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
    algorithm = stored.get("algorithm", 0)  # a record without one predates the rule in use
    binding = stored.get("binding", "")
    details = stored.get("details")
    if (
        isinstance(algorithm, bool)
        or not isinstance(algorithm, int)
        or not isinstance(binding, str)
    ):
        raise RagError(ErrorCode.E_INTEGRITY, "the stored sufficiency calibration is malformed")
    return Thresholds(
        model_id=str(stored["model_id"]),
        dense=number("dense"),
        rerank=number("rerank"),
        lexical=number("lexical"),
        dataset=dataset if isinstance(dataset, str) else "",
        algorithm=algorithm,
        binding=binding,
        below_target=stored.get("below_target") is True,
        details=details if isinstance(details, Mapping) else None,
    )
