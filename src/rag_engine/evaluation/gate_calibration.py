"""Calibrating the sufficiency gate on an installation's own index (ADR 0021).

``rag calibrate`` takes questions that the index should answer and questions it should not, runs
them through the same retrieval a real query uses, and fits one threshold for the signal that will
decide at query time. What comes back is an *estimate*: cross-validated rates, and the rates on a
second file that took no part in the fit when one is given. The threshold is stored together with
what it is valid for (embedding model, retrieval setup, decision rule), so a gate never applies a
calibration to scores it was not made for. Nothing here talks to a language model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rag_engine.domain import (
    ContextValue,
    ErrorCode,
    IdFactory,
    RagError,
    RerankMode,
    RetrievalMode,
    RetrievalQuery,
    thaw_context,
)
from rag_engine.evaluation.dataset import accepted_questions, load_dataset
from rag_engine.ports import Store
from rag_engine.retrieval.service import RetrievalService
from rag_engine.retrieval.sufficiency import (
    GATE_ALGORITHM,
    TARGET_CORRECT_REFUSAL,
    TARGET_FALSE_REFUSAL,
    RetrievalShape,
    Signals,
    Thresholds,
    binding_fingerprint,
    calibrate,
    signals_from,
    thresholds_from_metadata,
    thresholds_to_metadata,
)

MIN_PER_CLASS = 20
ANECDOTAL_BELOW = 30
MAX_QUESTIONS = 5000
MAX_QUESTION_CHARS = 2000
NEAR_DUPLICATE = 0.8
_SIGNAL_SHARE = 0.9
_WORD = re.compile(r"\w+", re.UNICODE)


def _invalid(message: str) -> RagError:
    return RagError(ErrorCode.E_QUERY_INVALID, message)


# ------------------------------------------------------------------------------ question sets


@dataclass(frozen=True, slots=True)
class CalQuestion:
    text: str
    answerable: bool


@dataclass(frozen=True, slots=True)
class QuestionSet:
    questions: tuple[CalQuestion, ...]
    identifier: str
    digest: str
    exploratory: bool

    @property
    def answerable(self) -> int:
        return sum(q.answerable for q in self.questions)

    @property
    def unanswerable(self) -> int:
        return len(self.questions) - self.answerable


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.casefold()) if len(w) > 2)


def _checked(rows: Sequence[CalQuestion], where: str) -> tuple[CalQuestion, ...]:
    """Drops repeats, refuses an empty text, a text that is too long and a contradiction."""
    if len(rows) > MAX_QUESTIONS:
        raise _invalid(f"{where}: at most {MAX_QUESTIONS} questions")
    seen: dict[str, bool] = {}
    kept: list[CalQuestion] = []
    for row in rows:
        key = _norm(row.text)
        if not key:
            raise _invalid(f"{where}: a question is empty")
        if len(row.text) > MAX_QUESTION_CHARS:
            raise _invalid(f"{where}: a question is longer than {MAX_QUESTION_CHARS} characters")
        if key in seen:
            if seen[key] != row.answerable:
                raise _invalid(f"{where}: one question is labelled both answerable and not")
            continue
        seen[key] = row.answerable
        kept.append(row)
    return tuple(kept)


def load_question_set(path: Path, *, include_unreviewed: bool = False) -> QuestionSet:
    """Questions from an evaluation dataset (its dev split) or from JSON Lines.

    A JSON Lines row is ``{"question": "...", "answerable": true}``. A dataset question counts when
    a person accepted it, unless ``include_unreviewed`` is set, which marks the result exploratory.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        raise _invalid(f"{path.name} cannot be read") from None
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{") and '"questions"' in text:
        dataset = load_dataset(path)
        pool = (
            [q for q in dataset.questions if q.review.status != "rejected"]
            if include_unreviewed
            else accepted_questions(dataset)
        )
        rows = [CalQuestion(q.question, q.answerable) for q in pool if q.split == "dev"]
        exploratory = include_unreviewed and any(
            q.review.status not in ("accepted", "edited") for q in pool if q.split == "dev"
        )
        return QuestionSet(_checked(rows, path.name), dataset.identifier, digest, exploratory)
    parsed: list[CalQuestion] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise _invalid(f"{path.name} line {number} is not JSON") from None
        if (
            not isinstance(row, Mapping)
            or set(row) != {"question", "answerable"}
            or not isinstance(row["question"], str)
            or not isinstance(row["answerable"], bool)
        ):
            raise _invalid(
                f'{path.name} line {number}: expected {{"question": text, "answerable": bool}}'
            )
        parsed.append(CalQuestion(row["question"], row["answerable"]))
    return QuestionSet(_checked(parsed, path.name), f"sha256:{digest[:16]}", digest, False)


def overlap(fit: QuestionSet, holdout: QuestionSet) -> int:
    """How many holdout questions are in the calibration set, exactly or nearly."""
    if fit.digest == holdout.digest:
        return len(holdout.questions)
    known = {_norm(q.text) for q in fit.questions}
    shingles = [_tokens(q.text) for q in fit.questions]
    count = 0
    for question in holdout.questions:
        if _norm(question.text) in known:
            count += 1
            continue
        words = _tokens(question.text)
        if len(words) >= 4 and any(
            len(words & other) / len(words | other) >= NEAR_DUPLICATE
            for other in shingles
            if len(other) >= 4
        ):
            count += 1
    return count


# ---------------------------------------------------------------------------------- observing


@dataclass(frozen=True, slots=True)
class Observation:
    answerable: bool
    signals: Signals | None
    """``None``: retrieval found nothing, so a real query is refused without a gate."""


async def observe(
    retrieval: RetrievalService,
    ids: IdFactory,
    questions: Sequence[CalQuestion],
    *,
    mode: RetrievalMode,
    top_k: int,
    candidate_k: int,
) -> list[Observation]:
    """The signals each question produces in a real query (the defaults of ``rag ask``)."""
    found: list[Observation] = []
    for question in questions:
        query = RetrievalQuery(
            request_id=ids.request_id(),
            text=question.text,
            mode=mode,
            top_k=top_k,
            candidate_k=max(candidate_k, top_k),
            rerank=RerankMode.AUTO,
        )
        outcome = await retrieval.search(query)
        hits = outcome.result.hits
        if not hits:
            found.append(Observation(question.answerable, None))
            continue
        shape = RetrievalShape(
            query.mode.value,
            query.top_k,
            query.candidate_k,
            any(hit.scores.rerank is not None for hit in hits),
        )
        found.append(
            Observation(
                question.answerable,
                signals_from(hits, question.text, None, outcome.diagnostics.language, shape),
            )
        )
    return found


# ------------------------------------------------------------------------------------ fitting


@dataclass(frozen=True, slots=True)
class Rates:
    answerable: int
    unanswerable: int
    false_refusals: int
    correct_refusals: int

    @property
    def false_refusal(self) -> float:
        return self.false_refusals / self.answerable if self.answerable else 0.0

    @property
    def correct_refusal(self) -> float:
        return self.correct_refusals / self.unanswerable if self.unanswerable else 0.0

    def meets_target(self) -> bool:
        return (
            self.correct_refusal >= TARGET_CORRECT_REFUSAL
            and self.false_refusal <= TARGET_FALSE_REFUSAL
        )

    def as_context(self) -> dict[str, ContextValue]:
        return {
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "false_refusal": round(self.false_refusal, 4),
            "correct_refusal": round(self.correct_refusal, 4),
        }


def _score(obs: Observation, signal: str) -> float | None:
    if obs.signals is None:
        return None
    return obs.signals.rerank if signal == "rerank" else obs.signals.dense


def choose_signal(observations: Sequence[Observation]) -> str:
    """The signal that decides at query time: the reranker's if it ran, else the dense cosine."""
    seen = [o.signals for o in observations if o.signals is not None]
    if not seen:
        raise _invalid("retrieval found nothing for any question; is the index empty?")
    for name in ("rerank", "dense"):
        have = sum(1 for s in seen if (s.rerank if name == "rerank" else s.dense) is not None)
        if have >= _SIGNAL_SHARE * len(seen):
            return name
    raise _invalid(
        "neither the reranker nor the dense score is available for most questions; the gate "
        "cannot be calibrated in lexical-only mode"
    )


def _rates(observations: Sequence[Observation], signal: str, threshold: float | None) -> Rates:
    """Refusals with ``threshold`` as a real query would see them.

    No hits: always refused. No score for the signal: never refused (the gate lets it through).
    """
    fr = cr = a = u = 0
    for obs in observations:
        score = _score(obs, signal)
        if obs.signals is None:
            refused = True
        elif score is None or threshold is None:
            refused = False
        else:
            refused = score < threshold
        if obs.answerable:
            a += 1
            fr += refused
        else:
            u += 1
            cr += refused
    return Rates(a, u, fr, cr)


def _fit(scored: Sequence[Observation], signal: str, max_false: float) -> float:
    yes = [s for o in scored if o.answerable and (s := _score(o, signal)) is not None]
    no = [s for o in scored if not o.answerable and (s := _score(o, signal)) is not None]
    return calibrate(yes, no, max_false_refusal=max_false).threshold


def cross_validate(
    observations: Sequence[Observation], signal: str, *, folds: int, max_false: float, seed: int
) -> Rates:
    """Fit on all folds but one and judge the one left out, for every fold in turn."""
    scored = [o for o in observations if _score(o, signal) is not None]
    fixed = [o for o in observations if _score(o, signal) is None]  # decided without a threshold
    rng = random.Random(seed)  # noqa: S311 - a reproducible split, not a secret
    assigned: dict[int, int] = {}
    for label in (True, False):
        idx = [i for i, o in enumerate(scored) if o.answerable is label]
        rng.shuffle(idx)
        for position, i in enumerate(idx):
            assigned[i] = position % folds
    total = _rates(fixed, signal, None)
    for fold in range(folds):
        train = [o for i, o in enumerate(scored) if assigned[i] != fold]
        test = [o for i, o in enumerate(scored) if assigned[i] == fold]
        part = _rates(test, signal, _fit(train, signal, max_false))
        total = Rates(
            total.answerable + part.answerable,
            total.unanswerable + part.unanswerable,
            total.false_refusals + part.false_refusals,
            total.correct_refusals + part.correct_refusals,
        )
    return total


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    thresholds: Thresholds
    signal: str
    estimate: Rates
    in_sample: Rates
    holdout: Rates | None
    anecdotal: bool
    exploratory: bool
    warnings: tuple[str, ...]

    @property
    def meets_target(self) -> bool:
        return not self.thresholds.below_target


def _modal_shape(observations: Sequence[Observation], signal: str) -> RetrievalShape:
    """The retrieval shape most questions had among those that produced the chosen signal."""
    counts: dict[RetrievalShape, int] = {}
    for obs in observations:
        shape = None if obs.signals is None else obs.signals.shape
        if shape is not None and _score(obs, signal) is not None:
            counts[shape] = counts.get(shape, 0) + 1
    return max(counts, key=lambda shape: counts[shape])


def fit_calibration(
    observations: Sequence[Observation],
    *,
    model_id: str,
    environment: str,
    dataset: str,
    exploratory: bool,
    created_at: str,
    engine_version: str,
    index_generation: int,
    documents: int,
    max_false: float = TARGET_FALSE_REFUSAL,
    folds: int = 5,
    holdout: Sequence[Observation] | None = None,
    holdout_dataset: str | None = None,
) -> CalibrationResult:
    """The threshold, its cross-validated estimate and the record to store."""
    yes = sum(o.answerable for o in observations)
    no = len(observations) - yes
    if min(yes, no) < MIN_PER_CLASS:
        raise _invalid(
            f"calibration needs at least {MIN_PER_CLASS} answerable and {MIN_PER_CLASS} "
            f"unanswerable questions; got {yes} and {no}"
        )
    if not 0.0 <= max_false < 1.0:
        raise _invalid("--max-false-refusal must be in [0, 1)")
    if not 2 <= folds <= min(yes, no):
        raise _invalid(f"--folds must be between 2 and {min(yes, no)}")
    signal = choose_signal(observations)
    scored = [o for o in observations if _score(o, signal) is not None]
    if min(sum(o.answerable for o in scored), sum(not o.answerable for o in scored)) < folds:
        raise _invalid("too few questions produced a score to calibrate")
    shape = _modal_shape(observations, signal)
    threshold = _fit(scored, signal, max_false)
    in_sample = _rates(observations, signal, threshold)
    seed = int(hashlib.sha256(dataset.encode()).hexdigest()[:8], 16)
    estimate = cross_validate(observations, signal, folds=folds, max_false=max_false, seed=seed)
    held = None if holdout is None else _rates(holdout, signal, threshold)
    anecdotal = min(yes, no) < ANECDOTAL_BELOW
    below = not estimate.meets_target()
    notes: list[str] = []
    if below:
        notes.append(
            f"estimated correct refusals {estimate.correct_refusal:.1%} (target "
            f"{TARGET_CORRECT_REFUSAL:.0%}), false refusals {estimate.false_refusal:.1%} (target "
            f"at most {TARGET_FALSE_REFUSAL:.0%}): the gate is written but below its targets"
        )
    if held is not None and not held.meets_target():
        notes.append(
            f"on the separate holdout: correct refusals {held.correct_refusal:.1%}, false "
            f"refusals {held.false_refusal:.1%}: below the targets"
        )
    if anecdotal:
        notes.append(
            f"fewer than {ANECDOTAL_BELOW} questions in a class: the estimate is anecdotal"
        )
    if exploratory:
        notes.append("some questions were not reviewed by a person: the result is exploratory")
    details: dict[str, ContextValue] = {
        "signal": signal,
        "answerable": yes,
        "unanswerable": no,
        "no_hits": sum(o.signals is None for o in observations),
        "folds": folds,
        "max_false_refusal": max_false,
        "estimate": estimate.as_context(),
        "in_sample": in_sample.as_context(),
        "holdout": None if held is None else held.as_context(),
        "holdout_dataset": holdout_dataset,
        "anecdotal": anecdotal,
        "exploratory": exploratory,
        "shape": {
            "mode": shape.mode,
            "top_k": shape.top_k,
            "candidate_k": shape.candidate_k,
            "reranked": shape.reranked,
        },
        "index_generation": index_generation,
        "documents": documents,
        "engine_version": engine_version,
        "created_at": created_at,
    }
    thresholds = Thresholds(
        model_id=model_id,
        dense=threshold if signal == "dense" else None,
        rerank=threshold if signal == "rerank" else None,
        dataset=dataset,
        algorithm=GATE_ALGORITHM,
        binding=binding_fingerprint(environment, shape),
        below_target=below,
        details=details,
    )
    return CalibrationResult(
        thresholds, signal, estimate, in_sample, held, anecdotal, exploratory, tuple(notes)
    )


# ----------------------------------------------------------------------------------- storing


def save_calibration(
    store: Store, result: CalibrationResult, *, model_id: str, index_generation: int
) -> None:
    """Replaces the stored record in one transaction, only if the index is still the one measured.

    The writer lock is taken first, so a concurrent ingest either finished before the check (and
    the generation differs: nothing is written) or waits for this transaction. A failure anywhere
    rolls back; the previous record, if any, stays whole.
    """
    with store.unit_of_work().begin() as writer:
        metadata = writer.get_index_metadata()
        active = None if metadata is None else metadata.active_embedding_model_id
        if (
            metadata is None
            or active is None
            or str(active) != model_id
            or metadata.index_generation != index_generation
        ):
            raise RagError(
                ErrorCode.E_STORE_LOCKED,
                "the index changed while it was being calibrated; run the command again",
            )
        merged = {**metadata.calibration, **thresholds_to_metadata(result.thresholds)}
        writer.update_index_metadata(replace(metadata, calibration=merged))


def clear_calibration(store: Store) -> bool:
    """Removes the stored record; ``False`` when there was none."""
    with store.unit_of_work().begin() as writer:
        metadata = writer.get_index_metadata()
        if metadata is None or "sufficiency" not in metadata.calibration:
            return False
        rest = {k: v for k, v in metadata.calibration.items() if k != "sufficiency"}
        writer.update_index_metadata(replace(metadata, calibration=rest))
        return True


def describe_calibration(
    stored: Mapping[str, ContextValue], *, active_model: str | None, documents: int
) -> dict[str, object]:
    """What ``rag calibrate show`` says: the record and whether it still applies."""
    thresholds = thresholds_from_metadata(stored)
    if thresholds is None:
        return {"state": "uncalibrated", "problems": ["no calibration is stored"]}
    problems = calibration_problems(thresholds, active_model=active_model, documents=documents)
    details = thresholds.details or {}
    return {
        "state": "stale" if problems else "calibrated",
        "problems": problems,
        "model_id": thresholds.model_id,
        "dataset": thresholds.dataset,
        "algorithm": thresholds.algorithm,
        "threshold": thresholds.rerank if thresholds.rerank is not None else thresholds.dense,
        "below_target": thresholds.below_target,
        "details": thaw_context(details),
    }


def calibration_problems(
    thresholds: Thresholds, *, active_model: str | None, documents: int
) -> list[str]:
    """Why a stored calibration should not be trusted, in words (empty when it can be)."""
    problems: list[str] = []
    if active_model is not None and thresholds.model_id != active_model:
        problems.append("made for another embedding model")
    if thresholds.algorithm != GATE_ALGORITHM:
        problems.append("made for an older decision rule")
    made = (thresholds.details or {}).get("documents")
    if (
        isinstance(made, int)
        and not isinstance(made, bool)
        and made > 0
        and abs(documents - made) / made > 0.5
    ):
        problems.append(
            f"the index has changed a lot since ({made} documents then, {documents} now)"
        )
    if thresholds.below_target:
        problems.append("its estimate was below the targets")
    return problems
