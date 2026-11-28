"""Fitting the gate: question files, independence of the holdout, estimates and records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.evaluation.gate_calibration import (
    CalibrationResult,
    CalQuestion,
    Observation,
    QuestionSet,
    calibration_problems,
    choose_signal,
    cross_validate,
    fit_calibration,
    load_question_set,
    overlap,
)
from rag_engine.retrieval.sufficiency import (
    GATE_ALGORITHM,
    GateDecision,
    RetrievalShape,
    Signals,
    SufficiencyGate,
    Thresholds,
    binding_fingerprint,
    gate_warnings,
    thresholds_from_metadata,
    thresholds_to_metadata,
)

SHAPE = RetrievalShape("dense", 8, 40, False)
MODEL = "fake/m@1"


def obs(answerable: bool, dense: float | None, rerank: float | None = None) -> Observation:
    return Observation(answerable, Signals(dense, rerank, None, SHAPE))


def separable(per_class: int = 30) -> list[Observation]:
    yes = [obs(True, 0.8 + i / 1000) for i in range(per_class)]
    no = [obs(False, 0.3 + i / 1000) for i in range(per_class)]
    return yes + no


def fit(observations: list[Observation], **extra: Any) -> CalibrationResult:
    params: dict[str, object] = {
        "model_id": MODEL, "environment": "env", "dataset": "d@v1", "exploratory": False,
        "created_at": "2026-11-28T00:00:00+00:00", "engine_version": "x", "index_generation": 3,
        "documents": 10,
    }  # fmt: skip
    return fit_calibration(observations, **{**params, **extra})


def write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ question files


def test_json_lines_are_read_and_repeats_dropped(tmp_path: Path) -> None:
    rows = [
        {"question": "What is X?", "answerable": True},
        {"question": "what  is x?", "answerable": True},
        {"question": "Where is Y?", "answerable": False},
    ]
    loaded = load_question_set(write(tmp_path / "q.jsonl", rows))
    assert [q.answerable for q in loaded.questions] == [True, False]
    assert loaded.answerable == 1 and loaded.unanswerable == 1 and not loaded.exploratory


@pytest.mark.parametrize(
    "rows",
    [
        [{"question": "a", "answerable": True, "extra": 1}],
        [{"question": "", "answerable": True}],
        [{"question": "a", "answerable": "yes"}],
        [{"question": "a", "answerable": True}, {"question": "A", "answerable": False}],
        [{"question": "x" * 2001, "answerable": True}],
    ],
)
def test_malformed_or_contradictory_rows_are_refused(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    with pytest.raises(RagError) as info:
        load_question_set(write(tmp_path / "q.jsonl", rows))
    assert info.value.code is ErrorCode.E_QUERY_INVALID


def test_a_line_that_is_not_json_and_a_missing_file_are_refused(tmp_path: Path) -> None:
    broken = tmp_path / "q.jsonl"
    broken.write_text("not json\n")
    with pytest.raises(RagError):
        load_question_set(broken)
    with pytest.raises(RagError):
        load_question_set(tmp_path / "absent.jsonl")


def test_too_many_questions_are_refused(tmp_path: Path) -> None:
    rows = [{"question": f"q{i}", "answerable": True} for i in range(5001)]
    with pytest.raises(RagError, match="at most"):
        load_question_set(write(tmp_path / "q.jsonl", rows))


# ------------------------------------------------------------------------------- the holdout


def qset(*texts: str, digest: str = "other") -> QuestionSet:
    return QuestionSet(tuple(CalQuestion(t, True) for t in texts), "id", digest, False)


def test_overlap_counts_exact_near_and_identical_files() -> None:
    fit = qset("Which protocol does the alpha subsystem use to synchronise records?", digest="d")
    assert (
        overlap(fit, qset("which  protocol does the ALPHA subsystem use to synchronise records?"))
        == 1
    )
    near = qset("Which protocol does the alpha subsystem use to synchronise all records?")
    assert overlap(fit, near) == 1
    assert overlap(fit, qset("How tall is the northern mountain range today?")) == 0
    assert overlap(fit, qset("anything", digest="d")) == 1  # the very same file


# ---------------------------------------------------------------------------------- the fit


def test_a_separable_set_meets_the_targets_and_binds_the_record() -> None:
    result = fit(separable())
    thresholds = result.thresholds
    assert result.signal == "dense" and thresholds.dense is not None
    assert 0.3 < thresholds.dense < 0.9 and thresholds.rerank is None
    assert result.estimate.correct_refusal == 1.0 and result.estimate.false_refusal == 0.0
    assert result.meets_target and not thresholds.below_target and not result.anecdotal
    assert thresholds.algorithm == GATE_ALGORITHM
    assert thresholds.binding == binding_fingerprint("env", SHAPE)
    assert thresholds.details is not None and thresholds.details["documents"] == 10


def test_the_reranker_signal_is_chosen_when_it_ran_and_is_what_gets_stored() -> None:
    rows = [obs(True, 0.5, 4.0 + i / 100) for i in range(25)]
    rows += [obs(False, 0.5, -3.0 - i / 100) for i in range(25)]
    assert choose_signal(rows) == "rerank"
    result = fit(rows)
    assert result.thresholds.rerank is not None and result.thresholds.dense is None


def test_with_no_usable_signal_or_no_hits_nothing_is_calibrated() -> None:
    with pytest.raises(RagError, match="found nothing"):
        choose_signal([Observation(True, None)] * 3)
    lexical = [Observation(True, Signals(None, None, 0.9, SHAPE))] * 3
    with pytest.raises(RagError, match="lexical-only"):
        choose_signal(lexical)


def test_too_little_data_and_bad_options_are_refused() -> None:
    with pytest.raises(RagError, match="at least 20"):
        fit(separable(10))
    with pytest.raises(RagError, match="--folds"):
        fit(separable(), folds=50)
    with pytest.raises(RagError, match="max-false"):
        fit(separable(), max_false=1.5)


def test_fewer_than_thirty_per_class_is_called_anecdotal() -> None:
    result = fit(separable(25))
    assert result.anecdotal and any("anecdotal" in w for w in result.warnings)


def test_overlapping_scores_give_a_below_target_estimate_that_says_so() -> None:
    rows = [obs(True, 0.5 + (i % 7) / 100) for i in range(40)]
    rows += [obs(False, 0.5 + (i % 7) / 100) for i in range(40)]
    result = fit(rows)
    assert not result.meets_target and result.thresholds.below_target
    assert any("below its targets" in w for w in result.warnings)


def test_questions_without_hits_count_as_refused_in_the_estimate() -> None:
    rows = separable(30) + [Observation(True, None)] * 3 + [Observation(False, None)] * 3
    result = fit(rows)
    assert result.estimate.false_refusals == 3 and result.estimate.correct_refusals == 33
    assert result.thresholds.details is not None and result.thresholds.details["no_hits"] == 6


def test_cross_validation_is_deterministic_and_covers_every_question_once() -> None:
    rows = separable(30)
    a = cross_validate(rows, "dense", folds=5, max_false=0.1, seed=1)
    b = cross_validate(rows, "dense", folds=5, max_false=0.1, seed=1)
    assert a == b and a.answerable == 30 and a.unanswerable == 30


def test_the_holdout_is_scored_with_the_fitted_threshold_only() -> None:
    held = [obs(True, 0.85), obs(True, 0.2), obs(False, 0.1), obs(False, 0.95)]
    result = fit(separable(), holdout=held, holdout_dataset="h@v1")
    assert result.holdout is not None
    assert (result.holdout.false_refusals, result.holdout.correct_refusals) == (1, 1)
    assert any("holdout" in w and "below" in w for w in result.warnings)


# ------------------------------------------------------------------- the gate and its record


def record(**changes: Any) -> Thresholds:
    return replace(fit(separable()).thresholds, **changes)


def decide(thresholds: Thresholds, signals: Signals, environment: str = "env") -> GateDecision:
    return SufficiencyGate(thresholds, MODEL, environment).decide(signals)


def test_a_bound_record_applies_only_to_the_setup_it_was_made_for() -> None:
    t = record()
    weak = Signals(0.1, None, None, SHAPE)
    assert decide(t, weak).reasons == ("dense_below_threshold",)
    other = Signals(0.1, None, None, RetrievalShape("hybrid", 8, 40, False))
    assert decide(t, other) == GateDecision(True, False, ("calibration_stale",))
    deeper = Signals(0.1, None, None, RetrievalShape("dense", 3, 40, False))
    assert decide(t, deeper).reasons == ("calibration_stale",)
    reranked = Signals(0.1, 2.0, None, RetrievalShape("dense", 8, 40, True))
    assert decide(t, reranked).reasons == ("calibration_stale",)
    assert decide(t, weak, environment="another chunker").reasons == ("calibration_stale",)


def test_an_older_rule_or_another_model_is_never_applied() -> None:
    weak = Signals(0.1, None, None, SHAPE)
    assert decide(record(algorithm=1), weak).reasons == ("calibration_outdated",)
    assert SufficiencyGate(record(), "other/model", "env").decide(weak).reasons == (
        "calibration_invalid",
    )


def test_below_target_travels_with_the_decision_and_becomes_a_warning() -> None:
    t = record(below_target=True)
    decision = decide(t, Signals(0.9, None, None, SHAPE))
    assert decision.sufficient and decision.below_target
    assert any("targets" in w for w in gate_warnings(decision))
    assert gate_warnings(GateDecision(True, False, ("uncalibrated",)))[0].endswith(
        "`rag calibrate`"
    )
    assert gate_warnings(None) == () and gate_warnings(GateDecision(True, True)) == ()


def test_the_record_round_trips_and_a_legacy_record_is_outdated() -> None:
    t = record()
    again = thresholds_from_metadata(thresholds_to_metadata(t))
    assert again is not None
    assert (again.dense, again.binding, again.algorithm) == (t.dense, t.binding, t.algorithm)
    legacy = thresholds_from_metadata({"sufficiency": {"model_id": MODEL, "dense": 0.5}})
    assert legacy is not None and legacy.algorithm == 0
    assert decide(legacy, Signals(0.1, None, None, SHAPE)).reasons == ("calibration_outdated",)
    with pytest.raises(RagError):
        thresholds_from_metadata({"sufficiency": {"model_id": MODEL, "algorithm": "2"}})


def test_problems_are_listed_in_words() -> None:
    t = record(below_target=True)
    problems = calibration_problems(t, active_model="other", documents=100)
    joined = " ".join(problems)
    assert "another embedding model" in joined and "changed a lot" in joined
    assert "below the targets" in joined
    assert calibration_problems(record(), active_model=MODEL, documents=10) == []


def test_a_repository_dataset_is_a_source_of_questions_and_review_matters() -> None:
    path = Path(__file__).resolve().parents[2] / "eval/datasets/retrieval_mini_v1.json"
    unreviewed = load_question_set(path, include_unreviewed=True)
    assert unreviewed.exploratory and unreviewed.identifier == "retrieval-mini@v1"
    assert unreviewed.answerable > 0 and unreviewed.unanswerable > 0
    assert load_question_set(path).questions == ()  # nothing in it was accepted by a person
