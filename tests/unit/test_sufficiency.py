"""The sufficiency gate, its signals, threshold calibration and calibration validity."""

from __future__ import annotations

import pytest

from rag_engine.domain import (
    ChunkId,
    ChunkRef,
    ErrorCode,
    HitScores,
    Language,
    RagError,
    RetrievalHit,
)
from rag_engine.retrieval.sufficiency import (
    GateDecision,
    Signals,
    SufficiencyGate,
    Thresholds,
    calibrate,
    lexical_coverage,
    signals_from,
    thresholds_from_metadata,
    thresholds_to_metadata,
)
from tests.factories import make_ids

pytestmark = pytest.mark.unit

IDS = make_ids()
VERSION = IDS.document_version_id()
DOC = IDS.document_id()
MODEL = "onnx/e5@abc#d384"


def hit(rank: int, **scores: float | None) -> RetrievalHit:
    chunk = ChunkId.derive(VERSION, "gate", rank)
    return RetrievalHit(
        chunk_ref=ChunkRef(chunk, DOC, VERSION), rank=rank, scores=HitScores(**scores)
    )


def test_calibration_respects_the_false_refusal_limit_and_refuses_the_most_negatives() -> None:
    positives = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.3]  # answerable, top score
    negatives = [0.62, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.2, 0.15, 0.1]  # no answer
    result = calibrate(positives, negatives, max_false_refusal=0.10)

    def rates(threshold: float) -> tuple[float, float]:
        return (
            sum(p < threshold for p in positives) / len(positives),
            sum(n < threshold for n in negatives) / len(negatives),
        )

    assert rates(result.threshold) == pytest.approx(
        (result.false_refusal_rate, result.correct_refusal_rate)
    )
    assert result.false_refusal_rate <= 0.10
    # no feasible threshold refuses more negatives: brute force over every candidate
    best = max(rates(t)[1] for t in sorted({*positives, *negatives, 1.0}) if rates(t)[0] <= 0.10)
    assert result.correct_refusal_rate == pytest.approx(best)
    assert result.positives == 10 and result.negatives == 10


def test_a_set_that_cannot_reach_the_target_says_so() -> None:
    result = calibrate([0.5, 0.6, 0.7, 0.8], [0.55, 0.65, 0.75, 0.85])  # fully overlapping
    assert not result.meets_target
    clean = calibrate([0.8, 0.9, 0.85, 0.95] * 5, [0.1, 0.2, 0.15, 0.3] * 5)
    assert clean.meets_target and clean.false_refusal_rate == 0 and clean.correct_refusal_rate == 1


def test_calibration_needs_both_kinds_of_questions() -> None:
    with pytest.raises(ValueError, match="both"):
        calibrate([], [0.1])
    with pytest.raises(ValueError, match="both"):
        calibrate([0.9], [])


def gate(**changes: float | None) -> SufficiencyGate:
    values: dict[str, float | None] = {"dense": 0.5, "rerank": None, "lexical": None}
    values.update(changes)
    return SufficiencyGate(Thresholds(MODEL, **values), MODEL)  # type: ignore[arg-type]


def test_the_dense_signal_decides_when_it_is_the_only_one() -> None:
    assert gate().decide(Signals(dense=0.6, rerank=None, lexical_coverage=None)).sufficient
    refused = gate().decide(Signals(dense=0.4, rerank=None, lexical_coverage=1.0))
    assert not refused.sufficient and refused.calibrated
    assert refused.reasons == ("dense_below_threshold",)


def test_the_reranker_decides_alone_when_it_ran() -> None:
    both = gate(rerank=2.0)
    # a good rerank logit passes even with a weak dense cosine
    assert both.decide(Signals(dense=0.4, rerank=2.5, lexical_coverage=None)).sufficient
    # a poor rerank logit refuses even when the dense cosine looks fine
    lenient = both.decide(Signals(dense=0.6, rerank=1.0, lexical_coverage=None))
    assert not lenient.sufficient and lenient.calibrated
    assert lenient.reasons == ("rerank_below_threshold",)
    # without a rerank score the dense cosine decides as before
    assert both.decide(Signals(dense=0.6, rerank=None, lexical_coverage=None)).sufficient
    assert not both.decide(Signals(dense=0.4, rerank=None, lexical_coverage=None)).sufficient


def test_lexical_coverage_decides_only_when_no_semantic_signal_exists() -> None:
    lexical = gate(dense=None, lexical=0.6)
    assert lexical.decide(Signals(dense=None, rerank=None, lexical_coverage=0.7)).sufficient
    refused = lexical.decide(Signals(dense=None, rerank=None, lexical_coverage=0.3))
    assert not refused.sufficient and refused.reasons == ("lexical_below_threshold",)
    # with a dense score present, a poor lexical overlap does not veto it
    mixed = gate(lexical=0.6)
    assert mixed.decide(Signals(dense=0.9, rerank=None, lexical_coverage=0.1)).sufficient


def test_a_missing_or_foreign_calibration_never_blocks_but_says_so() -> None:
    open_gate = SufficiencyGate(None, MODEL).decide(Signals(0.0, None, None))
    assert open_gate == GateDecision(True, False, ("uncalibrated",))
    other = SufficiencyGate(Thresholds("another/model", dense=0.5), MODEL)
    foreign = other.decide(Signals(dense=0.0, rerank=None, lexical_coverage=None))
    assert foreign == GateDecision(True, False, ("calibration_invalid",))
    nothing = gate().decide(Signals(None, None, None))
    assert nothing.sufficient and not nothing.calibrated and nothing.reasons == ("no_signal",)


def test_signals_use_the_best_score_found_among_the_hits() -> None:
    hits = [
        hit(1, fused=0.3, lexical=4.0),
        hit(2, dense=0.41, fused=0.2),
        hit(3, dense=0.55, rerank=1.5),
    ]
    signals = signals_from(hits, "backup schedule", "Backups run nightly", Language.EN)
    assert signals.dense == 0.55 and signals.rerank == 1.5
    assert signals.lexical_coverage == pytest.approx(0.5)  # "backup" is found, "schedule" is not
    empty = signals_from([], "anything", None, Language.EN)
    assert (empty.dense, empty.rerank, empty.lexical_coverage) == (None, None, None)


def test_lexical_coverage_is_the_share_of_query_terms_found_after_stemming() -> None:
    assert lexical_coverage("running backups", "the backup ran", Language.EN) == pytest.approx(0.5)
    assert lexical_coverage("резервные копии", "Копии хранятся 10 дней", Language.RU) == 0.5
    assert lexical_coverage("   ", "text", Language.EN) is None


def test_thresholds_round_trip_through_the_index_metadata() -> None:
    original = Thresholds(MODEL, dense=0.52, rerank=None, lexical=0.4, dataset="retrieval-mini@v1")
    stored = thresholds_to_metadata(original)
    assert thresholds_from_metadata(stored) == original
    assert thresholds_from_metadata({}) is None
    with pytest.raises(RagError) as info:
        thresholds_from_metadata({"sufficiency": {"model_id": 5}})
    assert info.value.code is ErrorCode.E_INTEGRITY
