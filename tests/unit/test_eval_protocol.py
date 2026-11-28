"""Manifests, the guarded test split, comparison, reports and the regression gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.evaluation.protocol import (
    Manifest,
    TestSplitGuard,
    build_run,
    check_regression,
    compare_runs,
    load_run,
    render_comparison,
    render_report,
    write_run,
)
from rag_engine.evaluation.runner import QuestionScore

pytestmark = pytest.mark.unit

METRICS = ("hit@1", "ndcg@10")


def manifest(**changes: object) -> Manifest:
    base: dict[str, object] = {
        "mode": "retrieval", "split": "dev", "final": False, "dataset": "demo@v1",
        "dataset_hash": "ab" * 32, "corpus_hash": "cd" * 32, "config_hash": "ef" * 32,
        "embedding_model": "hashing/1", "llm": None, "judge": None, "context_window": None,
        "index_generation": 3, "seed": 7, "engine_version": "0.0.1", "git_commit": "abc1234",
        "resources": {"mem_available_mib": 3000.0}, "exploratory": True,
        "created_at": "2026-11-28T00:00:00+00:00",
    }  # fmt: skip
    base.update(changes)
    return Manifest(**base)  # type: ignore[arg-type]


def scores(
    values: list[tuple[float, float]], kinds: tuple[str, ...] = ("keyword",)
) -> list[QuestionScore]:
    return [
        QuestionScore(
            f"q{i}", kinds[i % len(kinds)], "en" if i % 2 else "ru", {"hit@1": h, "ndcg@10": n}
        )
        for i, (h, n) in enumerate(values)
    ]


def run_of(values: list[tuple[float, float]], **changes: object) -> dict[str, Any]:
    return build_run(manifest(**changes), scores(values), METRICS)


# ------------------------------------------------------------------------------ test split


def test_the_test_split_needs_final_and_every_read_is_counted(tmp_path: Path) -> None:
    guard = TestSplitGuard(tmp_path / "access.json")
    with pytest.raises(RagError) as info:
        guard.read("demo@v1", final=False)
    assert info.value.code is ErrorCode.E_QUERY_INVALID and guard.reads("demo@v1") == 0
    assert guard.read("demo@v1", final=True) == 1
    assert guard.read("demo@v1", final=True) == 2
    assert guard.read("other@v1", final=True) == 1
    assert TestSplitGuard(tmp_path / "access.json").reads("demo@v1") == 2  # survives a restart


def test_a_corrupt_counter_is_an_integrity_error_not_a_reset(tmp_path: Path) -> None:
    path = tmp_path / "access.json"
    path.write_text("{not json")
    with pytest.raises(RagError) as info:
        TestSplitGuard(path).read("d", final=True)
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_a_repeated_read_makes_the_report_say_the_split_is_no_longer_held_out() -> None:
    once = build_run(
        manifest(split="test", final=True), scores([(1.0, 1.0)] * 30), METRICS, test_reads=1
    )
    twice = build_run(
        manifest(split="test", final=True), scores([(1.0, 1.0)] * 30), METRICS, test_reads=2
    )
    assert once["flags"]["held_out"] is True and twice["flags"]["held_out"] is False
    assert "NOT held out" in render_report(twice) and "NOT held out" not in render_report(once)


# ---------------------------------------------------------------------------- run and report


def test_a_run_records_the_manifest_the_scores_and_the_warnings() -> None:
    run = run_of([(1.0, 1.0)] * 10)
    assert (
        run["manifest"]["dataset_hash"] == "ab" * 32 and run["manifest"]["git_commit"] == "abc1234"
    )
    assert len(run["questions"]) == 10 and run["summary"]["all"]["hit@1"]["n"] == 10
    assert "type:keyword" in run["summary"] and "language:en" in run["summary"]
    assert run["flags"]["sample"].startswith("anecdotal") and run["flags"]["exploratory"] is True
    assert run["flags"]["saturated_metrics"] == ["hit@1", "ndcg@10"]


def test_the_report_flags_exploratory_labels_small_samples_and_saturation() -> None:
    text = render_report(run_of([(1.0, 1.0)] * 10))
    assert "EXPLORATORY" in text and "anecdotal" in text and "saturated" in text
    reviewed = render_report(
        build_run(manifest(exploratory=False), scores([(0.5, 0.5)] * 40), METRICS)
    )
    assert "EXPLORATORY" not in reviewed and "saturated" not in reviewed.split("|")[0]


def test_runs_round_trip_through_a_file_and_foreign_files_are_refused(tmp_path: Path) -> None:
    run = run_of([(1.0, 0.8), (0.0, 0.2)])
    path = tmp_path / "r.json"
    write_run(run, path)
    assert load_run(path) == json.loads(json.dumps(run))
    path.write_text('{"hello": 1}')
    with pytest.raises(RagError):
        load_run(path)
    with pytest.raises(RagError):
        load_run(tmp_path / "missing.json")


# ------------------------------------------------------------------------------ comparison


def test_a_comparison_is_paired_and_tests_with_the_right_test_for_the_metric() -> None:
    better = run_of([(1.0, 0.9)] * 18 + [(0.0, 0.1)] * 2)
    worse = run_of([(1.0, 0.7)] * 10 + [(0.0, 0.1)] * 10)
    rows = {r.metric: r for r in compare_runs(better, worse)}
    assert rows["hit@1"].mean_diff == pytest.approx(0.4) and rows["hit@1"].mcnemar_p is not None
    assert rows["ndcg@10"].mcnemar_p is None  # a graded metric has no McNemar test
    assert rows["hit@1"].significant and rows["hit@1"].low > 0
    assert "significant" in render_comparison(list(rows.values()), "A", "B")


def test_equal_runs_show_no_evidence_of_a_difference() -> None:
    run = run_of([(1.0, 0.9), (0.0, 0.1)] * 5)
    for row in compare_runs(run, run):
        assert row.mean_diff == 0 and not row.significant


def test_runs_with_no_common_question_cannot_be_compared() -> None:
    other = build_run(
        manifest(), [QuestionScore("zz", "keyword", "en", {"hit@1": 1.0, "ndcg@10": 1.0})], METRICS
    )
    with pytest.raises(ValueError, match="no questions"):
        compare_runs(run_of([(1.0, 1.0)]), other)


# ------------------------------------------------------------------------------- the gate


def test_the_gate_passes_an_equal_or_better_run_and_catches_a_worse_one() -> None:
    baseline = run_of([(1.0, 0.9)] * 20 + [(0.0, 0.1)] * 10)
    assert check_regression(baseline, baseline) == []
    better = run_of([(1.0, 0.95)] * 25 + [(0.0, 0.1)] * 5)
    assert check_regression(better, baseline) == []
    worse = run_of([(1.0, 0.9)] * 12 + [(0.0, 0.1)] * 18)
    violations = check_regression(worse, baseline)
    assert {v.metric for v in violations} == {"hit@1"} or "hit@1" in {v.metric for v in violations}
    assert any("floor" in v.reason or "paired" in v.reason for v in violations)


def test_a_small_drop_within_the_margin_does_not_fail_the_gate() -> None:
    baseline = run_of([(1.0, 0.90)] * 100)
    slightly = run_of([(1.0, 0.89)] * 100)
    assert check_regression(slightly, baseline, delta=0.03) == []
    assert check_regression(slightly, baseline, delta=0.001) != []


def test_a_different_question_set_is_judged_on_floors_only() -> None:
    baseline = run_of([(1.0, 0.9)] * 10)
    other = build_run(
        manifest(),
        [
            QuestionScore(f"x{i}", "keyword", "en", {"hit@1": 0.0, "ndcg@10": 0.0})
            for i in range(10)
        ],
        METRICS,
    )
    assert {v.metric for v in check_regression(other, baseline)} == {"hit@1", "ndcg@10"}
