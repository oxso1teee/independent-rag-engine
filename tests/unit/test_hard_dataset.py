"""The harder public dataset: reproducible from its builder, valid, balanced and unreviewed."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from rag_engine.evaluation.dataset import accepted_questions, load_dataset, validate_dataset

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "datasets" / "retrieval_hard_v2.json"
CORPUS = ROOT / "eval" / "corpus" / "hard_v1"


def builder() -> object:
    spec = importlib.util.spec_from_file_location(
        "build_hard_dataset", ROOT / "eval" / "scripts" / "build_hard_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_files_are_exactly_what_the_builder_produces() -> None:
    corpus, dataset, _, _ = builder().produce()  # type: ignore[attr-defined]
    assert json.loads(DATASET.read_text(encoding="utf-8")) == dataset
    for name, text in corpus.items():
        assert (CORPUS / name).read_text(encoding="utf-8") == text
    assert {p.name for p in CORPUS.glob("*.md")} == set(corpus)


def test_the_dataset_validates_against_its_corpus() -> None:
    dataset = load_dataset(DATASET)
    corpus = {p.name: p.read_text(encoding="utf-8") for p in CORPUS.glob("*.md")}
    report = validate_dataset(dataset, corpus)
    assert report.ok, report.errors


def test_nothing_in_it_counts_as_reviewed() -> None:
    dataset = load_dataset(DATASET)
    assert accepted_questions(dataset) == []
    assert {q.review.status for q in dataset.questions} == {"unreviewed"}
    assert all(q.review.reviewer is None for q in dataset.questions)
    assert {q.provenance.created_by for q in dataset.questions} == {"llm-assisted"}


def test_it_covers_the_hard_cases_in_both_languages_and_both_splits() -> None:
    dataset = load_dataset(DATASET)
    types = Counter(q.type for q in dataset.questions)
    assert types["conflicting"] >= 4 and types["multi_doc"] >= 4 and types["no_answer"] >= 30
    assert {q.language for q in dataset.questions} == {"en", "ru"}
    for split in ("dev", "test"):
        members = [q for q in dataset.questions if q.split == split]
        assert (
            sum(q.answerable for q in members) >= 50
            and sum(not q.answerable for q in members) >= 15
        )
    cross = [q for q in dataset.questions if q.id.endswith("-cross")]
    assert len(cross) >= 40 and {q.language for q in cross} == {"en", "ru"}


def test_no_gold_document_is_shared_between_the_splits() -> None:
    dataset = load_dataset(DATASET)
    split_of_doc: dict[str, set[str]] = {}
    for q in dataset.questions:
        for ref in q.gold:
            split_of_doc.setdefault(ref.doc, set()).add(q.split)
    assert all(len(splits) == 1 for splits in split_of_doc.values())


def test_the_corpus_has_hard_negatives_and_contradictory_versions() -> None:
    names = {p.name for p in CORPUS.glob("*.md")}
    assert {"en_db_backup.md", "en_laptop_backup.md"} <= names  # same structure, other numbers
    assert {
        "en_vacation_v1.md",
        "en_vacation_v2.md",
        "en_remote_2023.md",
        "en_remote_2024.md",
    } <= names
    v1 = (CORPUS / "en_vacation_v1.md").read_text(encoding="utf-8")
    v2 = (CORPUS / "en_vacation_v2.md").read_text(encoding="utf-8")
    assert "20 days" in v1 and "25 days" in v2 and "20 days" not in v2
