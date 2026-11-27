"""The committed mini corpus and question set stay valid and keep their promised coverage."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rag_engine.evaluation.dataset import load_dataset, validate_dataset

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_the_retrieval_mini_dataset_is_valid_against_its_corpus() -> None:
    dataset = load_dataset(ROOT / "eval/datasets/retrieval_mini_v1.json")
    corpus_dir = ROOT / dataset.corpus
    corpus = {path.name: path.read_text(encoding="utf-8") for path in corpus_dir.glob("*.md")}
    report = validate_dataset(dataset, corpus)
    assert report.errors == ()
    assert dataset.identifier == "retrieval-mini@v1"

    languages = Counter(question.language for question in dataset.questions)
    kinds = Counter(question.type for question in dataset.questions)
    assert len(dataset.questions) >= 40 and min(languages.values()) >= 20
    assert {"keyword", "paraphrase", "multi_doc", "no_answer"} <= set(kinds)
    assert {question.split for question in dataset.questions} == {"dev", "test"}
