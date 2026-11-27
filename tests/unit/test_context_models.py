"""Invariants of the evidence, budget and report models."""

from __future__ import annotations

import pytest

from rag_engine.domain import (
    ChunkId,
    ContextBudget,
    ContextReport,
    DropReason,
    ErrorCode,
    Evidence,
    EvidenceLocator,
    RagError,
)
from tests.factories import make_ids

pytestmark = pytest.mark.unit

IDS = make_ids()
VERSION = IDS.document_version_id()
DOC = IDS.document_id()
CHUNK = ChunkId.derive(VERSION, "ctx", 0)


def locator(**changes: object) -> EvidenceLocator:
    base: dict[str, object] = {
        "source_name": "notes",
        "source_key": "a.md",
        "title": None,
        "heading_path": ("Backups",),
        "page_start": None,
        "page_end": None,
        "char_start": 10,
        "char_end": 50,
    }
    base.update(changes)
    return EvidenceLocator(**base)  # type: ignore[arg-type]


def evidence(**changes: object) -> Evidence:
    base: dict[str, object] = {
        "evidence_id": "E1",
        "chunk_id": CHUNK,
        "matched": (CHUNK,),
        "document_id": DOC,
        "version_id": VERSION,
        "text": "Backups run nightly.",
        "truncated": False,
        "locator": locator(),
        "score": 0.5,
        "tokens": 6,
    }
    base.update(changes)
    return Evidence(**base)  # type: ignore[arg-type]


def test_a_valid_evidence_keeps_its_provenance() -> None:
    item = evidence()
    assert item.locator.source_key == "a.md" and item.matched == (CHUNK,)


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_id": "X1"},
        {"evidence_id": "E0"},
        {"text": ""},
        {"tokens": 0},
        {"matched": ()},
        {"score": float("nan")},
    ],
)
def test_invalid_evidence_is_refused(change: dict[str, object]) -> None:
    with pytest.raises(RagError) as info:
        evidence(**change)
    assert info.value.code is ErrorCode.E_INTEGRITY


@pytest.mark.parametrize(
    "change",
    [{"char_start": 50, "char_end": 10}, {"page_start": 3, "page_end": 2}, {"page_start": 2}],
)
def test_an_invalid_locator_is_refused(change: dict[str, object]) -> None:
    with pytest.raises(RagError) as info:
        locator(**change)
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_the_budget_adds_up() -> None:
    budget = ContextBudget(
        window=4096, system=300, question=40, max_new=512, margin=410, evidence_budget=2834
    )
    assert budget.window - 300 - 40 - 512 - 410 == budget.evidence_budget
    with pytest.raises(RagError):
        ContextBudget(
            window=4096, system=300, question=40, max_new=512, margin=410, evidence_budget=9999
        )
    with pytest.raises(RagError):
        ContextBudget(window=0, system=0, question=0, max_new=0, margin=0, evidence_budget=0)


def test_a_report_must_not_use_more_than_its_budget() -> None:
    ContextReport(
        included=(),
        dropped=(),
        merged=(),
        truncated=(),
        tokens_used=10,
        budget=10,
        context_hash="a" * 64,
    )
    with pytest.raises(RagError):
        ContextReport(
            included=(),
            dropped=(),
            merged=(),
            truncated=(),
            tokens_used=11,
            budget=10,
            context_hash="a" * 64,
        )


def test_drop_reasons_are_the_documented_ones() -> None:
    assert {reason.value for reason in DropReason} == {
        "duplicate",
        "budget",
        "doc_cap",
        "overlap",
        "missing",
    }
