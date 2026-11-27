"""Duplicate collapsing and the per-document cap, in rank order."""

from __future__ import annotations

import pytest

from rag_engine.domain import ChunkId, ChunkRef, DocumentId, TextHash
from rag_engine.ports import ChunkInfo
from rag_engine.retrieval.diversity import diversify
from rag_engine.retrieval.fusion import FusedCandidate
from tests.factories import make_ids

pytestmark = pytest.mark.unit

IDS = make_ids()
VERSION = IDS.document_version_id()
DOC_A, DOC_B, DOC_C = IDS.document_id(), IDS.document_id(), IDS.document_id()


def chunk(ordinal: int) -> ChunkId:
    return ChunkId.derive(VERSION, "diversity", ordinal)


def info(ordinal: int, document: DocumentId, text: str) -> ChunkInfo:
    return ChunkInfo(ChunkRef(chunk(ordinal), document, VERSION), TextHash.of_text(text))


def candidates(count: int) -> list[FusedCandidate]:
    return [FusedCandidate(chunk(i), 1.0 - i / 100) for i in range(count)]


def catalog(*entries: ChunkInfo) -> dict[ChunkId, ChunkInfo]:
    return {entry.ref.chunk_id: entry for entry in entries}


def test_the_first_of_identical_texts_survives_even_across_documents() -> None:
    known = catalog(
        info(0, DOC_A, "same"),
        info(1, DOC_B, "same"),
        info(2, DOC_B, "other"),
        info(3, DOC_C, "same"),
    )
    outcome = diversify(candidates(4), known, top_k=10)
    assert [kept.chunk_id for kept, _ in outcome.kept] == [chunk(0), chunk(2)]
    assert outcome.collapsed_duplicates == 2


def test_a_document_never_fills_more_than_its_share() -> None:
    known = catalog(*(info(i, DOC_A if i < 4 else DOC_B, f"text {i}") for i in range(6)))
    outcome = diversify(candidates(6), known, top_k=10, max_per_document=2)
    assert [kept.chunk_id for kept, _ in outcome.kept] == [chunk(0), chunk(1), chunk(4), chunk(5)]
    assert outcome.capped == 2


def test_no_cap_means_no_limit_and_rank_order_is_kept() -> None:
    known = catalog(*(info(i, DOC_A, f"text {i}") for i in range(5)))
    outcome = diversify(candidates(5), known, top_k=10, max_per_document=None)
    assert [kept.chunk_id for kept, _ in outcome.kept] == [chunk(i) for i in range(5)]
    assert outcome.capped == 0


def test_the_result_stops_at_top_k_and_later_skips_are_not_counted() -> None:
    known = catalog(*(info(i, DOC_A, f"text {i}") for i in range(6)))
    outcome = diversify(candidates(6), known, top_k=3)
    assert len(outcome.kept) == 3
    assert outcome.collapsed_duplicates == 0 and outcome.capped == 0


def test_a_candidate_the_catalog_does_not_know_is_dropped_and_reported() -> None:
    known = catalog(info(0, DOC_A, "a"), info(2, DOC_B, "c"))
    outcome = diversify(candidates(3), known, top_k=10)
    assert [kept.chunk_id for kept, _ in outcome.kept] == [chunk(0), chunk(2)]
    assert outcome.unknown == 1


def test_document_share_is_unique_documents_over_kept() -> None:
    known = catalog(
        info(0, DOC_A, "a"), info(1, DOC_A, "b"), info(2, DOC_B, "c"), info(3, DOC_C, "d")
    )
    assert diversify(candidates(4), known, top_k=10).document_share == pytest.approx(3 / 4)
    assert diversify([], {}, top_k=10).document_share == 0.0


@pytest.mark.parametrize(("top_k", "cap"), [(0, None), (-1, None), (5, 0), (5, -2)])
def test_limits_must_be_positive(top_k: int, cap: int | None) -> None:
    with pytest.raises(ValueError, match="positive"):
        diversify([], {}, top_k=top_k, max_per_document=cap)
