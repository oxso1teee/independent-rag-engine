from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import ChunkId, ChunkPolicy, DocumentStatus, RagError
from rag_engine.domain.invariants import (
    check_chunk_ids,
    check_chunk_layout,
    check_chunk_span,
    check_chunk_within_policy,
    check_current_version,
)
from tests.factories import FINGERPRINT, amend, make_chunk, make_document, make_ids, make_version

IDS = make_ids()
SOURCE_ID = IDS.source_id()
DOC_ID = IDS.document_id()
VERSION_ID = IDS.document_version_id()
TEXT = "Alpha beta.\n\nGamma delta epsilon."


def test_chunk_span_matches_the_normalised_text() -> None:
    chunk = make_chunk("Gamma delta epsilon.", version_id=VERSION_ID, start=13)
    check_chunk_span(chunk, TEXT)
    with pytest.raises(RagError):
        check_chunk_span(amend(chunk, char_start=12, char_end=32), TEXT)
    with pytest.raises(RagError):
        check_chunk_span(chunk, "short")


@given(st.text(min_size=1, max_size=200), st.data())
def test_every_slice_satisfies_the_span_invariant(text: str, data: st.DataObject) -> None:
    start = data.draw(st.integers(0, len(text) - 1))
    end = data.draw(st.integers(start + 1, len(text)))
    chunk = make_chunk(text[start:end], version_id=VERSION_ID, start=start)
    check_chunk_span(chunk, text)


def test_token_limit_applies_to_leaves_unless_oversize() -> None:
    policy = ChunkPolicy(target_tokens=10, max_tokens=20, min_tokens=2)
    check_chunk_within_policy(make_chunk(version_id=VERSION_ID, token_count=20), policy)
    big = make_chunk(version_id=VERSION_ID, token_count=21)
    with pytest.raises(RagError):
        check_chunk_within_policy(big, policy)
    check_chunk_within_policy(amend(big, oversize=True), policy)
    section = make_chunk("section", version_id=VERSION_ID, level=1, token_count=500)
    check_chunk_within_policy(section, policy)


def test_chunk_ids_must_be_derived() -> None:
    chunks = [make_chunk(f"c{i}", version_id=VERSION_ID, ordinal=i, start=i * 3) for i in range(3)]
    check_chunk_ids(chunks, FINGERPRINT)
    with pytest.raises(RagError):
        check_chunk_ids(chunks, "another-fingerprint")
    forged = amend(chunks[0], id=ChunkId.derive(VERSION_ID, FINGERPRINT, 99))
    with pytest.raises(RagError):
        check_chunk_ids([forged], FINGERPRINT)


def test_chunk_layout_accepts_a_parent_with_ordered_leaves() -> None:
    section = make_chunk("whole section", version_id=VERSION_ID, ordinal=0, level=1)
    leaf_a = make_chunk("whole", version_id=VERSION_ID, ordinal=1, parent_id=section.id)
    leaf_b = make_chunk("section", version_id=VERSION_ID, ordinal=2, start=6, parent_id=section.id)
    check_chunk_layout([section, leaf_a, leaf_b])
    check_chunk_layout([])


def test_chunk_layout_rejects_violations() -> None:
    section = make_chunk("whole section", version_id=VERSION_ID, ordinal=0, level=1)
    leaf_a = make_chunk("whole", version_id=VERSION_ID, ordinal=1, parent_id=section.id)
    leaf_b = make_chunk("section", version_id=VERSION_ID, ordinal=2, start=6, parent_id=section.id)

    with pytest.raises(RagError):  # duplicate ordinal
        check_chunk_layout([leaf_a, amend(leaf_b, ordinal=1)])
    with pytest.raises(RagError):  # duplicate id
        check_chunk_layout([leaf_a, leaf_a])
    with pytest.raises(RagError):  # unknown parent
        check_chunk_layout([leaf_a])
    with pytest.raises(RagError):  # parent is itself a leaf
        check_chunk_layout([leaf_a, amend(leaf_b, parent_id=leaf_a.id)])
    with pytest.raises(RagError):  # leaf offsets go backwards
        check_chunk_layout([section, amend(leaf_a, char_start=10, char_end=15), leaf_b])
    other_version = make_ids().document_version_id()
    with pytest.raises(RagError):  # mixed versions
        check_chunk_layout([leaf_a, make_chunk("x", version_id=other_version, ordinal=5)])


def test_current_version_belongs_to_its_document() -> None:
    document = make_document(document_id=DOC_ID, source_id=SOURCE_ID, version_id=VERSION_ID)
    version = make_version(version_id=VERSION_ID, document_id=DOC_ID)
    check_current_version(document, version)
    with pytest.raises(RagError):
        check_current_version(document, amend(version, document_id=IDS.document_id()))
    with pytest.raises(RagError):
        check_current_version(
            amend(document, current_version_id=IDS.document_version_id()), version
        )
    pending = make_document(
        document_id=DOC_ID,
        source_id=SOURCE_ID,
        version_id=None,
        status=DocumentStatus.FAILED,
    )
    with pytest.raises(RagError):
        check_current_version(pending, version)
