from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import (
    Block,
    BlockKind,
    ChunkKind,
    ChunkPolicy,
    ChunkRef,
    DocumentStatus,
    DocumentVersion,
    EmbeddingModel,
    EmbeddingModelId,
    HitScores,
    Language,
    ParsedDocument,
    RagError,
    RerankMode,
    RetrievalFilters,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
    ScoredChunk,
    Sha256Hex,
    Source,
    SourceKind,
    SourcePolicy,
    TextHash,
)
from tests.factories import T0, amend, make_chunk, make_document, make_ids, make_version

IDS = make_ids()
SOURCE_ID = IDS.source_id()
DOC_ID = IDS.document_id()
VERSION_ID = IDS.document_version_id()
REQUEST_ID = IDS.request_id()


def _fs_source(**overrides: object) -> Source:
    params: dict[str, object] = {
        "id": SOURCE_ID,
        "name": "notes",
        "kind": SourceKind.FS,
        "locator": "/data/notes",
        "locator_fingerprint": Sha256Hex.of_text("/data/notes"),
        "created_at": T0,
        "updated_at": T0,
    }
    params.update(overrides)
    return Source(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- sources


def test_valid_sources() -> None:
    assert _fs_source().policy == SourcePolicy()
    inline = _fs_source(kind=SourceKind.INLINE, locator=None, locator_fingerprint=None)
    assert inline.locator is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Bad Name"},
        {"name": ""},
        {"name": "x" * 65},
        {"locator": None, "locator_fingerprint": None},
        {"locator_fingerprint": Sha256Hex.of_text("/elsewhere")},
        {"kind": SourceKind.INLINE},
        {"created_at": datetime(2026, 1, 1)},  # noqa: DTZ001 - naive on purpose
        {"updated_at": T0 - timedelta(seconds=1)},
        {"created_at": datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=3)))},
    ],
)
def test_invalid_sources(overrides: dict[str, object]) -> None:
    with pytest.raises(RagError):
        _fs_source(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [{"max_file_bytes": 0}, {"delete_grace_days": -1}, {"include": ()}],
)
def test_invalid_source_policy(overrides: dict[str, object]) -> None:
    with pytest.raises(RagError):
        SourcePolicy(**overrides)  # type: ignore[arg-type]


def test_models_are_immutable() -> None:
    source = _fs_source()
    with pytest.raises(FrozenInstanceError):
        source.name = "other"  # type: ignore[misc]
    assert not hasattr(source, "__dict__")


# -------------------------------------------------------------------- documents


def test_document_rules() -> None:
    active = make_document(document_id=DOC_ID, source_id=SOURCE_ID, version_id=VERSION_ID)
    assert active.status is DocumentStatus.ACTIVE
    assert active.is_searchable
    # before its first commit a document is active but has no version and is not searchable
    awaiting_first_commit = make_document(document_id=DOC_ID, source_id=SOURCE_ID, version_id=None)
    assert awaiting_first_commit.status is DocumentStatus.ACTIVE
    assert awaiting_first_commit.current_version_id is None
    assert not awaiting_first_commit.is_searchable
    missing = amend(active, status=DocumentStatus.MISSING, missing_since=T0)
    assert not missing.is_searchable


def test_document_missing_since_matches_status() -> None:
    base = make_document(document_id=DOC_ID, source_id=SOURCE_ID, version_id=VERSION_ID)
    missing = amend(base, status=DocumentStatus.MISSING, missing_since=T0)
    assert missing.missing_since == T0
    with pytest.raises(RagError):
        amend(base, status=DocumentStatus.MISSING)
    with pytest.raises(RagError):
        amend(base, missing_since=T0)
    with pytest.raises(RagError):
        amend(base, missing_since=datetime(2026, 1, 1), status=DocumentStatus.MISSING)  # noqa: DTZ001


def test_document_cannot_duplicate_itself_or_travel_back_in_time() -> None:
    base = make_document(document_id=DOC_ID, source_id=SOURCE_ID, version_id=VERSION_ID)
    with pytest.raises(RagError):
        amend(base, duplicate_of=DOC_ID)
    with pytest.raises(RagError):
        amend(base, updated_at=T0 - timedelta(days=1))
    with pytest.raises(RagError):
        amend(base, created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2))  # noqa: DTZ001


def test_document_version_rules() -> None:
    version = make_version(version_id=VERSION_ID, document_id=DOC_ID)
    assert isinstance(version, DocumentVersion)
    for overrides in ({"seq": 0}, {"size_bytes": -1}, {"mtime_ns": -1}):
        with pytest.raises(RagError):
            amend(version, **overrides)
    with pytest.raises(RagError):
        amend(version, created_at=datetime(2026, 1, 1))  # noqa: DTZ001


# ---------------------------------------------------------------------- parsing


def test_block_validation() -> None:
    heading = Block(kind=BlockKind.HEADING, text="Title", char_start=0, char_end=5, level=1)
    assert heading.level == 1
    bad: list[dict[str, Any]] = [
        {"kind": BlockKind.HEADING, "level": None},
        {"kind": BlockKind.HEADING, "level": 7},
        {"kind": BlockKind.PARAGRAPH, "level": 2},
        {"kind": BlockKind.PARAGRAPH, "text": "toolong"},
        {"kind": BlockKind.PARAGRAPH, "char_start": 5, "char_end": 5, "text": ""},
        {"kind": BlockKind.PARAGRAPH, "page": 0},
    ]
    for overrides in bad:
        params: dict[str, object] = {
            "kind": BlockKind.PARAGRAPH,
            "text": "Title",
            "char_start": 0,
            "char_end": 5,
        }
        params.update(overrides)
        with pytest.raises(RagError):
            Block(**params)  # type: ignore[arg-type]
    item = Block(kind=BlockKind.LIST_ITEM, text="x", char_start=0, char_end=1, level=2)
    assert item.level == 2


def test_parsed_document_checks_block_offsets_against_the_text() -> None:
    text = "Title\n\nBody text"
    blocks = (
        Block(kind=BlockKind.HEADING, text="Title", char_start=0, char_end=5, level=1),
        Block(kind=BlockKind.PARAGRAPH, text="Body text", char_start=7, char_end=16),
    )
    parsed = ParsedDocument(normalized_text=text, blocks=blocks, language=Language.EN)
    assert parsed.blocks == blocks
    with pytest.raises(RagError):  # wrong slice
        ParsedDocument(normalized_text="Other\n\nBody text", blocks=blocks)
    with pytest.raises(RagError):  # beyond the text
        ParsedDocument(normalized_text="Title", blocks=blocks)
    with pytest.raises(RagError):  # out of order
        ParsedDocument(normalized_text=text, blocks=(blocks[1], blocks[0]))


# --------------------------------------------------------------------- chunking


def test_chunk_policy() -> None:
    assert ChunkPolicy().max_tokens == 384
    for overrides in (
        {"min_tokens": 0},
        {"min_tokens": 300},
        {"target_tokens": 400},
        {"overlap_ratio": 0.5},
        {"overlap_ratio": -0.1},
    ):
        with pytest.raises(RagError):
            ChunkPolicy(**overrides)


def test_valid_chunk() -> None:
    chunk = make_chunk(version_id=VERSION_ID)
    assert chunk.indexed
    assert not chunk.is_duplicate
    section = make_chunk("section text", version_id=VERSION_ID, ordinal=1, level=1)
    assert section.indexed is False


def test_chunk_invariants() -> None:
    base = make_chunk(version_id=VERSION_ID)
    other = make_chunk("other", version_id=VERSION_ID, ordinal=1)
    cases: list[dict[str, object]] = [
        {"ordinal": -1},
        {"level": 2},
        {"parent_id": base.id},
        {"level": 1, "indexed": False, "parent_id": other.id},
        {"text": ""},
        {"text": "changed"},  # hash mismatch
        {"token_count": -1},
        {"char_start": -1},
        {"char_end": 99},  # span/text length mismatch
        {"page_start": 1},
        {"page_start": 3, "page_end": 2},
        {"page_start": 0, "page_end": 1},
        {"level": 1},  # sections are not indexed
        {"is_duplicate": True},  # duplicates are not indexed
    ]
    for overrides in cases:
        with pytest.raises(RagError):
            amend(base, **overrides)
    assert amend(base, page_start=1, page_end=2).page_end == 2


@given(st.text(min_size=1, max_size=40), st.integers(0, 1000))
def test_chunk_text_hash_always_matches(text: str, start: int) -> None:
    chunk = make_chunk(text, version_id=VERSION_ID, start=start)
    assert chunk.text_hash == TextHash.of_text(text)
    assert chunk.char_end - chunk.char_start == len(text)


# -------------------------------------------------------------------- embedding


def _embedding_model(**overrides: object) -> EmbeddingModel:
    params: dict[str, object] = {
        "provider": "local",
        "name": "embed-small",
        "revision": "r1",
        "dimension": 1024,
        "normalized": True,
        "query_prefix": "q: ",
        "document_prefix": "d: ",
        "max_input_tokens": 8192,
        "created_at": T0,
    }
    params.update(overrides)
    model_id = EmbeddingModelId.build(
        provider=str(params["provider"]),
        name=str(params["name"]),
        revision=str(params["revision"]),
        dimension=int(params["dimension"]),  # type: ignore[call-overload]
        normalized=bool(params["normalized"]),
        query_prefix=str(params["query_prefix"]),
        document_prefix=str(params["document_prefix"]),
    )
    params.setdefault("id", model_id)
    return EmbeddingModel(**params)  # type: ignore[arg-type]


def test_embedding_model_consistency() -> None:
    assert _embedding_model().dimension == 1024
    wrong_id = EmbeddingModelId.build(
        provider="x", name="y", revision="z", dimension=8, normalized=True
    )
    with pytest.raises(RagError):
        _embedding_model(id=wrong_id)
    with pytest.raises(RagError):
        _embedding_model(max_input_tokens=0)
    with pytest.raises(RagError):
        _embedding_model(created_at=datetime(2026, 1, 1))  # noqa: DTZ001


# -------------------------------------------------------------------- retrieval


def _query(**overrides: object) -> RetrievalQuery:
    params: dict[str, object] = {
        "request_id": REQUEST_ID,
        "text": "what is hybrid retrieval?",
        "mode": RetrievalMode.HYBRID,
    }
    params.update(overrides)
    return RetrievalQuery(**params)  # type: ignore[arg-type]


def test_query_defaults_and_bounds() -> None:
    query = _query()
    assert (query.top_k, query.candidate_k, query.rerank) == (8, 40, RerankMode.AUTO)
    assert _query(top_k=100, candidate_k=500).top_k == 100
    for overrides in (
        {"text": "   "},
        {"top_k": 0},
        {"top_k": 101, "candidate_k": 200},
        {"top_k": 10, "candidate_k": 9},
        {"candidate_k": 501},
    ):
        with pytest.raises(RagError):
            _query(**overrides)


def test_filters() -> None:
    assert RetrievalFilters().sources == ()
    with pytest.raises(RagError):
        RetrievalFilters(doc_ids=(DOC_ID,), exclude_doc_ids=(DOC_ID,))
    with pytest.raises(RagError):
        RetrievalFilters(modified_after=T0, modified_before=T0)
    with pytest.raises(RagError):
        RetrievalFilters(modified_after=datetime(2026, 1, 1))  # noqa: DTZ001
    ok = RetrievalFilters(
        modified_after=T0,
        modified_before=datetime(2026, 2, 1, tzinfo=UTC),
        kinds=(ChunkKind.TABLE,),
    )
    assert ok.kinds == (ChunkKind.TABLE,)


def test_scores_must_be_finite() -> None:
    chunk_id = make_chunk(version_id=VERSION_ID).id
    assert ScoredChunk(chunk_id, 0.5).score == 0.5
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RagError):
            ScoredChunk(chunk_id, bad)
        with pytest.raises(RagError):
            HitScores(dense=bad)


def test_result_ranks_must_be_consecutive() -> None:
    chunk = make_chunk(version_id=VERSION_ID)
    ref = ChunkRef(chunk.id, DOC_ID, VERSION_ID)

    def result(*ranks: int, generation: int = 1) -> RetrievalResult:
        return RetrievalResult(
            request_id=REQUEST_ID,
            index_generation=generation,
            embedding_model_id=None,
            hits=tuple(RetrievalHit(chunk_ref=ref, rank=r) for r in ranks),
        )

    assert len(result(1, 2, 3).hits) == 3
    assert result().hits == ()
    for bad in ((2, 1), (1, 3), (0,)):
        with pytest.raises(RagError):
            result(*bad)
    with pytest.raises(RagError):
        result(1, generation=-1)
