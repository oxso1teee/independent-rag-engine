"""Builders of valid domain objects for tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from rag_engine.domain import (
    Chunk,
    ChunkId,
    ChunkKind,
    ContentHash,
    Document,
    DocumentId,
    DocumentStatus,
    DocumentVersion,
    DocumentVersionId,
    IdFactory,
    IngestionJob,
    JobKind,
    Language,
    Sha256Hex,
    Source,
    SourceId,
    SourceKey,
    SourceKind,
    SourcePolicy,
    SyncState,
    SyncStatus,
    TextHash,
    Uuid7Generator,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
FINGERPRINT = "f" * 64


def amend[T](obj: T, **changes: Any) -> T:
    """``dataclasses.replace`` with loosely typed changes, for negative tests."""
    return replace(obj, **changes)  # type: ignore[type-var]


def make_ids(start_ms: int = 1_767_225_600_000) -> IdFactory:
    """Deterministic identifier factory: fixed clock, counting entropy."""
    state = {"n": 0}

    def rand(size: int) -> bytes:
        state["n"] += 1
        return (state["n"].to_bytes(2, "big") * size)[:size]

    return IdFactory(Uuid7Generator(lambda: start_ms, rand))


def make_chunk(
    text: str = "hello world",
    *,
    version_id: DocumentVersionId,
    ordinal: int = 0,
    start: int = 0,
    level: int = 0,
    parent_id: ChunkId | None = None,
    token_count: int = 2,
    indexed: bool | None = None,
    fingerprint: str = FINGERPRINT,
) -> Chunk:
    return Chunk(
        id=ChunkId.derive(version_id, fingerprint, ordinal),
        document_version_id=version_id,
        ordinal=ordinal,
        level=level,
        parent_id=parent_id,
        kind=ChunkKind.PARAGRAPH,
        text=text,
        text_hash=TextHash.of_text(text),
        token_count=token_count,
        char_start=start,
        char_end=start + len(text),
        language=Language.EN,
        indexed=(level == 0) if indexed is None else indexed,
    )


def make_document(
    *,
    document_id: DocumentId,
    source_id: SourceId,
    version_id: DocumentVersionId | None,
    status: DocumentStatus = DocumentStatus.ACTIVE,
    key: str = "notes/a.md",
) -> Document:
    return Document(
        id=document_id,
        source_id=source_id,
        source_key=SourceKey(key),
        current_version_id=version_id,
        status=status,
        created_at=T0,
        updated_at=T0,
    )


def make_version(*, version_id: DocumentVersionId, document_id: DocumentId) -> DocumentVersion:
    return DocumentVersion(
        id=version_id,
        document_id=document_id,
        seq=1,
        content_hash=ContentHash.of_bytes(b"content"),
        size_bytes=7,
        mtime_ns=1,
        mime="text/markdown",
        detected_by="extension",
        parser_id="md",
        parser_version="1",
        normalizer_version="1",
        chunker_fingerprint=FINGERPRINT,
        created_at=T0,
    )


def make_source(
    ids: IdFactory,
    *,
    name: str = "notes",
    locator: str = "/data/notes",
    policy: SourcePolicy | None = None,
) -> Source:
    return Source(
        id=ids.source_id(),
        name=name,
        kind=SourceKind.FS,
        locator=locator,
        locator_fingerprint=Sha256Hex.of_text(locator),
        policy=policy or SourcePolicy(),
        created_at=T0,
        updated_at=T0,
    )


def make_sync_state(source_id: SourceId, key: str = "a.md", **changes: Any) -> SyncState:
    state = SyncState(
        source_id=source_id,
        source_key=SourceKey(key),
        last_revision=None,
        last_content_hash=None,
        last_seen_at=T0,
        status=SyncStatus.OK,
    )
    return replace(state, **changes) if changes else state


def make_job(
    ids: IdFactory,
    source_id: SourceId,
    *,
    document_id: DocumentId | None = None,
    kind: JobKind = JobKind.INGEST_DOC,
    **changes: Any,
) -> IngestionJob:
    job = IngestionJob(
        id=ids.job_id(),
        kind=kind,
        source_id=source_id,
        document_id=document_id
        if document_id is not None or kind is not JobKind.INGEST_DOC
        else ids.document_id(),
        available_at=T0,
        created_at=T0,
    )
    return replace(job, **changes) if changes else job
