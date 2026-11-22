"""Cross-entity invariants of the data model.

Each function raises :class:`~rag_engine.domain.errors.RagError`
(``E_INTEGRITY``) when the invariant does not hold and returns ``None`` otherwise.
Single-entity invariants are enforced by the models themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.domain.errors import require
from rag_engine.domain.ids import ChunkId
from rag_engine.domain.models import (
    Chunk,
    ChunkPolicy,
    Document,
    DocumentStatus,
    DocumentVersion,
    EmbeddingModel,
    IndexMetadata,
    ModelStatus,
    SyncState,
    SyncStatus,
)


def check_chunk_span(chunk: Chunk, normalized_text: str) -> None:
    """I3: ``normalized_text[char_start:char_end] == chunk.text``."""
    require(chunk.char_end <= len(normalized_text), "chunk span exceeds the text")
    require(
        normalized_text[chunk.char_start : chunk.char_end] == chunk.text,
        "chunk text differs from the version text",
        ordinal=chunk.ordinal,
    )


def check_chunk_within_policy(chunk: Chunk, policy: ChunkPolicy) -> None:
    """Leaf chunks respect ``max_tokens`` unless explicitly marked ``oversize``."""
    if chunk.level == 0 and not chunk.oversize:
        require(
            chunk.token_count <= policy.max_tokens,
            "leaf chunk exceeds max_tokens",
            ordinal=chunk.ordinal,
            token_count=chunk.token_count,
        )


def check_chunk_ids(chunks: Sequence[Chunk], fingerprint: str) -> None:
    """I6: chunk ids are exactly those derived from version, fingerprint and ordinal."""
    for chunk in chunks:
        expected = ChunkId.derive(chunk.document_version_id, fingerprint, chunk.ordinal)
        require(chunk.id == expected, "chunk id is not deterministic", ordinal=chunk.ordinal)


def check_chunk_layout(chunks: Sequence[Chunk]) -> None:
    """Chunks of one version: unique ordinals, known parents, ordered leaf offsets."""
    require(len({c.ordinal for c in chunks}) == len(chunks), "duplicate chunk ordinals")
    require(len({c.id for c in chunks}) == len(chunks), "duplicate chunk ids")
    require(
        len({c.document_version_id for c in chunks}) <= 1,
        "chunks belong to more than one version",
    )
    by_id = {c.id: c for c in chunks}
    last_start: dict[ChunkId | None, int] = {}
    for chunk in sorted(chunks, key=lambda c: c.ordinal):
        if chunk.parent_id is not None:
            parent = by_id.get(chunk.parent_id)
            require(parent is not None and parent.level == 1, "leaf parent is not a section")
        if chunk.level == 0:
            previous = last_start.get(chunk.parent_id, -1)
            require(chunk.char_start >= previous, "leaf offsets must not go backwards")
            last_start[chunk.parent_id] = chunk.char_start


def check_current_version(document: Document, version: DocumentVersion) -> None:
    """A document's current version belongs to that document."""
    require(version.document_id == document.id, "version belongs to another document")
    require(document.current_version_id == version.id, "version is not the current one")


def check_single_active_model(models: Sequence[EmbeddingModel], *, indexing_started: bool) -> None:
    """At most one embedding model is active; exactly one once indexing has started."""
    active = [m for m in models if m.status is ModelStatus.ACTIVE]
    require(len(active) <= 1, "more than one active embedding model", count=len(active))
    if indexing_started:
        require(len(active) == 1, "indexing started without an active embedding model")


def check_generation_monotonic(old: IndexMetadata, new: IndexMetadata) -> None:
    """I8: the index generation never decreases."""
    require(
        new.index_generation >= old.index_generation,
        "index generation went backwards",
        old=old.index_generation,
        new=new.index_generation,
    )


def check_sync_state_matches_document(document: Document, state: SyncState) -> None:
    """A sync state belongs to its document's source entry and agrees with its status."""
    require(
        state.source_id == document.source_id and state.source_key == document.source_key,
        "sync state belongs to another source entry",
    )
    require(
        (state.status is SyncStatus.MISSING) == (document.status is DocumentStatus.MISSING),
        "sync state and document disagree about a missing entry",
    )
