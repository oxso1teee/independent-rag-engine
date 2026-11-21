"""Metadata store port (system of record)."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from rag_engine.domain import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    Source,
    SourceId,
    SourceKey,
)


@runtime_checkable
class Store(Protocol):
    """Persistent entities. Writes happen inside :meth:`transaction`."""

    def transaction(self) -> AbstractContextManager[None]:
        """Atomic unit of work: commits on normal exit, rolls back on exception."""
        ...

    def get_source(self, source_id: SourceId) -> Source | None: ...

    def get_source_by_name(self, name: str) -> Source | None: ...

    def put_source(self, source: Source) -> None: ...

    def get_document(self, document_id: DocumentId) -> Document | None: ...

    def get_document_by_key(self, source_id: SourceId, key: SourceKey) -> Document | None: ...

    def put_document(self, document: Document) -> None: ...

    def get_version(self, version_id: DocumentVersionId) -> DocumentVersion | None: ...

    def put_version(self, version: DocumentVersion) -> None: ...

    def get_chunks(self, chunk_ids: Sequence[ChunkId]) -> Sequence[Chunk]:
        """Chunks in the order of ``chunk_ids``; unknown ids are skipped."""
        ...

    def put_chunks(self, chunks: Sequence[Chunk]) -> None: ...
