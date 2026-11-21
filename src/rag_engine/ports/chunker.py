"""Chunker port: parsed document to chunks with provenance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_engine.domain import Chunk, ChunkPolicy, DocumentVersionId, ParsedDocument


@runtime_checkable
class Chunker(Protocol):
    @property
    def chunker_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def fingerprint(self, policy: ChunkPolicy, normalizer_version: str) -> str:
        """Stable fingerprint of everything that influences chunk boundaries."""
        ...

    def chunk(
        self,
        document: ParsedDocument,
        policy: ChunkPolicy,
        *,
        version_id: DocumentVersionId,
        normalizer_version: str,
    ) -> Sequence[Chunk]:
        """Deterministic: same input and policy give identical chunks and ids."""
        ...
