"""Vector index port: nearest-neighbour search over chunk embeddings."""

from __future__ import annotations

from collections.abc import Sequence, Set
from typing import Protocol, runtime_checkable

from rag_engine.domain import ChunkId, EmbeddingModelId, ScoredChunk
from rag_engine.ports.embedding import Vectors


@runtime_checkable
class VectorIndex(Protocol):
    @property
    def model_id(self) -> EmbeddingModelId:
        """The single embedding model whose vectors this index holds."""
        ...

    def __len__(self) -> int: ...

    def upsert(self, ids: Sequence[ChunkId], vectors: Vectors) -> None:
        """Insert or replace; ``vectors`` has shape ``(len(ids), dimension)``."""
        ...

    def remove(self, ids: Sequence[ChunkId]) -> None: ...

    def search(
        self, query: Vectors, k: int, allowed: Set[ChunkId] | None = None
    ) -> Sequence[ScoredChunk]:
        """Top ``k`` by descending cosine score, ties broken by ``ChunkId``.

        ``allowed`` restricts candidates (prefilter); ``None`` means no restriction.
        """
        ...
