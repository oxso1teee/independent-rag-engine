"""Content-addressed embedding cache port (implemented together with embeddings)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rag_engine.domain import EmbeddingModelId, TextHash, require


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    """One cached vector. Kept out of the domain so that it stays free of numpy."""

    embedding_input_hash: TextHash
    vector: NDArray[np.float32]
    norm: float
    created_at: datetime

    def __post_init__(self) -> None:
        require(self.vector.ndim == 1 and self.vector.size > 0, "an embedding is a 1-D vector")
        require(bool(np.isfinite(self.vector).all()), "embedding values must be finite")
        require(np.isfinite(self.norm) and self.norm > 0.0, "norm must be finite and positive")


@runtime_checkable
class EmbeddingCache(Protocol):
    def get_many(
        self, model_id: EmbeddingModelId, hashes: Sequence[TextHash]
    ) -> Mapping[TextHash, StoredEmbedding]:
        """Cached vectors for exact embedding-input hashes; unknown hashes are absent."""
        ...


@runtime_checkable
class EmbeddingCacheWriter(EmbeddingCache, Protocol):
    """Mutation side of the cache, available only inside a unit of work."""

    def put_many(self, model_id: EmbeddingModelId, items: Sequence[StoredEmbedding]) -> None:
        """Idempotent and content-addressed: re-putting a hash changes nothing."""
        ...

    def delete_unreferenced(self, model_id: EmbeddingModelId) -> int:
        """Drop vectors no chunk refers to; returns how many were removed."""
        ...
