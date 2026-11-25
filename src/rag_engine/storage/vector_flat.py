"""Bounded exact cosine search over FP16 chunk embeddings."""

from __future__ import annotations

import heapq
import sqlite3
from collections.abc import Sequence, Set
from pathlib import Path

import numpy as np

from rag_engine.domain import ChunkId, EmbeddingModelId, ErrorCode, RagError, ScoredChunk
from rag_engine.ports.embedding import Vectors
from rag_engine.storage.sqlite.connection import connect


class FlatVectorIndex:
    """FP16 resident matrix; FP32 block scoring and deterministic top-k.

    ``refresh`` swaps a complete SQLite snapshot only when its generation changes.
    The old matrix stays usable if a refresh fails.
    """

    def __init__(
        self, model_id: EmbeddingModelId, dimension: int, *, max_memory_mb: int = 512
    ) -> None:
        if dimension < 1 or max_memory_mb < 1 or model_id.dimension != dimension:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "invalid flat vector index limits")
        self._model_id = model_id
        self._dimension = dimension
        self._limit = max_memory_mb * 1024 * 1024
        self._ids: list[ChunkId] = []
        self._positions: dict[ChunkId, int] = {}
        self._vectors = np.empty((0, dimension), dtype=np.float16)
        self._generation: int | None = None

    @property
    def model_id(self) -> EmbeddingModelId:
        return self._model_id

    @property
    def generation(self) -> int | None:
        return self._generation

    def __len__(self) -> int:
        return len(self._ids)

    def _check_capacity(self, rows: int) -> None:
        # Include the resident matrix, score array and at least one FP32 scoring row.
        if rows * (self._dimension * 2 + 4) + self._dimension * 4 > self._limit:
            raise RagError(ErrorCode.E_RESOURCE_GATE, "flat vector index exceeds memory budget")

    def upsert(self, ids: Sequence[ChunkId], vectors: Vectors) -> None:
        if vectors.shape != (len(ids), self._dimension) or not bool(np.isfinite(vectors).all()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "invalid vectors for flat index")
        if len(set(ids)) != len(ids):
            raise RagError(ErrorCode.E_INTEGRITY, "duplicate chunk ids in vector upsert")
        new_ids = [chunk_id for chunk_id in ids if chunk_id not in self._positions]
        self._check_capacity(len(self) + len(new_ids))
        if bool((np.abs(vectors) > np.finfo(np.float16).max).any()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "vector overflows FP16 storage")
        compact = np.asarray(vectors, dtype=np.float16)
        if not bool(np.isfinite(compact).all()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "vector overflows FP16 storage")
        if new_ids:
            self._vectors = np.concatenate(
                (self._vectors, np.empty((len(new_ids), self._dimension), dtype=np.float16))
            )
            for chunk_id in new_ids:
                self._positions[chunk_id] = len(self._ids)
                self._ids.append(chunk_id)
        for chunk_id, vector in zip(ids, compact, strict=True):
            self._vectors[self._positions[chunk_id]] = vector
        self._generation = None

    def remove(self, ids: Sequence[ChunkId]) -> None:
        removed = set(ids)
        if not removed:
            return
        kept = [i for i, chunk_id in enumerate(self._ids) if chunk_id not in removed]
        self._ids = [self._ids[i] for i in kept]
        self._vectors = self._vectors[kept].copy()
        self._positions = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        self._generation = None

    def search(
        self, query: Vectors, k: int, allowed: Set[ChunkId] | None = None
    ) -> Sequence[ScoredChunk]:
        if k <= 0:
            return []
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape == (1, self._dimension):
            vector = vector[0]
        if vector.shape != (self._dimension,) or not bool(np.isfinite(vector).all()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "invalid flat index query")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "zero flat index query")
        vector = vector / norm
        scores = np.empty(len(self), dtype=np.float32)
        # A bounded FP32 block is the only scoring allocation; the resident matrix stays FP16.
        spare = self._limit - self._vectors.nbytes - scores.nbytes
        block_rows = max(1, min(4096, spare // (self._dimension * 4)))
        for start in range(0, len(self), block_rows):
            end = min(start + block_rows, len(self))
            scores[start:end] = np.einsum(
                "ij,j->i", self._vectors[start:end].astype(np.float32), vector, optimize=False
            )
        candidates = (
            range(len(self))
            if allowed is None
            else (self._positions[chunk_id] for chunk_id in allowed if chunk_id in self._positions)
        )
        top = heapq.nsmallest(k, candidates, key=lambda i: (-float(scores[i]), str(self._ids[i])))
        return [ScoredChunk(self._ids[i], float(scores[i])) for i in top]

    def refresh(self, database: Path) -> bool:
        """Load the active model's visible chunks from one read transaction.

        Returns ``False`` on a cache hit. A model switch invalidates this index and requires a
        new instance with the new model identity.
        """
        conn = connect(database, busy_timeout_ms=5000, read_only=True)
        try:
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT index_generation, active_embedding_model_id FROM index_state WHERE id = 1"
            ).fetchone()
            if row is None or row["active_embedding_model_id"] != str(self.model_id):
                raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "active vector model changed")
            generation = int(row["index_generation"])
            if self._generation == generation:
                return False
            model = conn.execute(
                "SELECT dimension FROM embedding_models WHERE id = ? AND status = 'active'",
                (str(self.model_id),),
            ).fetchone()
            if model is None or model["dimension"] != self._dimension:
                raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "active model metadata differs")
            query = (
                " FROM chunks AS c"
                " JOIN document_versions AS v ON v.id = c.document_version_id"
                " JOIN documents AS d ON d.id = v.document_id"
                " JOIN embeddings AS e ON e.embedding_input_hash = c.embedding_input_hash"
                " AND e.model_id = ?"
                " WHERE d.status = 'active' AND d.current_version_id = v.id"
                " AND c.indexed = 1 AND c.is_duplicate = 0"
            )
            count_row = conn.execute("SELECT count(*)" + query, (str(self.model_id),)).fetchone()
            count = int(count_row[0])
            self._check_capacity(count)
            next_ids: list[ChunkId] = []
            next_vectors = np.empty((count, self._dimension), dtype=np.float16)
            cursor = conn.execute(
                "SELECT c.id, e.vector, e.dtype" + query + " ORDER BY c.id",
                (str(self.model_id),),
            )
            offset = 0
            while rows := cursor.fetchmany(256):
                for item in rows:
                    blob = item["vector"]
                    if item["dtype"] != "f16le" or len(blob) != self._dimension * 2:
                        raise RagError(ErrorCode.E_INTEGRITY, "stored vector has an invalid shape")
                    vector = np.frombuffer(blob, dtype="<f2")
                    if not bool(np.isfinite(vector).all()):
                        raise RagError(ErrorCode.E_INTEGRITY, "stored vector is non-finite")
                    next_ids.append(ChunkId(item["id"]))
                    next_vectors[offset] = vector
                    offset += 1
            if offset != count:
                raise RagError(ErrorCode.E_INTEGRITY, "vector snapshot changed while loading")
            self._ids = next_ids
            self._positions = {chunk_id: i for i, chunk_id in enumerate(next_ids)}
            self._vectors = next_vectors
            self._generation = generation
            return True
        except sqlite3.Error as exc:
            raise RagError(ErrorCode.E_INTEGRITY, "flat index could not read SQLite") from exc
        finally:
            conn.close()
