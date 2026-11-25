"""Atomic publication of a prepared document version and its derived index data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from rag_engine.domain import (
    DocumentStatus,
    DocumentVersion,
    DocumentVersionId,
    EmbeddingModel,
    ErrorCode,
    IngestionJob,
    JobStatus,
    ModelStatus,
    RagError,
    SyncStatus,
)
from rag_engine.embedding.service import EmbeddingPlan
from rag_engine.ports import Clock, Store, StoredChunk


class IndexPublisher:
    """Publish only after parsing and embedding have completed outside a write transaction."""

    def __init__(self, store: Store, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    def publish(
        self,
        job: IngestionJob,
        version: DocumentVersion,
        chunks: Sequence[StoredChunk],
        embeddings: EmbeddingPlan,
        model: EmbeddingModel,
        expected_current: DocumentVersionId | None,
    ) -> bool:
        """Commit all data, the visible pointer, generation and job completion together.

        Returns ``False`` only when this exact job already committed this version. A stale
        lease, changed source or competing document version aborts the transaction.
        """
        now = self._clock.now()
        with self._store.unit_of_work().begin() as writer:
            current_job = writer.get_job(job.id)
            document = None if job.document_id is None else writer.get_document(job.document_id)
            if (
                current_job is not None
                and current_job.status is JobStatus.DONE
                and document is not None
                and document.current_version_id == version.id
            ):
                return False
            if (
                current_job is None
                or current_job.status is not JobStatus.LEASED
                or current_job.attempts != job.attempts
                or current_job.lease_until is None
                or current_job.lease_until <= now
            ):
                raise RagError(ErrorCode.E_TIMEOUT, "index publication lost its job lease")
            if (
                document is None
                or document.status is not DocumentStatus.ACTIVE
                or document.source_id != job.source_id
                or document.current_version_id != expected_current
                or version.document_id != document.id
            ):
                raise RagError(ErrorCode.E_INTEGRITY, "document changed during indexing")
            state = writer.get_sync_state(document.source_id, document.source_key)
            if (
                state is None
                or state.status is not SyncStatus.CHANGED
                or state.last_content_hash != version.content_hash
                or job.payload.get("content_hash") != version.content_hash.hex
            ):
                raise RagError(ErrorCode.E_INTEGRITY, "source changed during indexing")
            metadata = writer.get_index_metadata()
            if (
                metadata is None
                or model.status is not ModelStatus.ACTIVE
                or metadata.active_embedding_model_id not in (None, model.id)
            ):
                raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "active embedding model changed")
            if (
                writer.get_version(version.id) is not None
                or version.seq != len(writer.list_versions(document.id)) + 1
            ):
                raise RagError(ErrorCode.E_INTEGRITY, "version identity or sequence is stale")
            indexed = {item.chunk.id for item in chunks if item.chunk.indexed}
            if indexed != set(embeddings.by_chunk) or any(
                item.chunk.indexed
                and (
                    item.embedding_input_hash != embeddings.by_chunk[item.chunk.id]
                    or not item.body_an
                )
                for item in chunks
            ):
                raise RagError(ErrorCode.E_INTEGRITY, "chunks and embedding plan disagree")
            if len({item.chunk.id for item in chunks}) != len(chunks):
                raise RagError(ErrorCode.E_INTEGRITY, "duplicate chunks in publication")

            writer.put_embedding_model(model)
            writer.put_many(model.id, embeddings.new)
            keys = tuple(embeddings.by_chunk.values())
            if len(writer.get_many(model.id, keys)) != len(set(keys)):
                raise RagError(ErrorCode.E_INTEGRITY, "published chunks lack embeddings")
            writer.put_version(version)
            writer.put_chunks(version.id, chunks)
            writer.index_chunks(version.id)
            writer.put_document(replace(document, current_version_id=version.id, updated_at=now))
            writer.put_sync_state(replace(state, status=SyncStatus.OK, last_seen_at=now))
            writer.update_index_metadata(replace(metadata, active_embedding_model_id=model.id))
            writer.bump_generation()
            if not writer.complete_job(job.id, job.attempts, now):
                raise RagError(ErrorCode.E_TIMEOUT, "index publication lost its job lease")
            return True
