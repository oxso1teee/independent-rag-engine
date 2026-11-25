"""Read, parse, chunk and embed a leased document before an atomic publication."""

from __future__ import annotations

import asyncio
import zlib
from collections.abc import Callable
from dataclasses import replace

from rag_engine.domain import (
    ChunkPolicy,
    ContentHash,
    DocumentStatus,
    DocumentVersion,
    EmbeddingModel,
    ErrorCode,
    IdFactory,
    IngestionJob,
    RagError,
)
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.identity import AdapterFactory
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.ingestion.publishing import IndexPublisher
from rag_engine.lexical.analysis import analyze
from rag_engine.ports import Clock, ParseLimits, Store, StoredChunk


class DocumentIndexingHandler:
    """One job handler; no write lock is held during source or provider I/O."""

    def __init__(
        self,
        store: Store,
        clock: Clock,
        ids: IdFactory,
        adapter_factory: AdapterFactory,
        processor_for: Callable[[tuple[str, ...]], DocumentProcessor],
        embeddings: EmbeddingService,
        model: EmbeddingModel,
        limits: ParseLimits,
        policy: ChunkPolicy,
    ) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids
        self._adapter_factory = adapter_factory
        self._processor_for = processor_for
        self._embeddings = embeddings
        self._model = model
        self._limits = limits
        self._policy = policy
        self._publisher = IndexPublisher(store, clock)

    def __call__(self, job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
        if job.document_id is None:
            raise RagError(ErrorCode.E_INTEGRITY, "indexing job has no document")
        reader = self._store.reader()
        document = reader.get_document(job.document_id)
        source = reader.get_source(job.source_id)
        if document is None or source is None or document.status is not DocumentStatus.ACTIVE:
            raise RagError(ErrorCode.E_INTEGRITY, "indexing source or document changed")
        state = reader.get_sync_state(source.id, document.source_key)
        if state is None or state.last_revision is None or state.last_content_hash is None:
            raise RagError(ErrorCode.E_INTEGRITY, "indexing source has no observed revision")
        adapter = self._adapter_factory(source)
        if adapter.identity().locator_fingerprint != source.locator_fingerprint:
            raise RagError(ErrorCode.E_SOURCE_LOCATOR_CHANGED, "source locator changed")
        processor = self._processor_for(source.policy.cleaners)
        with adapter.open(document.source_key) as stream:
            data = processor.read(
                stream,
                replace(
                    self._limits,
                    max_input_bytes=min(self._limits.max_input_bytes, source.policy.max_file_bytes),
                ),
            )
        if ContentHash.of_bytes(data) != state.last_content_hash:
            raise RagError(ErrorCode.E_INTEGRITY, "source content changed after discovery")
        if not heartbeat():
            raise RagError(ErrorCode.E_TIMEOUT, "indexing job lease expired")
        version_id = self._ids.document_version_id()
        processed = processor.process(
            data,
            document.source_key,
            self._limits,
            self._policy,
            version_id,
        )
        version = DocumentVersion(
            id=version_id,
            document_id=document.id,
            seq=len(reader.list_versions(document.id)) + 1,
            content_hash=processed.parsed.content_hash,
            size_bytes=processed.parsed.size_bytes,
            mtime_ns=state.last_revision.mtime_ns,
            mime=processed.parsed.mime.mime,
            detected_by=processed.parsed.mime.detected_by,
            parser_id=processed.parsed.parser_id,
            parser_version=processed.parsed.parser_version,
            normalizer_version=processed.parsed.normalizer_version,
            chunker_fingerprint=processed.chunker_fingerprint,
            parse_warnings=processed.parsed.document.warnings,
            normalized_text=zlib.compress(processed.parsed.document.normalized_text.encode()),
            created_at=self._clock.now(),
        )
        plan = asyncio.run(self._embeddings.prepare(processed.chunks, reader))
        if not heartbeat():
            raise RagError(ErrorCode.E_TIMEOUT, "indexing job lease expired")
        stored = [
            StoredChunk(
                chunk,
                plan.by_chunk.get(chunk.id),
                analyze(chunk.text, chunk.language) if chunk.indexed else "",
                analyze(" ".join(chunk.heading_path), chunk.language) if chunk.indexed else "",
            )
            for chunk in processed.chunks
        ]
        self._publisher.publish(
            job, version, stored, plan, self._model, document.current_version_id
        )
