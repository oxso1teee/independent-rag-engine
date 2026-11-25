"""Transactional retention, lexical rebuild and staged embedding model activation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from rag_engine.domain import DocumentStatus, EmbeddingModel, ModelStatus, SourceKey
from rag_engine.embedding.service import EmbeddingService
from rag_engine.lexical.analysis import ANALYZER_ID
from rag_engine.ports import Clock, Store


@dataclass(frozen=True, slots=True)
class RetentionResult:
    versions_purged: int
    embeddings_removed: int


class IndexMaintenance:
    """All visible changes use the store's single writer and unit of work."""

    def __init__(self, store: Store, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    def collect(self, *, keep_versions: int = 3) -> RetentionResult:
        if keep_versions < 1:
            raise ValueError("keep_versions must be positive")
        purged = removed = 0
        reader = self._store.reader()
        for source in reader.list_sources():
            after: SourceKey | None = None
            while batch := reader.list_documents(source.id, after=after, limit=100):
                for document in batch:
                    with self._store.unit_of_work().begin() as writer:
                        current = writer.get_document(document.id)
                        if current is None:
                            continue
                        versions = writer.list_versions(document.id)
                        candidates = versions[:-keep_versions]
                        changed = 0
                        for version in candidates:
                            if version.id != current.current_version_id:
                                changed += int(writer.purge_version_data(version.id))
                        purged += changed
                after = batch[-1].source_key
        with self._store.unit_of_work().begin() as writer:
            removed = sum(
                writer.delete_unreferenced(model.id) for model in writer.list_embedding_models()
            )
            metadata = writer.get_index_metadata()
            if metadata is not None and (purged or removed):
                writer.update_index_metadata(replace(metadata, last_gc_at=self._clock.now()))
        return RetentionResult(purged, removed)

    def rebuild_lexical(self, *, force: bool = False) -> bool:
        with self._store.unit_of_work().begin() as writer:
            metadata = writer.get_index_metadata()
            if metadata is None:
                return False
            if metadata.lexical_analyzer_id == ANALYZER_ID and not force:
                return False
            writer.rebuild_lexical_index()
            writer.update_index_metadata(replace(metadata, lexical_analyzer_id=ANALYZER_ID))
            writer.bump_generation()
            return True

    def switch_model(self, model: EmbeddingModel, service: EmbeddingService) -> bool:
        """Stage missing vectors outside the switch transaction, then verify and flip once.

        Interrupted staging is harmless: the next run reuses its content-addressed cache.
        A changed document makes the final verification fail without changing the active model.
        """
        metadata = self._store.reader().get_index_metadata()
        if metadata is not None and metadata.active_embedding_model_id == model.id:
            return False
        with self._store.unit_of_work().begin() as writer:
            writer.put_embedding_model(replace(model, status=ModelStatus.RETIRED))
        reader = self._store.reader()
        for source in reader.list_sources():
            after: SourceKey | None = None
            while batch := reader.list_documents(
                source.id, status=DocumentStatus.ACTIVE, after=after, limit=100
            ):
                for document in batch:
                    if document.duplicate_of is not None or document.current_version_id is None:
                        continue
                    chunks = [
                        stored.chunk
                        for stored in reader.list_chunks(document.current_version_id)
                        if stored.chunk.indexed
                    ]
                    plan = asyncio.run(service.prepare(chunks, reader))
                    if plan.new:
                        with self._store.unit_of_work().begin() as writer:
                            writer.put_many(model.id, plan.new)
                after = batch[-1].source_key
        with self._store.unit_of_work().begin() as writer:
            return writer.switch_active_embedding_model(model.id)
