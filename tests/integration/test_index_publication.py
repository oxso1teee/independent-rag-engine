"""A publication either commits every index component or leaves the old state."""

from __future__ import annotations

from datetime import timedelta
from types import MappingProxyType

import numpy as np
import pytest

from rag_engine.domain import (
    EmbeddingModel,
    EmbeddingModelId,
    ErrorCode,
    IdFactory,
    JobKind,
    JobStatus,
    RagError,
    SyncStatus,
    TextHash,
)
from rag_engine.embedding.service import EmbeddingPlan
from rag_engine.ingestion.jobs import JobQueue
from rag_engine.ingestion.publishing import IndexPublisher
from rag_engine.ports import StoredChunk, StoredEmbedding
from rag_engine.storage.sqlite import SqliteStore
from rag_engine.testing import FakeClock
from tests.factories import (
    T0,
    make_chunk,
    make_document,
    make_source,
    make_sync_state,
    make_version,
)


def test_publication_is_atomic_and_fenced(
    store: SqliteStore, clock: FakeClock, ids: IdFactory
) -> None:
    source = make_source(ids)
    waiting = make_document(document_id=ids.document_id(), source_id=source.id, version_id=None)
    version = make_version(version_id=ids.document_version_id(), document_id=waiting.id)
    state = make_sync_state(
        source.id,
        waiting.source_key.value,
        status=SyncStatus.CHANGED,
        last_content_hash=version.content_hash,
    )
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
        writer.put_document(waiting)
        writer.put_sync_state(state)
    queue = JobQueue(store, clock, ids)
    queue.enqueue(
        JobKind.INGEST_DOC, source.id, waiting.id, {"content_hash": version.content_hash.hex}
    )
    job = queue.claim()
    assert job is not None
    model = EmbeddingModel(
        id=EmbeddingModelId.build(
            provider="mock",
            name="hash",
            revision="1",
            dimension=4,
            normalized=True,
            query_prefix="",
            document_prefix="",
        ),
        provider="mock",
        name="hash",
        revision="1",
        dimension=4,
        normalized=True,
        max_input_tokens=64,
        created_at=T0,
    )
    chunk = make_chunk(version_id=version.id)
    key = TextHash.of_text(chunk.text)
    stored = StoredChunk(chunk, key, chunk.text, "")
    vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    plan = EmbeddingPlan(
        MappingProxyType({chunk.id: key}), (StoredEmbedding(key, vector, 1.0, T0),)
    )
    publisher = IndexPublisher(store, clock)

    with pytest.raises(RagError) as missing:
        publisher.publish(job, version, [stored], EmbeddingPlan(plan.by_chunk, ()), model, None)
    assert missing.value.code is ErrorCode.E_INTEGRITY
    assert store.reader().get_version(version.id) is None
    assert store.reader().get_document(waiting.id) == waiting
    assert store.reader().get_index_metadata().index_generation == 0  # type: ignore[union-attr]

    assert publisher.publish(job, version, [stored], plan, model, None)
    assert not publisher.publish(job, version, [stored], plan, model, None)
    reader = store.reader()
    assert reader.get_document(waiting.id).current_version_id == version.id  # type: ignore[union-attr]
    assert reader.get_index_metadata().index_generation == 1  # type: ignore[union-attr]
    assert reader.get_job(job.id).status is JobStatus.DONE  # type: ignore[union-attr]
    assert reader.get_sync_state(source.id, waiting.source_key).status is SyncStatus.OK  # type: ignore[union-attr]


def test_expired_lease_cannot_publish(store: SqliteStore, clock: FakeClock, ids: IdFactory) -> None:
    source = make_source(ids)
    document = make_document(document_id=ids.document_id(), source_id=source.id, version_id=None)
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
        writer.put_document(document)
    queue = JobQueue(store, clock, ids)
    queue.enqueue(JobKind.INGEST_DOC, source.id, document.id)
    job = queue.claim()
    assert job is not None
    clock.advance(timedelta(seconds=301))
    version = make_version(version_id=ids.document_version_id(), document_id=document.id)
    with pytest.raises(RagError) as expired:
        IndexPublisher(store, clock).publish(
            job,
            version,
            [],
            EmbeddingPlan(MappingProxyType({}), ()),
            EmbeddingModel(
                id=EmbeddingModelId.build(
                    provider="mock",
                    name="hash",
                    revision="1",
                    dimension=4,
                    normalized=True,
                    query_prefix="",
                    document_prefix="",
                ),
                provider="mock",
                name="hash",
                revision="1",
                dimension=4,
                normalized=True,
                max_input_tokens=64,
                created_at=T0,
            ),
            None,
        )
    assert expired.value.code is ErrorCode.E_TIMEOUT
