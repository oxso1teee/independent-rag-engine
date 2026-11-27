"""Every fencing guard of the index publisher refuses and leaves the old state untouched."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pytest

from rag_engine.domain import (
    DocumentId,
    DocumentStatus,
    DocumentVersion,
    EmbeddingModel,
    EmbeddingModelId,
    ErrorCode,
    IdFactory,
    IngestionJob,
    JobKind,
    ModelStatus,
    RagError,
    SourceId,
    SourceKey,
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


def build_model(revision: str = "1") -> EmbeddingModel:
    return EmbeddingModel(
        id=EmbeddingModelId.build(
            provider="mock",
            name="hash",
            revision=revision,
            dimension=4,
            normalized=True,
            query_prefix="",
            document_prefix="",
        ),
        provider="mock",
        name="hash",
        revision=revision,
        dimension=4,
        normalized=True,
        max_input_tokens=64,
        created_at=T0,
    )


@dataclass
class Scene:
    store: SqliteStore
    clock: FakeClock
    job: IngestionJob
    document_id: DocumentId
    source_id: SourceId
    source_key: SourceKey
    version: DocumentVersion
    chunks: list[StoredChunk]
    plan: EmbeddingPlan
    model: EmbeddingModel


def build_scene(
    store: SqliteStore, clock: FakeClock, ids: IdFactory, *, payload_hash: str | None
) -> Scene:
    source = make_source(ids)
    document = make_document(document_id=ids.document_id(), source_id=source.id, version_id=None)
    version = make_version(version_id=ids.document_version_id(), document_id=document.id)
    state = make_sync_state(
        source.id,
        document.source_key.value,
        status=SyncStatus.CHANGED,
        last_content_hash=version.content_hash,
    )
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
        writer.put_document(document)
        writer.put_sync_state(state)
    queue = JobQueue(store, clock, ids)
    queue.enqueue(
        JobKind.INGEST_DOC,
        source.id,
        document.id,
        {"content_hash": payload_hash or version.content_hash.hex},
    )
    job = queue.claim()
    assert job is not None
    chunk = make_chunk(version_id=version.id)
    key = TextHash.of_text(chunk.text)
    plan = EmbeddingPlan(
        MappingProxyType({chunk.id: key}),
        (StoredEmbedding(key, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 1.0, T0),),
    )
    return Scene(
        store,
        clock,
        job,
        document.id,
        source.id,
        document.source_key,
        version,
        [StoredChunk(chunk, key, chunk.text, "")],
        plan,
        build_model(),
    )


def document_status(scene: Scene, status: DocumentStatus) -> None:
    with scene.store.unit_of_work().begin() as writer:
        document = writer.get_document(scene.document_id)
        assert document is not None
        writer.put_document(replace(document, status=status))


def sync_status(scene: Scene, status: SyncStatus) -> None:
    with scene.store.unit_of_work().begin() as writer:
        state = writer.get_sync_state(scene.source_id, scene.source_key)
        assert state is not None
        writer.put_sync_state(replace(state, status=status))


def other_active_model(scene: Scene) -> None:
    other = build_model("2")
    with scene.store.unit_of_work().begin() as writer:
        writer.put_embedding_model(other)
        metadata = writer.get_index_metadata()
        assert metadata is not None
        writer.update_index_metadata(replace(metadata, active_embedding_model_id=other.id))


def retire_model(scene: Scene) -> None:
    scene.model = replace(scene.model, status=ModelStatus.RETIRED)


def foreign_embedding_key(scene: Scene) -> None:
    chunk = scene.chunks[0].chunk
    wrong = TextHash.of_text("something else")
    scene.plan = EmbeddingPlan(MappingProxyType({chunk.id: wrong}), scene.plan.new)


def duplicate_chunk(scene: Scene) -> None:
    scene.chunks = [scene.chunks[0], scene.chunks[0]]


def stale_sequence(scene: Scene) -> None:
    scene.version = replace(scene.version, seq=2)


def attempt_changed(scene: Scene) -> None:
    scene.job = replace(scene.job, attempts=scene.job.attempts + 1)


def missing_body(scene: Scene) -> None:
    scene.chunks = [replace(scene.chunks[0], body_an="")]


CASES: dict[str, tuple[Callable[[Scene], None], ErrorCode, str | None]] = {
    "document no longer active": (
        lambda s: document_status(s, DocumentStatus.DELETED),
        ErrorCode.E_INTEGRITY,
        None,
    ),
    "source state already published": (
        lambda s: sync_status(s, SyncStatus.OK),
        ErrorCode.E_INTEGRITY,
        None,
    ),
    "payload hash differs": (lambda s: None, ErrorCode.E_INTEGRITY, "different"),
    "another model is active": (other_active_model, ErrorCode.E_EMBED_MODEL_MISMATCH, None),
    "model retired": (retire_model, ErrorCode.E_EMBED_MODEL_MISMATCH, None),
    "version sequence stale": (stale_sequence, ErrorCode.E_INTEGRITY, None),
    "embedding plan disagrees": (foreign_embedding_key, ErrorCode.E_INTEGRITY, None),
    "indexed chunk without analysed body": (missing_body, ErrorCode.E_INTEGRITY, None),
    "duplicate chunk": (duplicate_chunk, ErrorCode.E_INTEGRITY, None),
    "job attempt changed": (attempt_changed, ErrorCode.E_TIMEOUT, None),
}


@pytest.mark.parametrize(("mutate", "code", "payload_hash"), CASES.values(), ids=CASES.keys())
def test_a_failed_guard_publishes_nothing(
    store: SqliteStore,
    clock: FakeClock,
    ids: IdFactory,
    mutate: Callable[[Scene], None],
    code: ErrorCode,
    payload_hash: str | None,
) -> None:
    scene = build_scene(store, clock, ids, payload_hash=("f" * 64) if payload_hash else None)
    mutate(scene)
    with pytest.raises(RagError) as info:
        IndexPublisher(store, clock).publish(
            scene.job,
            scene.version,
            scene.chunks,
            scene.plan,
            scene.model,
            None,
        )
    assert info.value.code is code
    reader = store.reader()
    assert reader.get_version(scene.version.id) is None
    metadata = reader.get_index_metadata()
    assert metadata is not None and metadata.index_generation == 0


def test_a_competing_current_version_blocks_publication(
    store: SqliteStore, clock: FakeClock, ids: IdFactory
) -> None:
    scene = build_scene(store, clock, ids, payload_hash=None)
    with pytest.raises(RagError) as info:
        IndexPublisher(store, clock).publish(
            scene.job,
            scene.version,
            scene.chunks,
            scene.plan,
            scene.model,
            ids.document_version_id(),  # the caller expected another current version
        )
    assert info.value.code is ErrorCode.E_INTEGRITY
    assert store.reader().get_version(scene.version.id) is None
