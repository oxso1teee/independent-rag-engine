"""Store port contract: what a MetadataReader/Writer promises, checked on SQLite."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from rag_engine.domain import (
    ContentHash,
    Document,
    DocumentStatus,
    DocumentVersion,
    ErrorCode,
    IdFactory,
    JobKind,
    JobStatus,
    RagError,
    Source,
    SourceKey,
    SyncStatus,
    TextHash,
)
from rag_engine.ports import StoredEmbedding
from rag_engine.storage.sqlite import SqliteStore
from tests.factories import (
    FINGERPRINT,
    T0,
    make_document,
    make_job,
    make_source,
    make_sync_state,
    make_version,
)


def seed(store: SqliteStore, ids: IdFactory) -> tuple[Source, Document, DocumentVersion]:
    source = make_source(ids)
    version_id = ids.document_version_id()
    document = make_document(
        document_id=ids.document_id(), source_id=source.id, version_id=version_id
    )
    version = make_version(version_id=version_id, document_id=document.id)
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
        writer.put_document(document)
        writer.put_version(version)
    return source, document, version


def test_documents_can_be_listed_filtered_counted_and_paged(
    store: SqliteStore, ids: IdFactory
) -> None:
    source = make_source(ids)
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
        for i in range(5):
            status = DocumentStatus.PENDING if i % 2 else DocumentStatus.FAILED
            writer.put_document(
                make_document(
                    document_id=ids.document_id(),
                    source_id=source.id,
                    version_id=None,
                    status=status,
                    key=f"k{i}.md",
                )
            )
    reader = store.reader()
    assert reader.count_documents(source.id) == 5
    assert reader.count_documents(source.id, DocumentStatus.PENDING) == 2
    page = reader.list_documents(source.id, limit=2)
    assert [d.source_key.value for d in page] == ["k0.md", "k1.md"]
    nxt = reader.list_documents(source.id, after=page[-1].source_key, limit=2)
    assert [d.source_key.value for d in nxt] == ["k2.md", "k3.md"]
    failed = reader.list_documents(source.id, DocumentStatus.FAILED)
    assert [d.source_key.value for d in failed] == ["k0.md", "k2.md", "k4.md"]
    assert reader.get_document_by_key(source.id, SourceKey("k3.md")) is not None
    assert reader.get_document_by_key(source.id, SourceKey("nope.md")) is None


def test_versions_are_immutable(store: SqliteStore, ids: IdFactory) -> None:
    _, document, version = seed(store, ids)
    reader = store.reader()
    assert reader.get_version(version.id) == version
    assert reader.list_versions(document.id) == [version]
    with store.unit_of_work().begin() as writer:
        writer.put_version(version)  # identical rewrite is a no-op
        second = replace(
            version,
            id=ids.document_version_id(),
            seq=2,
            content_hash=ContentHash.of_bytes(b"v2"),
            chunker_fingerprint=FINGERPRINT,
        )
        writer.put_version(second)
    assert [v.seq for v in reader.list_versions(document.id)] == [1, 2]

    def rewrite() -> None:
        with store.unit_of_work().begin() as writer:
            writer.put_version(replace(version, size_bytes=version.size_bytes + 1))

    with pytest.raises(RagError) as info:
        rewrite()
    assert info.value.code is ErrorCode.E_INTEGRITY
    assert reader.get_version(version.id) == version


def test_text_purge_is_recorded(store: SqliteStore, ids: IdFactory) -> None:
    _, _, version = seed(store, ids)
    with store.unit_of_work().begin() as writer:
        writer.mark_text_purged(version.id)
    purged = store.reader().get_version(version.id)
    assert isinstance(purged, DocumentVersion) and purged.text_purged

    def unknown() -> None:
        with store.unit_of_work().begin() as writer:
            writer.mark_text_purged(ids.document_version_id())

    with pytest.raises(RagError):
        unknown()


def test_deleting_a_document_removes_its_dependents(store: SqliteStore, ids: IdFactory) -> None:
    source, document, version = seed(store, ids)
    with store.unit_of_work().begin() as writer:
        writer.put_sync_state(make_sync_state(source.id, document.source_key.value))
        writer.enqueue_job(make_job(ids, source.id, document_id=document.id))
        writer.delete_document(document.id)
    reader = store.reader()
    assert reader.get_document(document.id) is None
    assert reader.get_version(version.id) is None
    assert reader.list_jobs() == []


def test_a_source_with_documents_needs_force_to_be_deleted(
    store: SqliteStore, ids: IdFactory
) -> None:
    source, document, _ = seed(store, ids)

    def delete(force: bool) -> None:
        with store.unit_of_work().begin() as writer:
            writer.delete_source(source.id, force=force)

    with pytest.raises(RagError) as info:
        delete(False)
    assert info.value.code is ErrorCode.E_INTEGRITY
    assert store.reader().get_source(source.id) is not None
    delete(True)
    assert store.reader().get_source(source.id) is None
    assert store.reader().get_document(document.id) is None


def test_sync_states_round_trip(store: SqliteStore, ids: IdFactory) -> None:
    source, _, _ = seed(store, ids)
    state = make_sync_state(
        source.id,
        "a.md",
        status=SyncStatus.FAILED,
        last_error_code=ErrorCode.E_UNSUPPORTED_FORMAT,
        attempts=3,
        next_retry_at=T0 + timedelta(minutes=5),
    )
    with store.unit_of_work().begin() as writer:
        writer.put_sync_state(state)
        writer.put_sync_state(make_sync_state(source.id, "b.md"))
    reader = store.reader()
    assert reader.get_sync_state(source.id, SourceKey("a.md")) == state
    assert [s.source_key.value for s in reader.list_sync_states(source.id)] == ["a.md", "b.md"]
    with store.unit_of_work().begin() as writer:
        writer.delete_sync_state(source.id, SourceKey("a.md"))
    assert reader.get_sync_state(source.id, SourceKey("a.md")) is None


def test_job_filters_and_counts(store: SqliteStore, ids: IdFactory) -> None:
    source, document, _ = seed(store, ids)
    with store.unit_of_work().begin() as writer:
        writer.enqueue_job(make_job(ids, source.id, document_id=document.id))
        writer.enqueue_job(make_job(ids, source.id, kind=JobKind.GC))
    reader = store.reader()
    assert len(reader.list_jobs()) == 2
    assert [j.kind for j in reader.list_jobs(kind=JobKind.GC)] == [JobKind.GC]
    assert len(reader.list_jobs(JobStatus.QUEUED, limit=1)) == 1
    assert reader.list_jobs(JobStatus.DONE) == []
    assert reader.count_jobs() == {JobStatus.QUEUED: 2}
    job = reader.list_jobs()[0]
    assert reader.get_job(job.id) == job
    assert reader.get_job(ids.job_id()) is None


def test_index_generation_is_monotonic(store: SqliteStore) -> None:
    metadata = store.reader().get_index_metadata()
    assert metadata is not None and metadata.index_generation == 0
    with store.unit_of_work().begin() as writer:
        assert writer.bump_generation() == 1
        assert writer.bump_generation() == 2

        def regress() -> None:
            current = writer.get_index_metadata()
            assert current is not None
            writer.update_index_metadata(replace(current, index_generation=1))

        with pytest.raises(RagError):
            regress()
    after = store.reader().get_index_metadata()
    assert after is not None and after.index_generation == 2


def test_stored_embedding_validates_its_shape() -> None:
    vector = np.ones(4, dtype=np.float32)
    ok = StoredEmbedding(
        text_hash=TextHash.of_text("x"),
        vector=vector,
        norm=2.0,
        created_at=T0,
    )
    assert ok.vector.shape == (4,)
    with pytest.raises(RagError):
        replace(ok, vector=np.ones((2, 2), dtype=np.float32))
    with pytest.raises(RagError):
        replace(ok, norm=-1.0)


def test_sql_metacharacters_are_plain_data(store: SqliteStore, ids: IdFactory) -> None:
    hostile = "x'; DROP TABLE sources; --"
    source = make_source(ids, name="hostile", locator=hostile)
    document = make_document(
        document_id=ids.document_id(),
        source_id=source.id,
        version_id=None,
        status=DocumentStatus.PENDING,
        key=f'we"ird/{hostile}.md',
    )
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
        writer.put_document(document)
    reader = store.reader()
    assert reader.get_source_by_name("hostile") == source
    assert reader.get_document_by_key(source.id, document.source_key) == document
    assert reader.get_source_by_name("x' OR '1'='1") is None
    assert len(reader.list_sources()) == 1
