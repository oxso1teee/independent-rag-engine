"""Data-model invariants (DATA_MODEL §5) that concern ingestion, as ``check_*`` functions.

They read a store through its public reader and raise ``AssertionError`` naming the broken
invariant. Integration tests run them at teardown; the model-based test runs them after
every step.
"""

from __future__ import annotations

from rag_engine.domain import DocumentStatus, JobStatus, SyncStatus
from rag_engine.domain.invariants import (
    check_current_version,
    check_duplicate_graph,
    check_sync_state_matches_document,
)
from rag_engine.ports import MetadataReader
from rag_engine.storage.migrations import load_packaged_migrations


def check_no_orphans(reader: MetadataReader) -> None:
    """I7: nothing refers to something that does not exist."""
    sources = {s.id for s in reader.list_sources()}
    for source_id in sources:
        documents = {d.id: d for d in reader.list_documents(source_id)}
        for document in documents.values():
            for version in reader.list_versions(document.id):
                assert version.document_id == document.id, "I7: version of another document"
        for state in reader.list_sync_states(source_id):
            assert state.source_id in sources, "I7: sync state without a source"
        for job in reader.list_jobs():
            assert job.source_id in sources, "I7: job without a source"
            assert job.document_id is None or reader.get_document(job.document_id) is not None, (
                "I7: job without its document"
            )


def check_current_versions(reader: MetadataReader) -> None:
    """A current version exists and belongs to its document; unversioned documents are not
    searchable; a searchable document is active (I5)."""
    for source in reader.list_sources():
        for document in reader.list_documents(source.id):
            if document.current_version_id is None:
                assert not document.is_searchable, "I5: a document without a version is searchable"
                continue
            version = reader.get_version(document.current_version_id)
            assert version is not None, "the current version does not exist"
            check_current_version(document, version)
            assert document.is_searchable == (document.status is DocumentStatus.ACTIVE)


def check_sync_consistency(reader: MetadataReader) -> None:
    """Every document has a sync state that agrees with it; no sync state is left behind."""
    for source in reader.list_sources():
        documents = {d.source_key: d for d in reader.list_documents(source.id)}
        states = {s.source_key: s for s in reader.list_sync_states(source.id)}
        assert set(states) == set(documents), "sync states and documents cover different keys"
        for key, document in documents.items():
            check_sync_state_matches_document(document, states[key])
            assert (states[key].status is SyncStatus.MISSING) == (
                document.status is DocumentStatus.MISSING
            )


def check_duplicates(reader: MetadataReader) -> None:
    """``duplicate_of`` follows the content exactly: among the active documents of one raw hash
    the smallest id is the original, the others point to it, nothing else has a link."""
    for source in reader.list_sources():
        documents = list(reader.list_documents(source.id))
        by_key = {s.source_key: s.last_content_hash for s in reader.list_sync_states(source.id)}
        hashes = {d.id: by_key[d.source_key] for d in documents if by_key.get(d.source_key)}
        check_duplicate_graph(documents, hashes)  # type: ignore[arg-type]
        live = [d for d in documents if d.status is DocumentStatus.ACTIVE and d.id in hashes]
        for document in live:
            group = [d.id for d in live if hashes[d.id] == hashes[document.id]]
            original = min(group)
            expected = None if document.id == original else original
            assert document.duplicate_of == expected, "duplicate_of does not follow the content"
        for document in documents:
            if document not in live:
                assert document.duplicate_of is None, "a document that is not active has a link"


def check_jobs(reader: MetadataReader) -> None:
    """At most one active job per document; finished jobs carry no lease."""
    seen: set[tuple[str, str, str]] = set()
    for job in reader.list_jobs():
        if job.status.is_active:
            key = (job.kind.value, str(job.source_id), str(job.document_id))
            assert key not in seen, "two active jobs for the same document"
            seen.add(key)
        if job.status in (JobStatus.DONE, JobStatus.DEAD):
            assert job.lease_until is None


def check_index_state(reader: MetadataReader) -> None:
    """I8: the generation is non-negative and the schema version mirrors the migrations."""
    metadata = reader.get_index_metadata()
    assert metadata is not None
    assert metadata.index_generation >= 0
    assert metadata.schema_version == len(load_packaged_migrations())


def check_store_invariants(reader: MetadataReader) -> None:
    check_no_orphans(reader)
    check_current_versions(reader)
    check_sync_consistency(reader)
    check_duplicates(reader)
    check_jobs(reader)
    check_index_state(reader)
