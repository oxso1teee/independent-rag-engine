"""Metadata store ports (system of record).

``MetadataReader`` can only read and never takes the writer lock. All changes go through
a ``MetadataWriter`` that exists only inside a ``UnitOfWork``: one atomic transaction under
the single writer lock. The store never reads the system clock; callers pass timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable

from rag_engine.domain import (
    Document,
    DocumentId,
    DocumentStatus,
    DocumentVersion,
    DocumentVersionId,
    ErrorCode,
    IndexMetadata,
    IngestionJob,
    JobId,
    JobKind,
    JobStatus,
    Sha256Hex,
    Source,
    SourceId,
    SourceKey,
    SyncState,
)


@runtime_checkable
class MetadataReader(Protocol):
    """Read-only access; safe to use concurrently with a writer."""

    def get_source(self, source_id: SourceId) -> Source | None: ...

    def get_source_by_name(self, name: str) -> Source | None: ...

    def list_sources(self) -> Sequence[Source]:
        """All sources ordered by name."""
        ...

    def get_document(self, document_id: DocumentId) -> Document | None: ...

    def get_document_by_key(self, source_id: SourceId, key: SourceKey) -> Document | None: ...

    def list_documents(
        self,
        source_id: SourceId,
        status: DocumentStatus | None = None,
        after: SourceKey | None = None,
        limit: int | None = None,
    ) -> Sequence[Document]:
        """Documents of a source ordered by key; ``after`` is an exclusive keyset cursor."""
        ...

    def count_documents(self, source_id: SourceId, status: DocumentStatus | None = None) -> int: ...

    def get_version(self, version_id: DocumentVersionId) -> DocumentVersion | None: ...

    def list_versions(self, document_id: DocumentId) -> Sequence[DocumentVersion]:
        """Versions ordered by ``seq``."""
        ...

    def get_sync_state(self, source_id: SourceId, key: SourceKey) -> SyncState | None: ...

    def list_sync_states(self, source_id: SourceId) -> Sequence[SyncState]:
        """Sync states of a source ordered by key."""
        ...

    def get_job(self, job_id: JobId) -> IngestionJob | None: ...

    def list_jobs(
        self,
        status: JobStatus | None = None,
        kind: JobKind | None = None,
        limit: int | None = None,
    ) -> Sequence[IngestionJob]:
        """Jobs ordered by creation (job id)."""
        ...

    def count_jobs(self) -> dict[JobStatus, int]:
        """Number of jobs per status (statuses without jobs are absent)."""
        ...

    def get_index_metadata(self) -> IndexMetadata | None: ...


@runtime_checkable
class MetadataWriter(MetadataReader, Protocol):
    """Mutations; obtainable only from :meth:`UnitOfWork.begin`."""

    # sources ---------------------------------------------------------------------------
    def put_source(self, source: Source) -> None:
        """Insert or update. Changing the locator of an existing source is refused with
        ``E_SOURCE_LOCATOR_CHANGED``; use :meth:`rebase_source`."""
        ...

    def rebase_source(
        self, source_id: SourceId, new_locator: str, new_fingerprint: Sha256Hex, now: datetime
    ) -> Source:
        """Explicitly move a source to a new locator and return the updated source."""
        ...

    def delete_source(self, source_id: SourceId, *, force: bool = False) -> None:
        """Delete a source; refused while it has documents unless ``force`` is set."""
        ...

    # documents and versions ------------------------------------------------------------
    def put_document(self, document: Document) -> None:
        """Upsert by id; the (source, key) pair stays unique."""
        ...

    def delete_document(self, document_id: DocumentId) -> None:
        """Physically remove a document with its versions, sync state and jobs."""
        ...

    def put_version(self, version: DocumentVersion) -> None:
        """Versions are immutable: rewriting an id with different content is ``E_INTEGRITY``."""
        ...

    def mark_text_purged(self, version_id: DocumentVersionId) -> None: ...

    # sync state ------------------------------------------------------------------------
    def put_sync_state(self, state: SyncState) -> None: ...

    def delete_sync_state(self, source_id: SourceId, key: SourceKey) -> None: ...

    # jobs ------------------------------------------------------------------------------
    def enqueue_job(self, job: IngestionJob) -> IngestionJob:
        """Idempotent: if an active job of the same kind exists for the same source and
        document, that job is returned and nothing is written."""
        ...

    def lease_next_job(
        self, kind: JobKind | None, now: datetime, lease_seconds: float
    ) -> IngestionJob | None:
        """Atomically lease the oldest claimable job (queued, or failed whose retry time
        has come); increments ``attempts``."""
        ...

    def heartbeat_job(
        self, job_id: JobId, attempt: int, now: datetime, lease_seconds: float
    ) -> bool:
        """Extend the lease; ``False`` if the caller no longer holds it."""
        ...

    def complete_job(self, job_id: JobId, attempt: int, now: datetime) -> bool:
        """Mark done; ``False`` if the caller no longer holds the lease."""
        ...

    def fail_job(
        self,
        job_id: JobId,
        attempt: int,
        error_code: ErrorCode,
        now: datetime,
        retry_at: datetime | None,
    ) -> bool:
        """Record a failed attempt: retry at ``retry_at``, or ``dead`` when it is ``None``.
        ``False`` if the caller no longer holds the lease."""
        ...

    def release_expired_leases(self, now: datetime, max_attempts: int) -> int:
        """Re-queue leased jobs whose lease expired (``dead`` once ``max_attempts`` is
        reached: a job that keeps killing its worker must not block the queue)."""
        ...

    # index state -----------------------------------------------------------------------
    def bump_generation(self) -> int:
        """Increase the monotonic index generation and return the new value."""
        ...

    def update_index_metadata(self, metadata: IndexMetadata) -> None: ...


@runtime_checkable
class UnitOfWork(Protocol):
    def begin(self) -> AbstractContextManager[MetadataWriter]:
        """Take the single writer lock (``E_STORE_LOCKED`` if held) and open a transaction.

        Commits on normal exit and rolls back on any exception. Nested units of work are
        not allowed.
        """
        ...


@runtime_checkable
class Store(Protocol):
    """Handle given to services."""

    def reader(self) -> MetadataReader: ...

    def unit_of_work(self) -> UnitOfWork: ...

    def close(self) -> None:
        """Release every connection and lock held by this handle (idempotent)."""
        ...
