"""Disappearing documents: tombstones, grace period, garbage collection, mass-delete guard.

A document that vanished from its source is never deleted outright. It becomes a
*tombstone* (``missing``) that is fully reversible: if the entry reappears during the grace
period the document returns with the same id. Only after the grace period does garbage
collection remove it. A guard refuses to tombstone anything when the scan looks wrong
(empty result, or an unusually large share of the corpus gone at once), because that
points to a wrong root or an access problem, not to real deletions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from rag_engine.domain import (
    Document,
    DocumentStatus,
    ErrorCode,
    RagError,
    Source,
    SourceKey,
    SyncStatus,
)
from rag_engine.ports import Clock, MetadataReader, MetadataWriter, Store

# Assumed defaults (the specification gives the 20 % threshold; the minimum count keeps a
# tiny corpus usable): trip when more than 20 % of the tracked documents vanished at once
# and at least two of them did, or when everything vanished from an empty scan.
MASS_DELETE_RATIO = 0.20
MASS_DELETE_MIN_COUNT = 2


@dataclass(frozen=True, slots=True)
class GcResult:
    removed: int
    removed_keys: tuple[SourceKey, ...] = ()


def check_mass_delete(
    *, vanished: int, tracked: int, entries_seen: int, allow: bool = False
) -> None:
    """Raise ``E_MASS_DELETE_GUARD`` when ``vanished`` of ``tracked`` documents look suspicious."""
    if vanished == 0 or allow:
        return
    reason: str | None = None
    if entries_seen == 0:
        reason = "the scan found no documents at all"
    elif vanished >= MASS_DELETE_MIN_COUNT and vanished / max(tracked, 1) > MASS_DELETE_RATIO:
        reason = "more than 20% of the tracked documents disappeared at once"
    if reason is not None:
        raise RagError(
            ErrorCode.E_MASS_DELETE_GUARD,
            f"{reason}; nothing was changed. Check the source root and permissions, "
            "or confirm explicitly",
            context={
                "vanished": vanished,
                "tracked": tracked,
                "entries_seen": entries_seen,
                "threshold_percent": int(MASS_DELETE_RATIO * 100),
            },
        )


def tombstone(writer: MetadataWriter, document: Document, now: datetime) -> None:
    """Mark a document missing (reversible) and record it in its sync state."""
    if document.status is DocumentStatus.MISSING:
        return
    writer.put_document(
        replace(document, status=DocumentStatus.MISSING, missing_since=now, updated_at=now)
    )
    state = writer.get_sync_state(document.source_id, document.source_key)
    if state is not None:
        writer.put_sync_state(
            replace(
                state,
                status=SyncStatus.MISSING,
                last_seen_at=now,
                last_error_code=None,
                next_retry_at=None,
            )
        )


def restore(document: Document, now: datetime) -> Document:
    """The document after its entry reappeared within the grace period."""
    status = (
        DocumentStatus.ACTIVE
        if document.current_version_id is not None
        else (DocumentStatus.PENDING)
    )
    return replace(document, status=status, missing_since=None, updated_at=now)


class Lifecycle:
    """Garbage collection of tombstones whose grace period is over."""

    def __init__(self, store: Store, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    @staticmethod
    def expired(document: Document, source: Source, now: datetime) -> bool:
        return (
            document.status is DocumentStatus.MISSING
            and document.missing_since is not None
            and document.missing_since + timedelta(days=source.policy.delete_grace_days) <= now
        )

    def due(self, reader: MetadataReader, source: Source) -> list[Document]:
        now = self._clock.now()
        missing = reader.list_documents(source.id, DocumentStatus.MISSING)
        return [d for d in missing if self.expired(d, source, now)]

    def collect_garbage(self, source: Source) -> GcResult:
        """Remove tombstones older than the source's grace period (irreversible)."""
        due = self.due(self._store.reader(), source)
        if not due:
            return GcResult(0)
        now = self._clock.now()
        with self._store.unit_of_work().begin() as writer:
            for document in due:
                writer.delete_sync_state(document.source_id, document.source_key)
                writer.delete_document(document.id)
            metadata = writer.get_index_metadata()
            if metadata is not None:
                writer.update_index_metadata(replace(metadata, last_gc_at=now))
        return GcResult(len(due), tuple(d.source_key for d in due))
