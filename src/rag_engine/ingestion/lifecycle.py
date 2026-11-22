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

# Mass-delete guard (ARCHITECTURE §A, plus two owner-approved safety rules). A scan that makes
# documents vanish is refused, and nothing is changed, when ANY of these holds:
#   * empty_scan:     the scan found no document at all;
#   * all_known_gone: every known document vanished at once (a swapped or wrong root);
#   * threshold:      more than max(10 files, 20 % of the known documents) vanished.
# ``--accept-mass-delete`` confirms the result explicitly.
MASS_DELETE_FLOOR = 10
MASS_DELETE_RATIO = 0.20


@dataclass(frozen=True, slots=True)
class GcResult:
    removed: int
    removed_keys: tuple[SourceKey, ...] = ()


def mass_delete_rule(*, vanished: int, tracked: int, entries_seen: int) -> str | None:
    """The guard rule that applies, or ``None`` when the disappearance looks ordinary."""
    if vanished == 0:
        return None
    if entries_seen == 0:
        return "empty_scan"
    if tracked > 0 and vanished >= tracked:
        return "all_known_gone"
    if vanished > max(MASS_DELETE_FLOOR, MASS_DELETE_RATIO * tracked):
        return "threshold"
    return None


_RULE_TEXT = {
    "empty_scan": "the scan found no documents at all",
    "all_known_gone": "every known document disappeared at once",
    "threshold": "more documents disappeared at once than the safety limit allows",
}


def check_mass_delete(
    *, vanished: int, tracked: int, entries_seen: int, accept: bool = False
) -> None:
    """Raise ``E_MASS_DELETE_GUARD`` when ``vanished`` documents disappeared suspiciously."""
    rule = mass_delete_rule(vanished=vanished, tracked=tracked, entries_seen=entries_seen)
    if rule is None or accept:
        return
    raise RagError(
        ErrorCode.E_MASS_DELETE_GUARD,
        f"{_RULE_TEXT[rule]}; nothing was changed. Check the source root and permissions, "
        "or confirm with --accept-mass-delete",
        context={
            "rule": rule,
            "vanished": vanished,
            "tracked": tracked,
            "entries_seen": entries_seen,
            "limit": int(max(MASS_DELETE_FLOOR, MASS_DELETE_RATIO * tracked)),
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
    return replace(document, status=DocumentStatus.ACTIVE, missing_since=None, updated_at=now)


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
