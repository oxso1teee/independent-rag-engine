"""Change detection: compare a fresh scan with the stored sync state.

Order of cheapness: (size, mtime) from the scan first; the file is only read, and
SHA-256 only computed, when that signal differs from the stored one (or the entry has no
reliable state yet). A rescan of an unchanged source therefore reads no file at all.
Detection is read-only and deterministic; applying the plan is the caller's job.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import BinaryIO

from rag_engine.domain import (
    ContentHash,
    Document,
    DocumentId,
    ErrorCode,
    RagError,
    SourceId,
    SourceKey,
    SourceRevision,
    SyncState,
    SyncStatus,
)
from rag_engine.ingestion.mime import MimeResult, detect_mime
from rag_engine.ports import ScanReport, SourceAdapter

_CHUNK = 1024 * 1024


class ChangeKind(StrEnum):
    UNCHANGED = "unchanged"
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"
    INACCESSIBLE = "inaccessible"


@dataclass(frozen=True, slots=True)
class DetectedChange:
    key: SourceKey
    kind: ChangeKind
    revision: SourceRevision | None = None
    content_hash: ContentHash | None = None
    mime: MimeResult | None = None
    touch: bool = False  # content identical, only the revision signal needs refreshing
    restored: bool = False  # the entry is back after having been tombstoned
    error: ErrorCode | None = None
    reason: str | None = None  # stable token, never document text


@dataclass(frozen=True, slots=True)
class Move:
    """The same document under a new key (proved by identical content)."""

    document_id: DocumentId
    old_key: SourceKey
    new_key: SourceKey
    revision: SourceRevision
    content_hash: ContentHash


@dataclass(frozen=True, slots=True)
class ChangePlan:
    source_id: SourceId
    changes: tuple[DetectedChange, ...]
    moves: tuple[Move, ...] = ()
    entries_seen: int = 0
    skipped: int = 0

    def of_kind(self, kind: ChangeKind) -> tuple[DetectedChange, ...]:
        return tuple(c for c in self.changes if c.kind is kind)

    def counts(self) -> dict[str, int]:
        counts = {kind.value: len(self.of_kind(kind)) for kind in ChangeKind}
        counts["moved"] = len(self.moves)
        return counts

    @property
    def mutates(self) -> bool:
        """Whether applying the plan writes anything."""
        return bool(self.moves) or any(
            c.kind is not ChangeKind.UNCHANGED or c.touch or c.restored for c in self.changes
        )


def hash_stream(stream: BinaryIO) -> ContentHash:
    digest = hashlib.sha256()
    while block := stream.read(_CHUNK):
        digest.update(block)
    return ContentHash(digest.hexdigest())


class ChangeDetector:
    """Classifies every entry of a scan; reads files only when the cheap signal differs."""

    def __init__(self, adapter: SourceAdapter) -> None:
        self._adapter = adapter

    def detect(
        self,
        source_id: SourceId,
        report: ScanReport,
        states: Mapping[SourceKey, SyncState],
        documents: Mapping[SourceKey, Document],
    ) -> ChangePlan:
        changes: list[DetectedChange] = []
        present: set[SourceKey] = set()
        for entry in report.entries:
            present.add(entry.key)
            changes.append(
                self._classify(entry.key, entry.revision, states.get(entry.key), documents)
            )
        for key, state in sorted(states.items(), key=lambda item: item[0].value):
            if key in present or state.status is SyncStatus.MISSING:
                continue
            if report.covers(key):
                changes.append(
                    DetectedChange(key, ChangeKind.INACCESSIBLE, reason="unreadable_location")
                )
            else:
                changes.append(DetectedChange(key, ChangeKind.DELETED))
        return ChangePlan(
            source_id=source_id,
            changes=tuple(changes),
            entries_seen=len(report.entries),
            skipped=len(report.skipped),
        )

    # ------------------------------------------------------------------------------

    def _classify(
        self,
        key: SourceKey,
        revision: SourceRevision,
        state: SyncState | None,
        documents: Mapping[SourceKey, Document],
    ) -> DetectedChange:
        if state is None:
            return self._examine(key, revision, ChangeKind.NEW, None)
        restored = state.status is SyncStatus.MISSING
        if (
            state.status is SyncStatus.FAILED
            and state.last_error_code is ErrorCode.E_UNSUPPORTED_FORMAT
            and state.last_revision == revision
        ):
            return DetectedChange(key, ChangeKind.UNCHANGED, revision)  # known, permanent
        if state.status is SyncStatus.FAILED or state.last_revision is None:
            # An earlier read failed or no hash is on record: look at the content again.
            return self._examine(key, revision, ChangeKind.MODIFIED, state, restored=restored)
        if state.last_revision == revision and not restored:
            return DetectedChange(key, ChangeKind.UNCHANGED, revision)  # nothing is read
        examined = self._examine(key, revision, ChangeKind.MODIFIED, state, restored=restored)
        if (
            examined.kind is ChangeKind.MODIFIED
            and examined.content_hash == state.last_content_hash
        ):
            needs_job = restored and (
                (doc := documents.get(key)) is None or doc.current_version_id is None
            )
            if not needs_job:
                return DetectedChange(
                    key,
                    ChangeKind.UNCHANGED,
                    revision,
                    content_hash=examined.content_hash,
                    touch=not restored,
                    restored=restored,
                )
        return examined

    def _examine(
        self,
        key: SourceKey,
        revision: SourceRevision,
        kind: ChangeKind,
        state: SyncState | None,
        *,
        restored: bool = False,
    ) -> DetectedChange:
        try:
            with self._adapter.open(key) as stream:
                mime = detect_mime(stream, key.value)
                stream.seek(0)
                digest = hash_stream(stream)
        except RagError as exc:
            if exc.code is not ErrorCode.E_SOURCE_NOT_FOUND:
                raise
            reason = str(exc.context.get("reason", "unreadable"))
            return DetectedChange(
                key,
                ChangeKind.INACCESSIBLE,
                error=ErrorCode.E_SOURCE_NOT_FOUND,
                reason=reason,
            )
        error = None if mime.supported else ErrorCode.E_UNSUPPORTED_FORMAT
        return DetectedChange(
            key,
            kind,
            revision,
            content_hash=digest,
            mime=mime,
            restored=restored,
            error=error,
            reason=None if mime.supported else "unsupported_format",
        )


def sync_states_by_key(states: Sequence[SyncState]) -> dict[SourceKey, SyncState]:
    return {state.source_key: state for state in states}


def documents_by_key(documents: Sequence[Document]) -> dict[SourceKey, Document]:
    return {document.source_key: document for document in documents}
