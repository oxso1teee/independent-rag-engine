"""Source adapter port: discovery and reading of raw documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import BinaryIO, Protocol, runtime_checkable

from rag_engine.domain import Sha256Hex, SourceKey, SourceKind, SourceRevision

__all__ = [
    "ScanIssue",
    "ScanReport",
    "SourceAdapter",
    "SourceEntry",
    "SourceIdentity",
    "SourceRevision",
]


@dataclass(frozen=True, slots=True)
class SourceEntry:
    key: SourceKey
    revision: SourceRevision
    mime_hint: str | None = None


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    kind: SourceKind
    locator_fingerprint: Sha256Hex | None


@dataclass(frozen=True, slots=True)
class ScanIssue:
    """Something the adapter could not read or deliberately skipped.

    ``path`` is a source key; with ``subtree`` set it is the prefix of a whole directory
    whose content could not be listed. ``reason`` is a short stable token, never text of
    a document.
    """

    path: str
    reason: str
    subtree: bool = False


@dataclass(frozen=True, slots=True)
class ScanReport:
    """Result of one complete enumeration of a source."""

    entries: tuple[SourceEntry, ...]
    unreadable: tuple[ScanIssue, ...] = ()
    skipped: tuple[ScanIssue, ...] = field(default=())

    def covers(self, key: SourceKey) -> bool:
        """Whether ``key`` lies in a part of the source that could not be inspected."""
        for issue in self.unreadable:
            if issue.path == key.value or (
                issue.subtree and key.value.startswith(issue.path.rstrip("/") + "/")
            ):
                return True
        return False


@runtime_checkable
class SourceAdapter(Protocol):
    def identity(self) -> SourceIdentity: ...

    def scan(self, previous: Mapping[SourceKey, SourceRevision]) -> ScanReport:
        """Enumerate the source once: current entries sorted by key plus what was unreadable.

        A part of the source that cannot be inspected is reported in ``unreadable`` so that
        callers never mistake it for deleted content. ``previous`` is the last known
        revision per key; adapters may use it to skip work but must still report every
        existing entry.
        """
        ...

    def discover(self, previous: Mapping[SourceKey, SourceRevision]) -> Iterable[SourceEntry]:
        """Current entries in a deterministic (sorted by key) order: ``scan(...).entries``."""
        ...

    def open(self, key: SourceKey) -> BinaryIO:
        """Open an entry for reading; raises ``E_SOURCE_NOT_FOUND`` if it is gone."""
        ...
