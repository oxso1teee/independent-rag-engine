"""Source adapter port: discovery and reading of raw documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import BinaryIO, Protocol, runtime_checkable

from rag_engine.domain import Sha256Hex, SourceKey, SourceKind, require


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """Cheap change signal of one entry."""

    size_bytes: int
    mtime_ns: int

    def __post_init__(self) -> None:
        require(self.size_bytes >= 0 and self.mtime_ns >= 0, "negative revision values")


@dataclass(frozen=True, slots=True)
class SourceEntry:
    key: SourceKey
    revision: SourceRevision
    mime_hint: str | None = None


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    kind: SourceKind
    locator_fingerprint: Sha256Hex | None


@runtime_checkable
class SourceAdapter(Protocol):
    def identity(self) -> SourceIdentity: ...

    def discover(self, previous: Mapping[SourceKey, SourceRevision]) -> Iterable[SourceEntry]:
        """Enumerate current entries in a deterministic (sorted by key) order.

        ``previous`` is the last known revision per key; adapters may use it to
        skip work but must still report every existing entry.
        """
        ...

    def open(self, key: SourceKey) -> BinaryIO:
        """Open an entry for reading; raises ``E_SOURCE_NOT_FOUND`` if it is gone."""
        ...
