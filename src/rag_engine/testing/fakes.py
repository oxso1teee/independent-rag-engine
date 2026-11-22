"""Deterministic test doubles for time and sources."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import BinaryIO

from rag_engine.domain import (
    ErrorCode,
    RagError,
    SourceKey,
    SourceKind,
    normalize_source_key,
    require,
)
from rag_engine.ports import ScanReport, SourceEntry, SourceIdentity, SourceRevision

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class FakeClock:
    """Manually advanced clock implementing the ``Clock`` port."""

    def __init__(
        self,
        start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
        monotonic_start_ns: int = 0,
    ) -> None:
        require(start.utcoffset() == timedelta(0), "FakeClock needs a UTC start time")
        self._now = start
        self._monotonic_ns = monotonic_start_ns

    def now(self) -> datetime:
        return self._now

    def monotonic_ns(self) -> int:
        return self._monotonic_ns

    def advance(self, delta: timedelta) -> None:
        require(delta >= timedelta(0), "a clock cannot move backwards")
        self._now += delta
        self._monotonic_ns += (delta // timedelta(microseconds=1)) * 1000

    def unix_ms(self) -> int:
        """Current time as Unix milliseconds (feeds ``Uuid7Generator``)."""
        return (self._now - _EPOCH) // timedelta(milliseconds=1)


class InMemorySource:
    """``SourceAdapter`` over a dictionary. Revisions are deterministic counters."""

    def __init__(self, files: Mapping[str, bytes | str] | None = None) -> None:
        self._files: dict[SourceKey, tuple[bytes, SourceRevision]] = {}
        self._tick = 0
        for key, data in (files or {}).items():
            self.put(key, data)

    def put(self, key: str, data: bytes | str) -> SourceKey:
        """Add or replace an entry; every call yields a new ``mtime_ns``."""
        source_key = normalize_source_key(key)
        raw = data.encode("utf-8") if isinstance(data, str) else data
        self._tick += 1
        self._files[source_key] = (raw, SourceRevision(len(raw), self._tick * 1_000_000_000))
        return source_key

    def delete(self, key: str) -> None:
        self._files.pop(normalize_source_key(key), None)

    def identity(self) -> SourceIdentity:
        return SourceIdentity(kind=SourceKind.INLINE, locator_fingerprint=None)

    def scan(self, previous: Mapping[SourceKey, SourceRevision]) -> ScanReport:
        del previous  # every entry is reported; nothing is expensive to compute here
        return ScanReport(
            entries=tuple(
                SourceEntry(key=key, revision=revision)
                for key, (_, revision) in sorted(self._files.items())
            )
        )

    def discover(self, previous: Mapping[SourceKey, SourceRevision]) -> tuple[SourceEntry, ...]:
        return self.scan(previous).entries

    def open(self, key: SourceKey) -> BinaryIO:
        try:
            data, _ = self._files[key]
        except KeyError:
            raise RagError(
                ErrorCode.E_SOURCE_NOT_FOUND,
                "entry does not exist",
                context={"key": key.value},
            ) from None
        return io.BytesIO(data)
