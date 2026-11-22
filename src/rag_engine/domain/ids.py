"""Typed identifiers and content hashes.

All identifiers are immutable value objects. Time-ordered UUIDs (version 7,
RFC 9562) are built by pure functions: the caller supplies the timestamp and
the entropy, so identifier creation is deterministic and testable. The standard
library of Python 3.12 has no ``uuid7``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self
from uuid import UUID

from rag_engine.domain.errors import ContextValue, require, violation

_MAX_UNIX_MS = (1 << 48) - 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHUNK_ID_RE = re.compile(r"[a-z2-7]{26}")
_MODEL_ID_RE = re.compile(
    r"(?P<provider>[a-z0-9_-]+)/(?P<name>[^@#\s]+)@(?P<revision>[^#\s]+)"
    r"#d(?P<dimension>[0-9]+):(?P<norm>l2|raw):(?P<prefix>[0-9a-f]{8})"
)


def uuid7(unix_ms: int, rand: bytes) -> UUID:
    """Build a UUIDv7 from a millisecond timestamp and 10 bytes of entropy.

    Layout (RFC 9562 section 5.7): 48-bit ``unix_ms``, version nibble 7, 12 bits
    ``rand_a`` (low nibble of ``rand[0]`` and ``rand[1]``), variant ``0b10``,
    62 bits ``rand_b`` (``rand[2:]`` with the two top bits masked off).
    """
    require(0 <= unix_ms <= _MAX_UNIX_MS, "uuid7 timestamp out of 48-bit range", unix_ms=unix_ms)
    require(len(rand) == 10, "uuid7 requires exactly 10 bytes of entropy", length=len(rand))
    raw = bytearray(16)
    raw[0:6] = unix_ms.to_bytes(6, "big")
    raw[6] = 0x70 | (rand[0] & 0x0F)
    raw[7] = rand[1]
    raw[8] = 0x80 | (rand[2] & 0x3F)
    raw[9:16] = rand[3:10]
    return UUID(bytes=bytes(raw))


class Uuid7Generator:
    """Strictly increasing UUIDv7 source (RFC 9562 method 1: counter in ``rand_a``).

    The clock and the entropy source are injected. Not thread-safe: use one
    generator per event loop or guard it externally.
    """

    def __init__(self, now_ms: Callable[[], int], random_bytes: Callable[[int], bytes]) -> None:
        self._now_ms = now_ms
        self._random_bytes = random_bytes
        self._last_ms = -1
        self._counter = 0

    def _seed(self) -> int:
        # 11 bits only, leaving headroom before the 12-bit counter overflows.
        return int.from_bytes(self._random_bytes(2), "big") & 0x7FF

    def next(self) -> UUID:
        now = max(self._now_ms(), self._last_ms)
        if now == self._last_ms:
            counter = self._counter + 1
            if counter > 0xFFF:
                now += 1
                counter = self._seed()
        else:
            counter = self._seed()
        self._last_ms, self._counter = now, counter
        rand = bytes([counter >> 8, counter & 0xFF]) + self._random_bytes(8)
        return uuid7(now, rand)


@dataclass(frozen=True, slots=True, order=True)
class _Uuid7Id:
    value: UUID

    def __post_init__(self) -> None:
        require(
            self.value.variant == uuid.RFC_4122 and self.value.version == 7,
            f"{type(self).__name__} must wrap a UUIDv7",
            value=str(self.value),
        )

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse the canonical textual form."""
        try:
            parsed = UUID(text)
        except ValueError as exc:
            raise violation(f"invalid {cls.__name__}", value=text[:64]) from exc
        return cls(parsed)

    @property
    def unix_ms(self) -> int:
        """Creation timestamp encoded in the identifier."""
        return self.value.int >> 80

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True, order=True)
class SourceId(_Uuid7Id):
    """Identifies a registered source. Immutable."""


@dataclass(frozen=True, slots=True, order=True)
class DocumentId(_Uuid7Id):
    """Identifies a document across versions."""


@dataclass(frozen=True, slots=True, order=True)
class DocumentVersionId(_Uuid7Id):
    """Identifies one immutable version of a document."""


@dataclass(frozen=True, slots=True, order=True)
class JobId(_Uuid7Id):
    """Identifies one ingestion job."""


@dataclass(frozen=True, slots=True, order=True)
class RequestId(_Uuid7Id):
    """End-to-end identifier of a request or job, present in every log event."""


class IdFactory:
    """Creates typed identifiers from a single :class:`Uuid7Generator`."""

    def __init__(self, generator: Uuid7Generator) -> None:
        self._generator = generator

    def source_id(self) -> SourceId:
        return SourceId(self._generator.next())

    def document_id(self) -> DocumentId:
        return DocumentId(self._generator.next())

    def document_version_id(self) -> DocumentVersionId:
        return DocumentVersionId(self._generator.next())

    def job_id(self) -> JobId:
        return JobId(self._generator.next())

    def request_id(self) -> RequestId:
        return RequestId(self._generator.next())


@dataclass(frozen=True, slots=True, order=True)
class Sha256Hex:
    """Lowercase hexadecimal SHA-256 digest."""

    hex: str

    def __post_init__(self) -> None:
        require(_SHA256_RE.fullmatch(self.hex) is not None, "invalid sha256 hex digest")

    @classmethod
    def of_bytes(cls, data: bytes) -> Self:
        return cls(hashlib.sha256(data).hexdigest())

    @classmethod
    def of_text(cls, text: str) -> Self:
        """Digest of the UTF-8 encoding of ``text`` (no further normalisation)."""
        return cls.of_bytes(text.encode("utf-8"))

    def __str__(self) -> str:
        return self.hex


@dataclass(frozen=True, slots=True, order=True)
class ContentHash(Sha256Hex):
    """SHA-256 of the raw bytes of a source file."""


@dataclass(frozen=True, slots=True, order=True)
class TextHash(Sha256Hex):
    """SHA-256 of normalised chunk text: the deduplication and embedding-cache key."""


def text_hash(text: str) -> TextHash:
    return TextHash.of_text(text)


def chunker_fingerprint(
    chunker_id: str,
    version: str,
    params: Mapping[str, ContextValue],
    normalizer_version: str,
) -> str:
    """Stable fingerprint of everything that influences chunk boundaries."""
    payload = json.dumps(
        {
            "chunker": chunker_id,
            "version": version,
            "params": dict(params),
            "normalizer": normalizer_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ChunkId:
    """Deterministic chunk identifier: 128 bits of a SHA-256, base32 (26 chars).

    Derived from ``(document_version_id, chunker_fingerprint, ordinal)``, so
    reprocessing the same version with the same chunker yields the same ids.
    """

    value: str

    def __post_init__(self) -> None:
        require(_CHUNK_ID_RE.fullmatch(self.value) is not None, "invalid ChunkId format")
        raw = base64.b32decode(self.value.upper() + "======")
        canonical = base64.b32encode(raw).decode("ascii").rstrip("=").lower()
        require(canonical == self.value, "non-canonical ChunkId encoding")

    @classmethod
    def derive(cls, version_id: DocumentVersionId, fingerprint: str, ordinal: int) -> Self:
        require(ordinal >= 0, "chunk ordinal must be non-negative", ordinal=ordinal)
        fp = fingerprint.encode("utf-8")
        message = (
            version_id.value.bytes + len(fp).to_bytes(4, "big") + fp + ordinal.to_bytes(8, "big")
        )
        digest = hashlib.sha256(message).digest()[:16]
        return cls(base64.b32encode(digest).decode("ascii").rstrip("=").lower())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class EmbeddingModelId:
    """Identity of an embedding model configuration.

    Rendered as ``{provider}/{name}@{revision}#d{dim}:{norm}:{prefix_hash8}``.
    Any change of a component yields a different identifier.
    """

    provider: str
    name: str
    revision: str
    dimension: int
    normalized: bool
    prefix_hash: str

    def __post_init__(self) -> None:
        require(
            _MODEL_ID_RE.fullmatch(self._render()) is not None,
            "invalid EmbeddingModelId components",
            provider=self.provider,
            name=self.name,
        )
        require(self.dimension > 0, "embedding dimension must be positive")

    def _render(self) -> str:
        norm = "l2" if self.normalized else "raw"
        return (
            f"{self.provider}/{self.name}@{self.revision}"
            f"#d{self.dimension}:{norm}:{self.prefix_hash}"
        )

    @classmethod
    def build(
        cls,
        *,
        provider: str,
        name: str,
        revision: str,
        dimension: int,
        normalized: bool,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> Self:
        prefixes = f"{query_prefix}\x00{document_prefix}".encode()
        return cls(
            provider,
            name,
            revision,
            dimension,
            normalized,
            hashlib.sha256(prefixes).hexdigest()[:8],
        )

    @classmethod
    def parse(cls, text: str) -> Self:
        match = _MODEL_ID_RE.fullmatch(text)
        if match is None:
            raise violation("invalid EmbeddingModelId", value=text[:128])
        return cls(
            match["provider"],
            match["name"],
            match["revision"],
            int(match["dimension"]),
            match["norm"] == "l2",
            match["prefix"],
        )

    def __str__(self) -> str:
        return self._render()


@dataclass(frozen=True, slots=True, order=True)
class SourceKey:
    """Document key inside a source: relative, NFC-normalised, ``/``-separated."""

    value: str

    def __post_init__(self) -> None:
        value = self.value
        require(bool(value), "source key must not be empty")
        require(unicodedata.is_normalized("NFC", value), "source key must be NFC", key=value)
        require("\\" not in value and "\x00" not in value, "invalid character in source key")
        require(not value.startswith("/"), "source key must be relative", key=value)
        segments = value.split("/")
        require(
            all(s not in ("", ".", "..") for s in segments),
            "source key has an empty or dot segment",
            key=value,
        )

    def __str__(self) -> str:
        return self.value


def normalize_source_key(raw: str) -> SourceKey:
    """Normalise a user-supplied relative path into a :class:`SourceKey`."""
    return SourceKey(unicodedata.normalize("NFC", raw.replace("\\", "/")))
