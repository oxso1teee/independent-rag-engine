"""Core domain models.

All models are frozen dataclasses with ``slots`` and validate their own
invariants on construction by raising :class:`~rag_engine.domain.errors.RagError`
(``E_INTEGRITY``). Sequences are tuples, so instances are deeply immutable.
Cross-entity invariants live in :mod:`rag_engine.domain.invariants`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from rag_engine.domain.errors import ContextValue, ErrorCode, freeze_context, require
from rag_engine.domain.ids import (
    ChunkId,
    ContentHash,
    DocumentId,
    DocumentVersionId,
    EmbeddingModelId,
    JobId,
    RequestId,
    Sha256Hex,
    SourceId,
    SourceKey,
    TextHash,
)

_SOURCE_NAME_RE = re.compile(r"[a-z0-9-]{1,64}")
MAX_TOP_K = 100
MAX_CANDIDATE_K = 500


def _require_utc(value: datetime, name: str) -> None:
    offset = value.utcoffset()
    require(offset is not None and offset.total_seconds() == 0, f"{name} must be UTC-aware")


class Language(StrEnum):
    RU = "ru"
    EN = "en"
    MIXED = "mixed"
    UND = "und"


# --------------------------------------------------------------------- sources


class SourceKind(StrEnum):
    FS = "fs"
    INLINE = "inline"


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    max_file_bytes: int = 50 * 1024 * 1024
    cleaners: tuple[str, ...] = ()
    follow_symlinks: bool = False
    delete_grace_days: int = 30

    def __post_init__(self) -> None:
        require(self.max_file_bytes > 0, "max_file_bytes must be positive")
        require(self.delete_grace_days >= 0, "delete_grace_days must be non-negative")
        require(len(self.include) > 0, "a source policy needs at least one include glob")


@dataclass(frozen=True, slots=True, kw_only=True)
class Source:
    id: SourceId
    name: str
    kind: SourceKind
    locator: str | None
    locator_fingerprint: Sha256Hex | None
    policy: SourcePolicy = field(default_factory=SourcePolicy)
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        require(_SOURCE_NAME_RE.fullmatch(self.name) is not None, "invalid source name")
        if self.kind is SourceKind.INLINE:
            require(
                self.locator is None and self.locator_fingerprint is None,
                "an inline source has no locator",
            )
        else:
            require(bool(self.locator), "a filesystem source requires a locator")
            require(
                self.locator is not None
                and self.locator_fingerprint == Sha256Hex.of_text(self.locator),
                "locator fingerprint does not match the locator",
            )
        _require_utc(self.created_at, "created_at")
        _require_utc(self.updated_at, "updated_at")
        require(self.updated_at >= self.created_at, "updated_at precedes created_at")


# ------------------------------------------------------------------- documents


class DocumentStatus(StrEnum):
    ACTIVE = "active"
    MISSING = "missing"
    DELETED = "deleted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class Document:
    """One entry of a source.

    ``current_version_id`` is ``None`` from the first discovery until the first successful
    indexing commit; such a document is ``active`` but *not searchable* (it has no chunks).
    Once set, it never goes back to ``None`` (see ``check_version_transition``).
    """

    id: DocumentId
    source_id: SourceId
    source_key: SourceKey
    current_version_id: DocumentVersionId | None
    status: DocumentStatus
    duplicate_of: DocumentId | None = None
    missing_since: datetime | None = None
    title: str | None = None
    tags: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        require(self.duplicate_of != self.id, "a document cannot duplicate itself")
        require(
            (self.status is DocumentStatus.MISSING) == (self.missing_since is not None),
            "missing_since is set exactly when the status is 'missing'",
        )
        if self.missing_since is not None:
            _require_utc(self.missing_since, "missing_since")
        _require_utc(self.created_at, "created_at")
        _require_utc(self.updated_at, "updated_at")
        require(self.updated_at >= self.created_at, "updated_at precedes created_at")

    @property
    def is_searchable(self) -> bool:
        """Only active documents that already have a committed version take part in retrieval."""
        return self.status is DocumentStatus.ACTIVE and self.current_version_id is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentVersion:
    id: DocumentVersionId
    document_id: DocumentId
    seq: int
    content_hash: ContentHash
    size_bytes: int
    mtime_ns: int
    mime: str
    detected_by: str
    parser_id: str
    parser_version: str
    normalizer_version: str
    chunker_fingerprint: str
    text_purged: bool = False
    parse_warnings: tuple[str, ...] = ()
    normalized_text: bytes | None = None  # zlib-compressed; dropped when the text is purged
    created_at: datetime

    def __post_init__(self) -> None:
        require(self.seq >= 1, "version sequence starts at 1", seq=self.seq)
        require(
            not (self.text_purged and self.normalized_text is not None),
            "a version with purged text keeps no normalized text",
        )
        require(self.size_bytes >= 0 and self.mtime_ns >= 0, "negative size or mtime")
        _require_utc(self.created_at, "created_at")


# ------------------------------------------------------------- synchronisation


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """Cheap change signal of one source entry."""

    size_bytes: int
    mtime_ns: int

    def __post_init__(self) -> None:
        require(self.size_bytes >= 0 and self.mtime_ns >= 0, "negative revision values")


class SyncStatus(StrEnum):
    OK = "ok"  # in sync with the last processed content
    CHANGED = "changed"  # change detected, processing pending
    FAILED = "failed"  # the entry could not be read or is not supported
    MISSING = "missing"  # the entry disappeared from the source


@dataclass(frozen=True, slots=True, kw_only=True)
class SyncState:
    """Last observation of one source entry; drives change detection."""

    source_id: SourceId
    source_key: SourceKey
    last_revision: SourceRevision | None
    last_content_hash: ContentHash | None
    last_seen_at: datetime
    status: SyncStatus
    attempts: int = 0
    last_error_code: ErrorCode | None = None
    next_retry_at: datetime | None = None

    def __post_init__(self) -> None:
        require(self.attempts >= 0, "attempts must be non-negative")
        if self.status is SyncStatus.FAILED:
            require(self.last_error_code is not None, "a failed sync state needs an error code")
        require(
            self.next_retry_at is None or self.status is SyncStatus.FAILED,
            "next_retry_at is only meaningful for a failed state",
        )
        _require_utc(self.last_seen_at, "last_seen_at")
        if self.next_retry_at is not None:
            _require_utc(self.next_retry_at, "next_retry_at")


# ------------------------------------------------------------------------ jobs


class JobKind(StrEnum):
    SCAN = "scan"
    INGEST_DOC = "ingest_doc"
    REINDEX = "reindex"
    GC = "gc"


class JobStage(StrEnum):
    DISCOVERED = "discovered"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    COMMITTED = "committed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    DONE = "done"
    FAILED = "failed"  # last attempt failed; waits for ``available_at`` to be retried
    DEAD = "dead"  # gave up: needs operator attention

    @property
    def is_active(self) -> bool:
        """Whether the job can still be processed (used for de-duplication)."""
        return self in (JobStatus.QUEUED, JobStatus.LEASED, JobStatus.FAILED)


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestionJob:
    id: JobId
    kind: JobKind
    source_id: SourceId
    document_id: DocumentId | None
    stage: JobStage = JobStage.DISCOVERED
    status: JobStatus = JobStatus.QUEUED
    lease_until: datetime | None = None
    attempts: int = 0
    error_code: ErrorCode | None = None
    payload: Mapping[str, ContextValue] = field(default_factory=dict)
    available_at: datetime
    created_at: datetime
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_context(self.payload))
        require(self.attempts >= 0, "attempts must be non-negative")
        require(
            (self.status is JobStatus.LEASED) == (self.lease_until is not None),
            "lease_until is set exactly while the job is leased",
        )
        require(
            (self.status in (JobStatus.DONE, JobStatus.DEAD)) == (self.finished_at is not None),
            "finished_at is set exactly for done and dead jobs",
        )
        require(
            self.kind is not JobKind.INGEST_DOC or self.document_id is not None,
            "an ingest_doc job needs a document",
        )
        require(
            self.error_code is None or self.status in (JobStatus.FAILED, JobStatus.DEAD),
            "error_code is only kept for failed and dead jobs",
        )
        for name in ("available_at", "created_at", "lease_until", "finished_at"):
            value: datetime | None = getattr(self, name)
            if value is not None:
                _require_utc(value, name)


# ----------------------------------------------------------------- index state


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexMetadata:
    """Singleton describing the derived indexes of one store."""

    schema_version: int
    index_generation: int
    active_embedding_model_id: EmbeddingModelId | None
    lexical_analyzer_id: str
    chunker_default_fingerprint: str
    vector_store_kind: str
    calibration: Mapping[str, ContextValue] = field(default_factory=dict)
    created_at: datetime
    last_gc_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration", freeze_context(self.calibration))
        require(self.schema_version >= 1, "schema_version starts at 1")
        require(self.index_generation >= 0, "index_generation must be non-negative")
        _require_utc(self.created_at, "created_at")
        if self.last_gc_at is not None:
            _require_utc(self.last_gc_at, "last_gc_at")


# ---------------------------------------------------------------------- parsing


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"


@dataclass(frozen=True, slots=True, kw_only=True)
class Block:
    kind: BlockKind
    text: str
    char_start: int
    char_end: int
    level: int | None = None
    page: int | None = None
    heading_path: tuple[str, ...] = ()
    attrs: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.attrs.items():
            require(
                isinstance(key, str) and isinstance(value, str | int | float | bool),
                "block attributes are scalars keyed by strings",
            )
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))
        require(0 <= self.char_start < self.char_end, "invalid block span")
        require(len(self.text) == self.char_end - self.char_start, "block text/span mismatch")
        if self.kind is BlockKind.HEADING:
            require(self.level is not None and 1 <= self.level <= 6, "heading level must be 1..6")
        elif self.kind is not BlockKind.LIST_ITEM:
            require(self.level is None, "only headings and list items carry a level")
        require(self.page is None or self.page >= 1, "page numbers start at 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedDocument:
    """Normalised text plus structural blocks whose offsets index that text."""

    normalized_text: str
    blocks: tuple[Block, ...] = ()
    title: str | None = None
    author: str | None = None
    created: datetime | None = None
    language: Language = Language.UND
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.created is not None:
            _require_utc(self.created, "created")
        previous_start = 0
        for block in self.blocks:
            require(block.char_end <= len(self.normalized_text), "block exceeds the text")
            require(
                self.normalized_text[block.char_start : block.char_end] == block.text,
                "block text differs from the normalised text",
                char_start=block.char_start,
            )
            require(block.char_start >= previous_start, "blocks must be ordered by offset")
            previous_start = block.char_start


# ------------------------------------------------------------------- chunking


class ChunkKind(StrEnum):
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    target_tokens: int = 256
    max_tokens: int = 384
    min_tokens: int = 48
    overlap_ratio: float = 0.12

    def __post_init__(self) -> None:
        require(
            0 < self.min_tokens <= self.target_tokens <= self.max_tokens,
            "chunk policy requires 0 < min <= target <= max tokens",
        )
        require(0.0 <= self.overlap_ratio < 0.5, "overlap ratio must be in [0, 0.5)")


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    id: ChunkId
    document_version_id: DocumentVersionId
    ordinal: int
    level: int
    parent_id: ChunkId | None
    kind: ChunkKind
    text: str
    text_hash: TextHash
    token_count: int
    char_start: int
    char_end: int
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    language: Language = Language.UND
    indexed: bool = True
    is_duplicate: bool = False
    oversize: bool = False

    def __post_init__(self) -> None:
        require(self.ordinal >= 0, "chunk ordinal must be non-negative")
        require(self.level in (0, 1), "chunk level must be 0 (leaf) or 1 (section)")
        require(self.parent_id != self.id, "a chunk cannot be its own parent")
        require(self.level == 0 or self.parent_id is None, "a section chunk has no parent")
        require(bool(self.text), "chunk text must not be empty")
        require(self.text_hash == TextHash.of_text(self.text), "text_hash does not match the text")
        require(self.token_count >= 0, "token_count must be non-negative")
        require(0 <= self.char_start < self.char_end, "invalid chunk span")
        require(len(self.text) == self.char_end - self.char_start, "chunk text/span mismatch")
        require(
            (self.page_start is None) == (self.page_end is None),
            "page_start and page_end are set together",
        )
        if self.page_start is not None and self.page_end is not None:
            require(1 <= self.page_start <= self.page_end, "invalid page range")
        require(not (self.level == 1 and self.indexed), "section chunks are not indexed")
        require(not (self.is_duplicate and self.indexed), "duplicate chunks are not indexed")


# ------------------------------------------------------------------- embedding


class ModelStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingModel:
    id: EmbeddingModelId
    provider: str
    name: str
    revision: str
    dimension: int
    normalized: bool
    query_prefix: str = ""
    document_prefix: str = ""
    max_input_tokens: int
    status: ModelStatus = ModelStatus.ACTIVE
    created_at: datetime

    def __post_init__(self) -> None:
        expected = EmbeddingModelId.build(
            provider=self.provider,
            name=self.name,
            revision=self.revision,
            dimension=self.dimension,
            normalized=self.normalized,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
        )
        require(self.id == expected, "embedding model id does not match its parameters")
        require(self.max_input_tokens > 0, "max_input_tokens must be positive")
        _require_utc(self.created_at, "created_at")


# ------------------------------------------------------------------- retrieval


class RetrievalMode(StrEnum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class RerankMode(StrEnum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalFilters:
    """Typed metadata filters. Unknown keys are impossible by construction."""

    sources: tuple[str, ...] = ()
    path_prefix: str | None = None
    path_glob: str | None = None
    tags_any: tuple[str, ...] = ()
    tags_all: tuple[str, ...] = ()
    language: Language | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    doc_ids: tuple[DocumentId, ...] = ()
    exclude_doc_ids: tuple[DocumentId, ...] = ()
    kinds: tuple[ChunkKind, ...] = ()

    def __post_init__(self) -> None:
        for name in ("modified_after", "modified_before"):
            value: datetime | None = getattr(self, name)
            if value is not None:
                _require_utc(value, name)
        if self.modified_after is not None and self.modified_before is not None:
            require(self.modified_after < self.modified_before, "empty modification window")
        require(
            not set(self.doc_ids) & set(self.exclude_doc_ids),
            "a document cannot be both included and excluded",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalQuery:
    request_id: RequestId
    text: str
    mode: RetrievalMode
    top_k: int = 8
    candidate_k: int = 40
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    rerank: RerankMode = RerankMode.AUTO
    language_hint: Language | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        require(bool(self.text.strip()), "query text must not be blank")
        require(1 <= self.top_k <= MAX_TOP_K, "top_k out of range", top_k=self.top_k)
        require(
            self.top_k <= self.candidate_k <= MAX_CANDIDATE_K,
            "candidate_k must satisfy top_k <= candidate_k <= 500",
            candidate_k=self.candidate_k,
        )


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: ChunkId
    score: float

    def __post_init__(self) -> None:
        require(math.isfinite(self.score), "scores must be finite")


@dataclass(frozen=True, slots=True)
class ChunkRef:
    chunk_id: ChunkId
    document_id: DocumentId
    version_id: DocumentVersionId


@dataclass(frozen=True, slots=True)
class HitScores:
    dense: float | None = None
    lexical: float | None = None
    fused: float | None = None
    rerank: float | None = None

    def __post_init__(self) -> None:
        for value in (self.dense, self.lexical, self.fused, self.rerank):
            require(value is None or math.isfinite(value), "scores must be finite")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalHit:
    chunk_ref: ChunkRef
    rank: int
    scores: HitScores = field(default_factory=HitScores)

    def __post_init__(self) -> None:
        require(self.rank >= 1, "ranks start at 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalResult:
    request_id: RequestId
    index_generation: int
    embedding_model_id: EmbeddingModelId | None
    hits: tuple[RetrievalHit, ...] = ()
    fallbacks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require(self.index_generation >= 0, "index_generation must be non-negative")
        require(
            [hit.rank for hit in self.hits] == list(range(1, len(self.hits) + 1)),
            "hit ranks must be 1..n in order",
        )
