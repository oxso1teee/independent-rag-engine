"""Conversion between domain objects and SQLite rows (the only place that knows columns)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from rag_engine.domain import (
    ContentHash,
    Document,
    DocumentId,
    DocumentStatus,
    DocumentVersion,
    DocumentVersionId,
    EmbeddingModel,
    EmbeddingModelId,
    ErrorCode,
    IndexMetadata,
    IngestionJob,
    JobId,
    JobKind,
    JobStage,
    JobStatus,
    ModelStatus,
    Sha256Hex,
    Source,
    SourceId,
    SourceKey,
    SourceKind,
    SourcePolicy,
    SourceRevision,
    SyncState,
    SyncStatus,
    thaw_context,
)


def ts(value: datetime) -> str:
    """Fixed-width UTC text: lexicographic order equals chronological order."""
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_ts_opt(value: str | None) -> datetime | None:
    return None if value is None else parse_ts(value)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def embedding_model_from_row(row: sqlite3.Row) -> EmbeddingModel:
    return EmbeddingModel(
        id=EmbeddingModelId.parse(row["id"]),
        provider=row["provider"],
        name=row["name"],
        revision=row["revision"],
        dimension=row["dimension"],
        normalized=bool(row["normalized"]),
        query_prefix=row["query_prefix"],
        document_prefix=row["document_prefix"],
        max_input_tokens=row["max_input_tokens"],
        status=ModelStatus(row["status"]),
        created_at=parse_ts(row["created_at"]),
    )


# ------------------------------------------------------------------------ source


def policy_to_json(policy: SourcePolicy) -> str:
    return dumps(
        {
            "include": list(policy.include),
            "exclude": list(policy.exclude),
            "max_file_bytes": policy.max_file_bytes,
            "cleaners": list(policy.cleaners),
            "follow_symlinks": policy.follow_symlinks,
            "delete_grace_days": policy.delete_grace_days,
        }
    )


def policy_from_json(text: str) -> SourcePolicy:
    raw = json.loads(text)
    return SourcePolicy(
        include=tuple(raw["include"]),
        exclude=tuple(raw["exclude"]),
        max_file_bytes=raw["max_file_bytes"],
        cleaners=tuple(raw["cleaners"]),
        follow_symlinks=raw["follow_symlinks"],
        delete_grace_days=raw["delete_grace_days"],
    )


def source_from_row(row: sqlite3.Row) -> Source:
    fingerprint = row["locator_fingerprint"]
    return Source(
        id=SourceId.parse(row["id"]),
        name=row["name"],
        kind=SourceKind(row["kind"]),
        locator=row["locator"],
        locator_fingerprint=None if fingerprint is None else Sha256Hex(fingerprint),
        policy=policy_from_json(row["policy_json"]),
        created_at=parse_ts(row["created_at"]),
        updated_at=parse_ts(row["updated_at"]),
    )


# ---------------------------------------------------------------------- document


def document_from_row(row: sqlite3.Row) -> Document:
    version = row["current_version_id"]
    duplicate = row["duplicate_of"]
    return Document(
        id=DocumentId.parse(row["id"]),
        source_id=SourceId.parse(row["source_id"]),
        source_key=SourceKey(row["source_key"]),
        current_version_id=None if version is None else DocumentVersionId.parse(version),
        status=DocumentStatus(row["status"]),
        duplicate_of=None if duplicate is None else DocumentId.parse(duplicate),
        missing_since=parse_ts_opt(row["missing_since"]),
        title=row["title"],
        tags=tuple(json.loads(row["tags_json"])),
        created_at=parse_ts(row["created_at"]),
        updated_at=parse_ts(row["updated_at"]),
    )


def version_from_row(row: sqlite3.Row) -> DocumentVersion:
    return DocumentVersion(
        id=DocumentVersionId.parse(row["id"]),
        document_id=DocumentId.parse(row["document_id"]),
        seq=row["seq"],
        content_hash=ContentHash(row["content_hash"]),
        size_bytes=row["size_bytes"],
        mtime_ns=row["mtime_ns"],
        mime=row["mime"],
        detected_by=row["detected_by"],
        parser_id=row["parser_id"],
        parser_version=row["parser_version"],
        normalizer_version=row["normalizer_version"],
        chunker_fingerprint=row["chunker_fingerprint"],
        text_purged=bool(row["text_purged"]),
        parse_warnings=tuple(json.loads(row["parse_warnings_json"])),
        normalized_text=row["normalized_text"],
        created_at=parse_ts(row["created_at"]),
    )


# -------------------------------------------------------------------- sync state


def sync_state_from_row(row: sqlite3.Row) -> SyncState:
    size, mtime = row["last_size_bytes"], row["last_mtime_ns"]
    digest, code = row["last_content_hash"], row["last_error_code"]
    return SyncState(
        source_id=SourceId.parse(row["source_id"]),
        source_key=SourceKey(row["source_key"]),
        last_revision=None if size is None or mtime is None else SourceRevision(size, mtime),
        last_content_hash=None if digest is None else ContentHash(digest),
        last_seen_at=parse_ts(row["last_seen_at"]),
        status=SyncStatus(row["status"]),
        attempts=row["attempts"],
        last_error_code=None if code is None else ErrorCode(code),
        next_retry_at=parse_ts_opt(row["next_retry_at"]),
    )


# -------------------------------------------------------------------------- jobs


def job_from_row(row: sqlite3.Row) -> IngestionJob:
    document, code = row["document_id"], row["error_code"]
    return IngestionJob(
        id=JobId.parse(row["id"]),
        kind=JobKind(row["kind"]),
        source_id=SourceId.parse(row["source_id"]),
        document_id=None if document is None else DocumentId.parse(document),
        stage=JobStage(row["stage"]),
        status=JobStatus(row["status"]),
        lease_until=parse_ts_opt(row["lease_until"]),
        attempts=row["attempts"],
        error_code=None if code is None else ErrorCode(code),
        payload=json.loads(row["payload_json"]),
        available_at=parse_ts(row["available_at"]),
        created_at=parse_ts(row["created_at"]),
        finished_at=parse_ts_opt(row["finished_at"]),
    )


def payload_to_json(job: IngestionJob) -> str:
    return dumps(thaw_context(job.payload))


# ------------------------------------------------------------------- index state


def index_metadata_from_row(row: sqlite3.Row) -> IndexMetadata:
    model = row["active_embedding_model_id"]
    return IndexMetadata(
        schema_version=row["schema_version"],
        index_generation=row["index_generation"],
        active_embedding_model_id=None if model is None else EmbeddingModelId.parse(model),
        lexical_analyzer_id=row["lexical_analyzer_id"],
        chunker_default_fingerprint=row["chunker_default_fingerprint"],
        vector_store_kind=row["vector_store_kind"],
        calibration=json.loads(row["calibration_json"]),
        created_at=parse_ts(row["created_at"]),
        last_gc_at=parse_ts_opt(row["last_gc_at"]),
    )
