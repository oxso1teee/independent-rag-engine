"""SQL implementation of ``MetadataReader`` and ``MetadataWriter``.

Every statement uses bound parameters; values never become part of SQL text. The only
interpolated fragments are fixed literals chosen in this module.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, TypeVar

import numpy as np

from rag_engine.domain import (
    ChunkId,
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
    JobStatus,
    RagError,
    Sha256Hex,
    Source,
    SourceId,
    SourceKey,
    SyncState,
    TextHash,
    thaw_context,
)
from rag_engine.domain.invariants import check_generation_monotonic, check_version_transition
from rag_engine.ports.cache import StoredEmbedding
from rag_engine.ports.store import StoredChunk
from rag_engine.storage.sqlite import codec

T = TypeVar("T")


def translate(exc: sqlite3.Error) -> RagError:
    """Map a SQLite failure onto the engine's error taxonomy (never leaks SQL or values)."""
    text = str(exc).lower()
    if isinstance(exc, sqlite3.OperationalError) and ("locked" in text or "busy" in text):
        return RagError(ErrorCode.E_STORE_LOCKED, "the database is busy")
    if isinstance(exc, sqlite3.IntegrityError):
        return RagError(
            ErrorCode.E_INTEGRITY,
            "the change violates a storage constraint",
            context={"sqlite_error": type(exc).__name__},
        )
    return RagError(
        ErrorCode.E_INTEGRITY,
        "the database operation failed",
        context={"sqlite_error": type(exc).__name__},
    )


class SqliteReader:
    """Read-only queries over one connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- plumbing ----------------------------------------------------------------------

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise translate(exc) from exc

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def _run(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise translate(exc) from exc

    @staticmethod
    def _map(rows: Sequence[sqlite3.Row], build: Callable[[sqlite3.Row], T]) -> list[T]:
        return [build(row) for row in rows]

    # -- sources -----------------------------------------------------------------------

    def get_source(self, source_id: SourceId) -> Source | None:
        row = self._one("SELECT * FROM sources WHERE id = ?", (str(source_id),))
        return None if row is None else codec.source_from_row(row)

    def get_source_by_name(self, name: str) -> Source | None:
        row = self._one("SELECT * FROM sources WHERE name = ?", (name,))
        return None if row is None else codec.source_from_row(row)

    def list_sources(self) -> Sequence[Source]:
        return self._map(self._all("SELECT * FROM sources ORDER BY name"), codec.source_from_row)

    # -- documents ---------------------------------------------------------------------

    def get_document(self, document_id: DocumentId) -> Document | None:
        row = self._one("SELECT * FROM documents WHERE id = ?", (str(document_id),))
        return None if row is None else codec.document_from_row(row)

    def get_document_by_key(self, source_id: SourceId, key: SourceKey) -> Document | None:
        row = self._one(
            "SELECT * FROM documents WHERE source_id = ? AND source_key = ?",
            (str(source_id), key.value),
        )
        return None if row is None else codec.document_from_row(row)

    def list_documents(
        self,
        source_id: SourceId,
        status: DocumentStatus | None = None,
        after: SourceKey | None = None,
        limit: int | None = None,
    ) -> Sequence[Document]:
        sql = "SELECT * FROM documents WHERE source_id = ?"
        params: list[Any] = [str(source_id)]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        if after is not None:
            sql += " AND source_key > ?"
            params.append(after.value)
        sql += " ORDER BY source_key"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._map(self._all(sql, params), codec.document_from_row)

    def count_documents(
        self,
        source_id: SourceId,
        status: DocumentStatus | None = None,
        *,
        has_version: bool | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) FROM documents WHERE source_id = ?"
        params: list[Any] = [str(source_id)]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        if has_version is not None:
            sql += " AND current_version_id IS " + ("NOT NULL" if has_version else "NULL")
        row = self._one(sql, params)
        return int(row[0]) if row is not None else 0

    def get_version(self, version_id: DocumentVersionId) -> DocumentVersion | None:
        row = self._one("SELECT * FROM document_versions WHERE id = ?", (str(version_id),))
        return None if row is None else codec.version_from_row(row)

    def list_versions(self, document_id: DocumentId) -> Sequence[DocumentVersion]:
        rows = self._all(
            "SELECT * FROM document_versions WHERE document_id = ? ORDER BY seq",
            (str(document_id),),
        )
        return self._map(rows, codec.version_from_row)

    def list_chunks(self, version_id: DocumentVersionId) -> Sequence[StoredChunk]:
        rows = self._all(
            "SELECT * FROM chunks WHERE document_version_id = ? ORDER BY ordinal",
            (str(version_id),),
        )
        return self._map(rows, codec.stored_chunk_from_row)

    def get_embedding_model(self, model_id: EmbeddingModelId) -> EmbeddingModel | None:
        row = self._one("SELECT * FROM embedding_models WHERE id = ?", (str(model_id),))
        return None if row is None else codec.embedding_model_from_row(row)

    def get_many(
        self, model_id: EmbeddingModelId, hashes: Sequence[TextHash]
    ) -> Mapping[TextHash, StoredEmbedding]:
        found: dict[TextHash, StoredEmbedding] = {}
        model = self.get_embedding_model(model_id)
        if model is None:
            return found
        for key in dict.fromkeys(hashes):
            row = self._one(
                "SELECT vector, dtype, norm, created_at FROM embeddings"
                " WHERE embedding_input_hash = ? AND model_id = ?",
                (key.hex, str(model_id)),
            )
            if row is None:
                continue
            blob = row["vector"]
            if row["dtype"] != "f16le" or len(blob) != model.dimension * 2:
                raise RagError(ErrorCode.E_INTEGRITY, "cached embedding has an invalid shape")
            vector = np.frombuffer(blob, dtype="<f2").astype(np.float32)
            if not np.isfinite(vector).all():
                raise RagError(ErrorCode.E_INTEGRITY, "cached embedding has non-finite values")
            found[key] = StoredEmbedding(
                key, vector, row["norm"], codec.parse_ts(row["created_at"])
            )
        return found

    # -- sync state --------------------------------------------------------------------

    def get_sync_state(self, source_id: SourceId, key: SourceKey) -> SyncState | None:
        row = self._one(
            "SELECT * FROM sync_state WHERE source_id = ? AND source_key = ?",
            (str(source_id), key.value),
        )
        return None if row is None else codec.sync_state_from_row(row)

    def list_sync_states(self, source_id: SourceId) -> Sequence[SyncState]:
        rows = self._all(
            "SELECT * FROM sync_state WHERE source_id = ? ORDER BY source_key", (str(source_id),)
        )
        return self._map(rows, codec.sync_state_from_row)

    # -- jobs --------------------------------------------------------------------------

    def get_job(self, job_id: JobId) -> IngestionJob | None:
        row = self._one("SELECT * FROM jobs WHERE id = ?", (str(job_id),))
        return None if row is None else codec.job_from_row(row)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        kind: JobKind | None = None,
        limit: int | None = None,
    ) -> Sequence[IngestionJob]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM jobs{where} ORDER BY id"  # noqa: S608 - fixed literals only
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._map(self._all(sql, params), codec.job_from_row)

    def count_jobs(self) -> dict[JobStatus, int]:
        rows = self._all("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        return {JobStatus(row[0]): int(row[1]) for row in rows}

    # -- index state -------------------------------------------------------------------

    def get_index_metadata(self) -> IndexMetadata | None:
        row = self._one("SELECT * FROM index_state WHERE id = 1")
        return None if row is None else codec.index_metadata_from_row(row)


class SqliteWriter(SqliteReader):
    """Mutations; only ever constructed inside an open transaction."""

    # -- sources -----------------------------------------------------------------------

    def put_source(self, source: Source) -> None:
        existing = self.get_source(source.id)
        if existing is not None and existing.locator_fingerprint != source.locator_fingerprint:
            raise RagError(
                ErrorCode.E_SOURCE_LOCATOR_CHANGED,
                "the locator of a source changes only through an explicit rebase",
                context={"source": source.name},
            )
        self._run(
            "INSERT INTO sources (id, name, kind, locator, locator_fingerprint, policy_json,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET name = excluded.name,"
            " policy_json = excluded.policy_json, updated_at = excluded.updated_at",
            (
                str(source.id),
                source.name,
                source.kind.value,
                source.locator,
                None if source.locator_fingerprint is None else source.locator_fingerprint.hex,
                codec.policy_to_json(source.policy),
                codec.ts(source.created_at),
                codec.ts(source.updated_at),
            ),
        )

    def rebase_source(
        self, source_id: SourceId, new_locator: str, new_fingerprint: Sha256Hex, now: datetime
    ) -> Source:
        if self.get_source(source_id) is None:
            raise RagError(ErrorCode.E_SOURCE_NOT_FOUND, "unknown source")
        self._run(
            "UPDATE sources SET locator = ?, locator_fingerprint = ?, updated_at = ? WHERE id = ?",
            (new_locator, new_fingerprint.hex, codec.ts(now), str(source_id)),
        )
        updated = self.get_source(source_id)
        assert updated is not None  # just written inside this transaction  # noqa: S101
        return updated

    def delete_source(self, source_id: SourceId, *, force: bool = False) -> None:
        if self.count_documents(source_id) > 0:
            if not force:
                raise RagError(
                    ErrorCode.E_INTEGRITY, "the source still has documents; deletion needs force"
                )
            self._run("DELETE FROM documents WHERE source_id = ?", (str(source_id),))
        self._run("DELETE FROM sources WHERE id = ?", (str(source_id),))

    # -- documents and versions ---------------------------------------------------------

    def put_document(self, document: Document) -> None:
        check_version_transition(self.get_document(document.id), document)
        if document.current_version_id is not None:
            version = self.get_version(document.current_version_id)
            if version is None or version.document_id != document.id:
                raise RagError(
                    ErrorCode.E_INTEGRITY,
                    "the current version must exist and belong to the document",
                    context={"document": str(document.id)},
                )
        self._run(
            "INSERT INTO documents (id, source_id, source_key, current_version_id, status,"
            " duplicate_of, missing_since, title, tags_json, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET source_key = excluded.source_key,"
            " current_version_id = excluded.current_version_id, status = excluded.status,"
            " duplicate_of = excluded.duplicate_of, missing_since = excluded.missing_since,"
            " title = excluded.title, tags_json = excluded.tags_json,"
            " updated_at = excluded.updated_at",
            (
                str(document.id),
                str(document.source_id),
                document.source_key.value,
                None if document.current_version_id is None else str(document.current_version_id),
                document.status.value,
                None if document.duplicate_of is None else str(document.duplicate_of),
                None if document.missing_since is None else codec.ts(document.missing_since),
                document.title,
                codec.dumps(list(document.tags)),
                codec.ts(document.created_at),
                codec.ts(document.updated_at),
            ),
        )

    def delete_document(self, document_id: DocumentId) -> None:
        self._run("DELETE FROM documents WHERE id = ?", (str(document_id),))

    def put_version(self, version: DocumentVersion) -> None:
        existing = self.get_version(version.id)
        if existing is not None:
            if existing != version:
                raise RagError(
                    ErrorCode.E_INTEGRITY,
                    "a version is immutable and was rewritten with different content",
                    context={"version": str(version.id)},
                )
            return
        self._run(
            "INSERT INTO document_versions (id, document_id, seq, content_hash, size_bytes,"
            " mtime_ns, mime, detected_by, parser_id, parser_version, normalizer_version,"
            " chunker_fingerprint,"
            " text_purged, parse_warnings_json, normalized_text, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(version.id),
                str(version.document_id),
                version.seq,
                version.content_hash.hex,
                version.size_bytes,
                version.mtime_ns,
                version.mime,
                version.detected_by,
                version.parser_id,
                version.parser_version,
                version.normalizer_version,
                version.chunker_fingerprint,
                int(version.text_purged),
                codec.dumps(list(version.parse_warnings)),
                version.normalized_text,
                codec.ts(version.created_at),
            ),
        )

    def mark_text_purged(self, version_id: DocumentVersionId) -> None:
        cursor = self._run(
            "UPDATE document_versions SET text_purged = 1, normalized_text = NULL WHERE id = ?",
            (str(version_id),),
        )
        if cursor.rowcount == 0:
            raise RagError(ErrorCode.E_INTEGRITY, "unknown version")

    def put_chunks(self, version_id: DocumentVersionId, chunks: Sequence[StoredChunk]) -> None:
        version = self.get_version(version_id)
        if version is None or version.text_purged:
            raise RagError(ErrorCode.E_INTEGRITY, "chunks require a non-purged version")
        for stored in chunks:
            chunk = stored.chunk
            if chunk.document_version_id != version_id or chunk.id != ChunkId.derive(
                version_id, version.chunker_fingerprint, chunk.ordinal
            ):
                raise RagError(ErrorCode.E_INTEGRITY, "chunk identity differs from its version")
            old = self._one("SELECT * FROM chunks WHERE id = ?", (str(chunk.id),))
            if old is not None:
                if codec.stored_chunk_from_row(old) != stored:
                    raise RagError(ErrorCode.E_INTEGRITY, "stored chunk changed for one id")
                continue
            self._run(
                "INSERT INTO chunks (id, document_version_id, ordinal, level, parent_id,"
                " kind, text, text_hash, embedding_input_hash, token_count, char_start,"
                " char_end, heading_path_json, page_start, page_end, language, indexed,"
                " is_duplicate, oversize, body_an, headings_an)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(chunk.id),
                    str(version_id),
                    chunk.ordinal,
                    chunk.level,
                    None if chunk.parent_id is None else str(chunk.parent_id),
                    chunk.kind.value,
                    chunk.text,
                    chunk.text_hash.hex,
                    None
                    if stored.embedding_input_hash is None
                    else stored.embedding_input_hash.hex,
                    chunk.token_count,
                    chunk.char_start,
                    chunk.char_end,
                    codec.dumps(list(chunk.heading_path)),
                    chunk.page_start,
                    chunk.page_end,
                    chunk.language.value,
                    int(chunk.indexed),
                    int(chunk.is_duplicate),
                    int(chunk.oversize),
                    stored.body_an,
                    stored.headings_an,
                ),
            )

    def put_embedding_model(self, model: EmbeddingModel) -> None:
        existing = self.get_embedding_model(model.id)
        if existing is not None:
            if existing != model:
                raise RagError(ErrorCode.E_INTEGRITY, "embedding model definition changed")
            return
        self._run(
            "INSERT INTO embedding_models (id, provider, name, revision, dimension,"
            " normalized, query_prefix, document_prefix, max_input_tokens, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(model.id),
                model.provider,
                model.name,
                model.revision,
                model.dimension,
                int(model.normalized),
                model.query_prefix,
                model.document_prefix,
                model.max_input_tokens,
                model.status.value,
                codec.ts(model.created_at),
            ),
        )

    def index_chunks(self, version_id: DocumentVersionId) -> None:
        rows = self._all(
            "SELECT chunk_rowid, body_an, headings_an FROM chunks"
            " WHERE document_version_id = ? AND indexed = 1 ORDER BY ordinal",
            (str(version_id),),
        )
        for row in rows:
            self._run(
                "INSERT INTO chunks_fts (rowid, body_an, headings_an) VALUES (?, ?, ?)",
                (row["chunk_rowid"], row["body_an"], row["headings_an"]),
            )

    def put_many(self, model_id: EmbeddingModelId, items: Sequence[StoredEmbedding]) -> None:
        model = self.get_embedding_model(model_id)
        if model is None:
            raise RagError(ErrorCode.E_INTEGRITY, "embedding model is not registered")
        for item in items:
            vector = item.vector
            if vector.ndim != 1 or vector.size != model.dimension:
                raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "embedding dimension differs")
            if not np.isfinite(vector).all():
                raise RagError(ErrorCode.E_INTEGRITY, "embedding contains non-finite values")
            packed = np.asarray(vector, dtype="<f2").tobytes()
            restored = np.frombuffer(packed, dtype="<f2").astype(np.float32)
            norm = float(np.linalg.norm(restored))
            if not np.isfinite(norm) or norm <= 0:
                raise RagError(ErrorCode.E_INTEGRITY, "embedding has invalid norm")
            if model.normalized and abs(norm - 1.0) >= 1e-3:
                raise RagError(ErrorCode.E_INTEGRITY, "normalized embedding has invalid norm")
            old = self._one(
                "SELECT vector FROM embeddings WHERE embedding_input_hash = ? AND model_id = ?",
                (item.embedding_input_hash.hex, str(model_id)),
            )
            if old is not None:
                if old["vector"] != packed:
                    raise RagError(ErrorCode.E_INTEGRITY, "cached embedding changed for one key")
                continue
            self._run(
                "INSERT INTO embeddings (embedding_input_hash, model_id, vector, dtype, norm,"
                " created_at) VALUES (?, ?, ?, 'f16le', ?, ?)",
                (
                    item.embedding_input_hash.hex,
                    str(model_id),
                    packed,
                    norm,
                    codec.ts(item.created_at),
                ),
            )

    def delete_unreferenced(self, model_id: EmbeddingModelId) -> int:
        cursor = self._run(
            "DELETE FROM embeddings WHERE model_id = ? AND NOT EXISTS"
            " (SELECT 1 FROM chunks WHERE chunks.embedding_input_hash ="
            " embeddings.embedding_input_hash AND chunks.indexed = 1)",
            (str(model_id),),
        )
        return cursor.rowcount

    # -- sync state --------------------------------------------------------------------

    def put_sync_state(self, state: SyncState) -> None:
        revision = state.last_revision
        self._run(
            "INSERT INTO sync_state (source_id, source_key, last_size_bytes, last_mtime_ns,"
            " last_content_hash, last_seen_at, status, attempts, last_error_code, next_retry_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (source_id, source_key) DO UPDATE SET"
            " last_size_bytes = excluded.last_size_bytes, last_mtime_ns = excluded.last_mtime_ns,"
            " last_content_hash = excluded.last_content_hash, last_seen_at = excluded.last_seen_at,"
            " status = excluded.status, attempts = excluded.attempts,"
            " last_error_code = excluded.last_error_code, next_retry_at = excluded.next_retry_at",
            (
                str(state.source_id),
                state.source_key.value,
                None if revision is None else revision.size_bytes,
                None if revision is None else revision.mtime_ns,
                None if state.last_content_hash is None else state.last_content_hash.hex,
                codec.ts(state.last_seen_at),
                state.status.value,
                state.attempts,
                None if state.last_error_code is None else state.last_error_code.value,
                None if state.next_retry_at is None else codec.ts(state.next_retry_at),
            ),
        )

    def delete_sync_state(self, source_id: SourceId, key: SourceKey) -> None:
        self._run(
            "DELETE FROM sync_state WHERE source_id = ? AND source_key = ?",
            (str(source_id), key.value),
        )

    # -- jobs --------------------------------------------------------------------------

    def enqueue_job(self, job: IngestionJob) -> IngestionJob:
        document = None if job.document_id is None else str(job.document_id)
        existing = self._one(
            "SELECT * FROM jobs WHERE kind = ? AND source_id = ? AND COALESCE(document_id, '') = ?"
            " AND status IN ('queued', 'leased', 'failed')",
            (job.kind.value, str(job.source_id), document or ""),
        )
        if existing is not None:
            return codec.job_from_row(existing)
        self._run(
            "INSERT INTO jobs (id, kind, source_id, document_id, stage, status, lease_until,"
            " attempts, error_code, payload_json, available_at, created_at, finished_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(job.id),
                job.kind.value,
                str(job.source_id),
                document,
                job.stage.value,
                job.status.value,
                None if job.lease_until is None else codec.ts(job.lease_until),
                job.attempts,
                None if job.error_code is None else job.error_code.value,
                codec.payload_to_json(job),
                codec.ts(job.available_at),
                codec.ts(job.created_at),
                None if job.finished_at is None else codec.ts(job.finished_at),
            ),
        )
        return job

    def lease_next_job(
        self, kind: JobKind | None, now: datetime, lease_seconds: float
    ) -> IngestionJob | None:
        sql = "SELECT id FROM jobs WHERE status IN ('queued', 'failed') AND available_at <= ?"
        params: list[Any] = [codec.ts(now)]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind.value)
        sql += " ORDER BY available_at, id LIMIT 1"
        row = self._one(sql, params)
        if row is None:
            return None
        self._run(
            "UPDATE jobs SET status = 'leased', lease_until = ?, attempts = attempts + 1,"
            " error_code = NULL WHERE id = ?",
            (codec.ts(now + timedelta(seconds=lease_seconds)), row["id"]),
        )
        return self.get_job(JobId.parse(row["id"]))

    def heartbeat_job(
        self, job_id: JobId, attempt: int, now: datetime, lease_seconds: float
    ) -> bool:
        cursor = self._run(
            "UPDATE jobs SET lease_until = ? WHERE id = ? AND status = 'leased' AND attempts = ?",
            (codec.ts(now + timedelta(seconds=lease_seconds)), str(job_id), attempt),
        )
        return cursor.rowcount == 1

    def complete_job(self, job_id: JobId, attempt: int, now: datetime) -> bool:
        cursor = self._run(
            "UPDATE jobs SET status = 'done', stage = 'committed', lease_until = NULL,"
            " finished_at = ? WHERE id = ? AND status = 'leased' AND attempts = ?",
            (codec.ts(now), str(job_id), attempt),
        )
        return cursor.rowcount == 1

    def fail_job(
        self,
        job_id: JobId,
        attempt: int,
        error_code: ErrorCode,
        now: datetime,
        retry_at: datetime | None,
    ) -> bool:
        if retry_at is None:
            cursor = self._run(
                "UPDATE jobs SET status = 'dead', lease_until = NULL, error_code = ?,"
                " finished_at = ? WHERE id = ? AND status = 'leased' AND attempts = ?",
                (error_code.value, codec.ts(now), str(job_id), attempt),
            )
        else:
            cursor = self._run(
                "UPDATE jobs SET status = 'failed', lease_until = NULL, error_code = ?,"
                " available_at = ? WHERE id = ? AND status = 'leased' AND attempts = ?",
                (error_code.value, codec.ts(retry_at), str(job_id), attempt),
            )
        return cursor.rowcount == 1

    def release_expired_leases(self, now: datetime, max_attempts: int) -> int:
        stamp = codec.ts(now)
        dead = self._run(
            "UPDATE jobs SET status = 'dead', lease_until = NULL, error_code = ?, finished_at = ?"
            " WHERE status = 'leased' AND lease_until <= ? AND attempts >= ?",
            (ErrorCode.E_TIMEOUT.value, stamp, stamp, max_attempts),
        ).rowcount
        requeued = self._run(
            "UPDATE jobs SET status = 'queued', lease_until = NULL, available_at = ?"
            " WHERE status = 'leased' AND lease_until <= ?",
            (stamp, stamp),
        ).rowcount
        return dead + requeued

    # -- index state -------------------------------------------------------------------

    def bump_generation(self) -> int:
        row = self._one(
            "UPDATE index_state SET index_generation = index_generation + 1 WHERE id = 1"
            " RETURNING index_generation"
        )
        if row is None:
            raise RagError(ErrorCode.E_INTEGRITY, "the index state has not been initialised")
        return int(row[0])

    def update_index_metadata(self, metadata: IndexMetadata) -> None:
        current = self.get_index_metadata()
        if current is not None:
            check_generation_monotonic(current, metadata)
        self._run(
            "INSERT INTO index_state (id, schema_version, index_generation,"
            " active_embedding_model_id, lexical_analyzer_id, chunker_default_fingerprint,"
            " vector_store_kind, calibration_json, created_at, last_gc_at)"
            " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET schema_version = excluded.schema_version,"
            " index_generation = excluded.index_generation,"
            " active_embedding_model_id = excluded.active_embedding_model_id,"
            " lexical_analyzer_id = excluded.lexical_analyzer_id,"
            " chunker_default_fingerprint = excluded.chunker_default_fingerprint,"
            " vector_store_kind = excluded.vector_store_kind,"
            " calibration_json = excluded.calibration_json, last_gc_at = excluded.last_gc_at",
            (
                metadata.schema_version,
                metadata.index_generation,
                None
                if metadata.active_embedding_model_id is None
                else str(metadata.active_embedding_model_id),
                metadata.lexical_analyzer_id,
                metadata.chunker_default_fingerprint,
                metadata.vector_store_kind,
                codec.dumps(thaw_context(metadata.calibration)),
                codec.ts(metadata.created_at),
                None if metadata.last_gc_at is None else codec.ts(metadata.last_gc_at),
            ),
        )
