# 0004. Store boundaries and the domain additions needed for Phase 2

Status: Proposed (2026-11-22). Nothing in this record is implemented yet; it defines the
exact scope of the first Phase 2 tasks.

## Context

The `Store` port written in Phase 1 has `get/put` operations for sources, documents,
versions and chunks plus `transaction()`. The data model and architecture additionally
require synchronisation state, a persistent job queue, index metadata, embedding models
with a single active one, a content-addressed embedding cache, a single writer and one
transaction spanning all tables. Phase 2 (SQLite core, WAL, unit of work, schema v1,
ingestion) cannot be built on the current port without breaking changes, so they are made
now, before any implementation exists.

## Gap analysis

| Item | Status today | Required for Phase 2 |
|---|---|---|
| `Source`, `Document`, `DocumentVersion`, `Chunk` | exist in `domain` | add list/count/update/delete operations (below) |
| `SyncState` | missing | new domain model |
| `IngestionJob` (+ `JobId`) | missing | new domain model and typed id |
| `IndexMetadata` | missing | new domain model |
| `EmbeddingModel` | exists in `domain` | store operations and the "exactly one active" rule |
| Embedding (vector per `text_hash` and model) | missing | port-level type, not a domain model (see below) |
| `EmbeddingCache` | missing | new port (needed from Phase 4; shape fixed now) |
| `SourceRevision` | lives in `ports/source.py` | move to `domain` (the sync state uses it) |
| `UnitOfWork` | `Store.transaction()` only | separate port with reader/writer split |

## Domain additions

All frozen, slotted, validating dataclasses, like the existing models.

- `SourceRevision(size_bytes, mtime_ns)`: moved from `ports.source`; the port re-exports it.
- `SyncStatus` = `ok | changed | failed | missing`.
- `SyncState(source_id, source_key, last_revision: SourceRevision | None,
  last_content_hash: ContentHash | None, last_seen_at, status: SyncStatus, attempts: int,
  last_error_code: ErrorCode | None, next_retry_at: datetime | None)`.
  Invariants: `attempts >= 0`; `status=failed` requires `last_error_code`;
  `next_retry_at` only when failed; UTC-aware timestamps.
- `JobId` (UUIDv7 typed id), `JobKind` = `scan | ingest_doc | reindex | gc`,
  `JobStage` = `discovered | parsed | chunked | embedded | committed`,
  `JobStatus` = `queued | leased | done | failed | dead`.
- `IngestionJob(id, kind, source_id, document_id | None, stage, status, lease_until | None,
  attempts, error_code | None, payload: Mapping[str, ContextValue], created_at,
  finished_at | None)`. Invariants: `leased` requires `lease_until`; `done`/`dead` require
  `finished_at`; `ingest_doc` requires `document_id`; `error_code` only for `failed|dead`;
  payload is deeply frozen (same rules as error context) and never holds document text.
- `IndexMetadata(schema_version, index_generation, active_embedding_model_id | None,
  lexical_analyzer_id, chunker_default_fingerprint, vector_store_kind,
  calibration: Mapping[str, ContextValue], created_at, last_gc_at | None)`.
  Invariants: `index_generation >= 0`; `schema_version >= 1`.
- Cross-entity checks in `domain/invariants.py`: `check_single_active_model(models)` (at
  most one active; exactly one once indexing started), `check_generation_monotonic(old, new)`
  (I8), `check_sync_state_matches_document(...)`.
- The embedding vector itself is **not** a domain model (the domain stays free of numpy);
  `StoredEmbedding(text_hash, vector, norm, created_at)` is defined next to the
  `EmbeddingCache` port.

## Port boundaries

Two responsibilities are separated: *what can be read* and *what can be changed only
inside one atomic unit of work*.

- `MetadataReader` (read-only, usable concurrently, never takes the writer lock):
  `get_source`, `get_source_by_name`, `list_sources`; `get_document`,
  `get_document_by_key`, `list_documents(source_id, status, after, limit)`,
  `count_documents(source_id, status)`; `get_version`, `list_versions(document_id)`;
  `get_chunks(ids)`, `list_chunks(version_id)`; `get_sync_state`,
  `list_sync_states(source_id)`; `list_jobs(status, kind, limit)`; `get_index_metadata`;
  `get_embedding_model`, `list_embedding_models`.
- `MetadataWriter` (extends the reader; obtainable only from a unit of work):
  - sources: `put_source`, `rebase_source(source_id, new_locator, new_fingerprint)`
    (guards `E_SOURCE_LOCATOR_CHANGED` on plain updates), `delete_source` (refuses while
    active documents exist unless forced);
  - documents and versions: `put_document` (upsert), `put_version` (immutable; a second
    write of the same id with different content is `E_INTEGRITY`), `mark_text_purged`;
  - chunks: `put_chunks`, `delete_chunks(version_id)`;
  - sync state: `put_sync_state`, `delete_sync_state`;
  - jobs: `enqueue_job`, `lease_next_job(kind, now, lease_seconds)`, `heartbeat_job`,
    `complete_job`, `fail_job(error_code, retry_at | None)` (moves to `dead` after the
    attempt limit), `release_expired_leases(now)`;
  - index state: `bump_generation()` (monotonic, returns the new value),
    `set_active_embedding_model`, `update_index_metadata`, `put_embedding_model`.
- `UnitOfWork`: `begin() -> ContextManager[MetadataWriter]`. Acquires the single writer
  lock (`E_STORE_LOCKED` if held), commits on normal exit, rolls back on any exception.
  Nested units of work are not allowed. All timestamps come from the caller (`Clock`), the
  store never reads the system clock.
- `Store`: the handle given to services: `reader() -> MetadataReader` and
  `unit_of_work() -> UnitOfWork`. The Phase 1 `Store` (`transaction()`, `get_*`/`put_*`)
  is replaced; nothing implements or uses it yet.
- `EmbeddingCache` (synchronous port): `get_many(model_id, hashes) -> Mapping[TextHash,
  StoredEmbedding]`, `put_many(model_id, items)` (idempotent, content-addressed),
  `delete_unreferenced(model_id) -> int`. It participates in the same unit of work through
  the adapter, not through the port.
- Not ports: schema migrations and the backup (`VACUUM INTO`) are adapter-internal and
  surface only as `E_MIGRATION_REQUIRED` / `E_STORE_LOCKED`.
- `VectorIndex` and `LexicalIndex` stay separate ports. The SQLite adapters share the
  connection with the unit of work, so "version + chunks + FTS + `current_version`" commits
  atomically (Phase 4) without leaking a transaction type into the ports.

## Consequences

- Phase 2 starts with: move `SourceRevision`, add the domain models and invariants with
  tests, replace the `Store` port, add the `UnitOfWork` and `EmbeddingCache` ports and
  their contract tests, and only then the SQLite adapter.
- Contract tests for the store ports must run against every adapter, including an
  in-memory test double added to `rag_engine.testing` when Phase 2 needs it.
- `ports` still imports only `domain`; moving `SourceRevision` into `domain` keeps domain
  models independent of ports.
- The write surface is a single, auditable interface; readers cannot mutate state.
