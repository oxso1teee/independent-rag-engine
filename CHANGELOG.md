# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/); versions follow the phases of the
project until a release is cut.

## [Unreleased]

### Phase 2 refinements
- Schema v2 (`0002_align_data_model.sql`, forward only; `0001` untouched): `document_versions`,
  `normalized_text`, `UNIQUE (document_id, content_hash)`, no `pending` status. Documents
  without a version are `active` and not searchable until their first commit. Phase 4's
  migration becomes v3.
- Automatic verified backup before migrating a populated database; foreign-key-safe table
  rebuilds in the migration runner; the schema version is mirrored from `schema_migrations`.
- Mass-delete guard: `max(10, 20 %)` plus `empty_scan` and `all_known_gone`; the flag is
  `--accept-mass-delete` (`--allow-mass-delete` is removed).
- CLI exit codes follow the error category.
- Retry delays have jitter; retry and lease parameters are `[ingest]` settings.
- MIME detection on `filetype` and `charset-normalizer` (cp1251, koi8-r, cp866, UTF-16 with BOM);
  the encoding is stored with the job.
- Ingestion emits structured stage events (numbers only).
- Composition root moved to `rag_engine/container.py` (new entry point); storage split into
  `core.py`, `uow.py`, `backup.py`; tests reorganised into `perf`, `chaos`, `integrity`; PR CI
  excludes `perf`.
- New tests: `SIGKILL` at every stage, invariants at teardown, a model-based ingestion test.
- ADRs 0005-0007; ADR 0004 refined.

## Phase 2: SQLite core and document ingestion

### Added
- SQLite store: WAL, single writer lock (`E_STORE_LOCKED`), unit of work with rollback,
  read-only opening, verified `VACUUM INTO` backup.
- Numbered, checksummed SQL migrations; schema v1 (sources, documents, versions, sync state,
  jobs, index state).
- `FilesystemSource` with include/exclude globs, symlink policy, root boundary, NFC keys.
- Change detection (size/mtime first, SHA-256 only when needed), content-based MIME detection.
- Source identity, explicit rebase, content-proven move detection.
- Tombstones with a per-source grace period, garbage collection and the mass-delete guard
  (`E_MASS_DELETE_GUARD`).
- Persistent job queue: idempotent enqueue, leases, heartbeats, exponential backoff, dead
  letters, crash recovery.
- `rag source add|list|rebase`, `rag ingest [--dry-run]`, `rag status` (Typer), JSON output and
  stable exit codes.
- Domain: `SyncState`, `IngestionJob`, `IndexMetadata`, `JobId`, `DocumentStatus.PENDING`;
  `SourceRevision` moved into the domain.
- Documentation: data model, testing guide; ADR 0004 accepted with amendments.

### Changed
- The Phase 1 `Store` port is replaced by `MetadataReader`, `MetadataWriter`, `UnitOfWork`
  and `Store`; `SourceAdapter` gained `scan`.
- The command line is built with Typer instead of `argparse`.
- Import contracts: `ingestion` sits above the adapters and may not import them.

### Not included
Parsing, chunking, embeddings, indexes, retrieval, generation, API (later phases).

## [0.0.1] - Phase 1: core foundation
Domain model, error taxonomy, ports, configuration, logging, test doubles, CI.
