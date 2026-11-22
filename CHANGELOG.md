# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/); versions follow the phases of the
roadmap until a release is cut.

## [Unreleased] - Phase 2: SQLite core and document ingestion

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
