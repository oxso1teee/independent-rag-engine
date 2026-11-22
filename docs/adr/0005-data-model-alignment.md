# 0005. Aligning the Phase 2 data model with the specification

Status: Accepted (2026-11-22). Corrects parts of [ADR 0004](0004-store-boundaries-for-phase-2.md).

## Context

A compliance review of Phase 2 against the original specification (`DATA_MODEL`,
`ARCHITECTURE`, `IMPLEMENTATION_ROADMAP`) found places where the first implementation
differed from it: an extra `pending` document status, table and column names, a missing
uniqueness rule, and `duplicate_of` without an owner phase. The owner decided each point.

## Decisions

1. **No `pending` status.** `DocumentStatus` is `active | missing | deleted | failed`.
   A document is created at its first discovery with `current_version_id = NULL` and status
   `active`; it receives its first version in the commit that ends its first successful
   indexing. The specification's invariant "`active` ⇒ `current_version_id` NOT NULL" is
   therefore refined to:
   - `current_version_id` is NULL only until the first commit;
   - a document without a version is **not searchable** (`Document.is_searchable` is true only
     for `active` documents that have a version), so it never reaches retrieval;
   - the transition is one-way: NULL → first version → newer versions, never back to NULL
     (`check_version_transition`, enforced by `MetadataWriter.put_document`);
   - a current version must exist and belong to the document.

   *Alternatives rejected:* keeping `pending` (not in the specification); using `failed` for
   "not yet indexed" (wrong meaning, would hide real failures); creating the document row only
   at the first commit (the specification creates the `DocumentId` at first discovery, and jobs
   refer to it). `rag status` reports such documents as "awaiting first ingestion".
2. **Forward migration `0002_align_data_model.sql`.** The published `0001` and its checksum
   (pinned in a test) are unchanged. `0002` renames `versions` to `document_versions`
   (the physical name in `DATA_MODEL` §6), adds `normalized_text BLOB` and
   `UNIQUE (document_id, content_hash)`, and rebuilds `documents` with the narrower status
   `CHECK` (existing `pending` rows become `active`, their NULL version is kept). The roadmap's
   Phase 4 migration for chunks, embeddings and FTS5 (`0002` in the roadmap) becomes **`0003`**.
   The rebuild runs with foreign keys off and `foreign_key_check` before commit (the SQLite
   procedure) so that versions and jobs are never cascaded away; a database with data is
   backed up first (`<db>.pre-migration-v<from>-to-v<to>.bak`, never overwritten).
   No local database existed when this was decided.
3. **The runner owns the schema version.** `schema_migrations` is the source of truth.
   `index_state.schema_version` is mirrored from it by the store after migrating; migration
   scripts never edit it (a test asserts that).
4. **`IngestionJob.available_at` stays.** It is the time before which a job must not be
   claimed: the creation time for a new job, the retry time after a retryable failure, the
   re-queue time after a lease expired. The specification has no such field on jobs
   (`next_retry_at` exists only on `SyncState`); the more general name was kept.
5. **`duplicate_of` belongs to Phase 3.** The column and the rule "a document cannot duplicate
   itself" exist; marking duplicates needs normalised content and is implemented after
   normalisation, not in Phase 2.
6. **Specification table typo.** The `DocumentVersion` listing in `DATA_MODEL` also names
   `vector` and `norm`; these belong to embeddings and are not columns of `document_versions`.

## Consequences

- Phase 3 writes `normalized_text` (zlib) and the first versions; Phase 4 starts at migration 3.
- `rag status --json` lists such documents under `awaiting_first_ingest`.
- ADR 0004's `PENDING` amendment is superseded by decision 1.
