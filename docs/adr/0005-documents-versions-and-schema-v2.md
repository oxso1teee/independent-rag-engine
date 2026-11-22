# 0005. Documents, versions and schema v2

Status: Accepted (2026-11-22). Refines [ADR 0004](0004-store-boundaries-for-phase-2.md).

## Context

A document is discovered before it has been indexed, so it exists without a version for some
time. Versions, jobs and the schema history need rules that keep identifiers stable and every
transition checkable.

## Decisions

1. **Statuses are `active | missing | deleted | failed`.** A document is created at its first
   discovery with `current_version_id = NULL` and status `active`; it receives its first
   version in the commit that ends its first successful indexing. Therefore:
   - `current_version_id` is NULL only until the first commit;
   - a document without a version is **not searchable** (`Document.is_searchable` is true only
     for `active` documents that have a version), so it never reaches retrieval;
   - the transition is one-way: NULL, first version, newer versions, never back to NULL
     (`check_version_transition`, enforced by `MetadataWriter.put_document`);
   - a current version must exist and belong to the document.

   *Alternatives rejected:* a separate "not yet indexed" status (one more state in every
   query for no gain); using `failed` for it (wrong meaning, hides real failures); creating the
   row only at the first commit (the `DocumentId` must exist from first discovery because jobs
   refer to it). `rag status` reports such documents as "awaiting first ingestion".
2. **Schema v2 is a forward migration (`0002_align_data_model.sql`).** Migration `0001` is
   published and immutable (its checksum is pinned in a test). `0002` renames `versions` to
   `document_versions`, adds `normalized_text BLOB` and `UNIQUE (document_id, content_hash)`,
   and rebuilds `documents` with the status `CHECK` above (existing `pending` rows become
   `active`, their NULL version is kept). The migration for chunks, embeddings and the
   full-text index is **`0003`**. The rebuild runs with foreign keys off and
   `foreign_key_check` before commit, so versions and jobs are never cascaded away; a
   database that already has data is backed up first
   (`<db>.pre-migration-v<from>-to-v<to>.bak`, never overwritten).
3. **The migration runner owns the schema version.** `schema_migrations` is the source of
   truth; `index_state.schema_version` is mirrored from it by the store after migrating and no
   migration script edits it (a test asserts that).
4. **`IngestionJob.available_at`** is the time before which a job must not be claimed: the
   creation time for a new job, the retry time after a retryable failure, the re-queue time
   after a lease expired.
5. **`duplicate_of`** marks documents whose content is identical to another document of the
   same source; the rules are in [ADR 0008](0008-parsing-and-chunking-contracts.md).
6. **Version rows hold document facts only.** Vectors and norms belong to embeddings, not to
   `document_versions`.

## Consequences

- Phase 3 produces `normalized_text` (zlib) and the first versions' content; the commit that
  stores them arrives with schema v3.
- `rag status --json` lists unversioned documents under `awaiting_first_ingest`.
