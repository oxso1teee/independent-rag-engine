# Data model

State of the model as implemented through Phase 2 (schema version 2). Domain objects are
frozen dataclasses that validate their own invariants; SQLite holds their persistent form
(`storage/migrations/0001_core.sql` and `0002_align_data_model.sql`, mapping in
`storage/sqlite/codec.py`). The rules behind documents, versions and jobs are in
[ADR 0005](adr/0005-documents-versions-and-schema-v2.md).

Identifiers are typed UUIDv7 values (`SourceId`, `DocumentId`, `DocumentVersionId`, `JobId`,
`RequestId`); content hashes are SHA-256 hex. Timestamps are UTC, stored as fixed-width ISO-8601
text with microseconds, so text order is time order.

## Entities

| Entity | Purpose | Key invariants |
|---|---|---|
| `Source` | A registered origin of documents (`fs` or `inline`) | name `[a-z0-9-]{1,64}`; a filesystem source has a locator whose SHA-256 is `locator_fingerprint`; the locator changes only through a rebase |
| `SourcePolicy` | include/exclude globs, `max_file_bytes`, `follow_symlinks`, `delete_grace_days` | positive size, non-negative grace, at least one include |
| `Document` | One entry of a source, identified by `(source_id, source_key)` | `missing_since` is set exactly when `missing`; `current_version_id` is NULL only until the first commit and never returns to NULL; a current version exists and belongs to the document; it cannot duplicate itself |
| `DocumentVersion` | Immutable result of ingesting one content | `seq >= 1`; `UNIQUE (document_id, content_hash)`; rewriting an id with different content is `E_INTEGRITY`; purged text keeps no `normalized_text` |
| `SyncState` | Last observation of an entry (revision, content hash, status) | `failed` needs an error code; `next_retry_at` only when failed |
| `IngestionJob` | Unit of queued work | see the state machine below |
| `IndexMetadata` | Singleton: schema version (mirrored from `schema_migrations`), monotonic `index_generation`, placeholders for index settings | generation never decreases |

`SourceKey` is the path of an entry inside its source: relative, `/`-separated, Unicode NFC.

### Document status and "searchable"

`active` · `missing` (tombstone, reversible during the grace period) · `deleted` (user soft
deletion, reserved) · `failed` (unsupported or unreadable).

A document is created at first discovery as `active` with `current_version_id = NULL`. It is
**searchable** only when it is `active` *and* has a version (`Document.is_searchable`), so a
document that has not been indexed yet never reaches retrieval. The first successful commit sets
the first version (NULL → version); later commits replace it; it never goes back to NULL.
`rag status` shows documents without a version as "awaiting first ingestion".

### Sync status

`ok` · `changed` (change detected, processing pending) · `failed` (`E_UNSUPPORTED_FORMAT`, or
`E_SOURCE_NOT_FOUND` for an entry that could not be read) · `missing`.

### Job state machine

```
queued ──lease──▶ leased ──complete──▶ done
   ▲                │  └──fail(retryable, attempts left)──▶ failed ──(available_at)──▶ leased
   │                └──fail(non-retryable / attempts used up)──▶ dead
   └──lease expired (attempts left)──┘   lease expired (attempts used up) ──▶ dead
```

`available_at` is the time before which a job must not be claimed (creation time, retry time,
or re-queue time). `attempts` counts leases. At most one *active* job (`queued`, `leased`,
`failed`) exists per `(kind, source, document)`. The payload holds scalars only (content hash,
MIME type, detected encoding, size), never document text.

## Tables (schema v2)

`sources`, `documents` (unique `(source_id, source_key)`), `document_versions` (unique
`(document_id, seq)` and `(document_id, content_hash)`, `normalized_text` zlib BLOB,
cascade from documents), `sync_state` (primary key `(source_id, source_key)`), `jobs`
(indexes `jobs_claim`, partial unique `jobs_one_active`), `index_state` (one row, `id = 1`),
`schema_migrations`.
`documents.current_version_id` is deliberately not a foreign key (documents and versions
reference each other); the writer verifies that the version exists and belongs to the document.

Schema v3 (Phase 4) adds chunks, embedding models, embeddings and the full-text index.
`duplicate_of` exists but is filled in Phase 3 (after normalisation).

## What the store does not hold yet

Document text and vectors: Phase 2 stores metadata, hashes and queue state. `normalized_text`
is written from Phase 3 on.
