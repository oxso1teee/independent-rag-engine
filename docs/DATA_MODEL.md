# Data model

State of the model as implemented through Phase 2 (schema version 1). Domain objects are
frozen dataclasses that validate their own invariants; SQLite holds their persistent form
(`storage/migrations/0001_core.sql`, mapping in `storage/sqlite/codec.py`).

Identifiers are typed UUIDv7 values (`SourceId`, `DocumentId`, `DocumentVersionId`, `JobId`,
`RequestId`); content hashes are SHA-256 hex. Timestamps are UTC, stored as fixed-width ISO-8601
text with microseconds, so text order is time order.

## Entities

| Entity | Purpose | Key invariants |
|---|---|---|
| `Source` | A registered origin of documents (`fs` or `inline`) | name `[a-z0-9-]{1,64}`; a filesystem source has a locator whose SHA-256 is `locator_fingerprint`; the locator changes only through a rebase |
| `SourcePolicy` | include/exclude globs, `max_file_bytes`, `follow_symlinks`, `delete_grace_days` | positive size, non-negative grace, at least one include |
| `Document` | One entry of a source, identified by `(source_id, source_key)` | `active` needs a version, `pending` has none, `missing_since` is set exactly when `missing` |
| `DocumentVersion` | Immutable result of ingesting one content | `seq >= 1`; rewriting an id with different content is `E_INTEGRITY` |
| `SyncState` | Last observation of an entry (revision, content hash, status) | `failed` needs an error code; `next_retry_at` only when failed |
| `IngestionJob` | Unit of queued work | see the state machine below |
| `IndexMetadata` | Singleton: schema version, monotonic `index_generation`, placeholders for index settings | generation never decreases |

`SourceKey` is the path of an entry inside its source: relative, `/`-separated, Unicode NFC.

### Document status

`pending` (discovered, not ingested) → `active` (has a current version) · `missing`
(tombstone, reversible during the grace period) · `failed` (unsupported or unreadable) ·
`deleted` (reserved). A tombstone is physically removed only by garbage collection.

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

`attempts` counts leases. At most one *active* job (`queued`, `leased`, `failed`) exists per
`(kind, source, document)`. The payload holds scalars only (content hash, MIME type, size),
never document text.

## Tables (schema v1)

`sources`, `documents` (unique `(source_id, source_key)`), `versions` (unique
`(document_id, seq)`, cascade from documents), `sync_state` (primary key
`(source_id, source_key)`), `jobs` (indexes `jobs_claim`, partial unique `jobs_one_active`),
`index_state` (one row, `id = 1`), `schema_migrations`.
`documents.current_version_id` is deliberately not a foreign key (documents and versions
reference each other); the unit of work writes both together.

Schema v2 (Phase 4) adds chunks, embedding models, embeddings and the full-text index.

## What the store does not hold

Document text and vectors. Phase 2 stores only metadata, hashes and queue state.
