# 0009. Transactional index publication and schema v3

Status: Accepted (2026-11-25).

## Context

Schema v2 stores sources, documents, versions and jobs. A document can be active without a
published version. Phase 4 adds chunks, embeddings and FTS5. A version must never become current
before every required index entry is durable.

## Decision

1. Migration `0003` adds chunks, embedding models, embeddings and FTS5. Published migrations
   `0001` and `0002` and their checksums remain unchanged. The existing migrator owns the schema
   version and takes its normal pre-migration backup.
2. Parsing and embedding run outside the writer transaction. The publisher acquires the existing
   single-writer lock and opens one `BEGIN IMMEDIATE` unit of work. It rechecks the job lease and
   fencing token, source revision/content hash, document status, current version, and active model.
   A stale result cannot publish.
3. The unit of work writes model metadata and missing embeddings, the version and all chunks,
   removes old visible FTS rows, inserts the new visible FTS rows, switches `current_version_id`,
   increments `index_generation` once, and marks the job done. All steps commit or roll back
   together. FTS covers only indexed leaves of current versions of active documents.
4. Chunk IDs and `(document_id, content_hash)` prevent duplicate versions. Replaying a job that
   already published its result closes the job without changing the generation. A failure leaves
   the previous current version searchable. Expired leases are recovered by the existing queue.
5. A reader uses one SQLite snapshot for metadata and FTS. The flat vector cache is keyed by
   `(active_model_id, index_generation)` and discarded when either changes. Search adapters read
   only active documents and their current versions.
6. Retention and garbage collection use the same writer lock and transaction rules. They remove
   obsolete FTS rows and chunks before deleting unreferenced embeddings, while retaining version
   metadata with `text_purged=true`. A change to visible results increments the generation.

## Invariants

- Every visible indexed leaf has exactly one FTS row and an embedding of the active model.
- No FTS row points to an inactive, non-current, duplicate or parent chunk.
- A published version's chunks, embedding references, FTS rows and `current_version_id` agree.
- Every stored vector has finite values, the declared dimension and normalized norm tolerance.
- No chunk lacks a version, no version lacks a document, and no embedding refers to a missing model.
- The generation is monotonic and changes atomically with visible index changes.

`rag doctor --integrity` checks these invariants, SQLite integrity and foreign keys, and reports
discrepancies without silently repairing them. Recovery and deliberate repair are separate actions.

## Consequences

FTS and metadata share SQLite's WAL and transaction boundary. External provider failures cannot
leave a partially published index. A model switch requires a complete shadow population before
one transaction activates it; mixed-model visible indexes are forbidden.
