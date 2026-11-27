# Architecture

Independent RAG Engine is a **modular monolith** built as **hexagonal architecture**:
a pure core talks to the outside world only through `Protocol` ports, and concrete
adapters are injected at a single composition root.

## Layers and the dependency rule

```
container                                                 composition root: configuration, adapters, entry point
cli                                                       command line (thin)
evaluation                                                metrics, datasets and runners that measure the pipeline
ingestion | retrieval                                      use cases: the ingestion pipeline and search
storage | system | testing | observability | config | parsing | chunking
                                                          adapters and cross-cutting services
ports                                                     Protocol contracts
domain                                                    identifiers, models, invariants, errors
```

A layer may import only from layers below it; layers on the same row are independent.
`ingestion` additionally may not import `storage`, `system`, `sqlite3`, `pydantic` or the
CLI framework directly: it receives its store, clock and adapters through constructors and sees
them only as ports (logging goes through `observability`). `parsing` and `chunking` are pure
computation: they depend on the domain and the ports only, never on each other, on storage,
configuration, ingestion or the network. Later pipeline modules (embedding, retrieval, rerank,
context, generation, orchestrator, evaluation) follow the same rule. [`container.py`](../src/rag_engine/container.py) is the only place that reads
configuration and creates adapters; the command line receives a `Container` (a protocol in
`cli/context.py`), so tests substitute fakes.

The rules are executable: [`.importlinter`](../.importlinter) declares them and CI runs
`lint-imports`. They also forbid the domain from touching I/O or third-party libraries,
and the core from importing the CLI or the test doubles. Negative tests prove that
violations make the linter fail, including new imports between `parsing` and `chunking` or
from either into storage, configuration, ingestion or the network.

## Domain

- **Identifiers** are immutable, typed value objects: `SourceId`, `DocumentId`,
  `DocumentVersionId`, `RequestId` (UUIDv7), `ChunkId`, `EmbeddingModelId`, plus
  `ContentHash` and `TextHash` (SHA-256).
- **UUIDv7** (RFC 9562) is built by a pure function from a millisecond timestamp and
  10 bytes of entropy, because Python 3.12 has no `uuid.uuid7()`. `Uuid7Generator` adds a
  counter so ids from one generator are strictly increasing, and takes its clock and
  entropy source as parameters. The function is verified against the RFC test vector.
- **`ChunkId`** is the first 128 bits of
  `SHA-256(version_id ‖ len(fingerprint) ‖ fingerprint ‖ ordinal)`, base32 encoded
  (26 characters). Reprocessing a version with the same chunker yields the same ids;
  length-prefixing keeps the encoding unambiguous. Golden values are pinned in tests.
- **Models** are frozen, slotted dataclasses that validate their own invariants on
  construction (offsets match text, hashes match text, lifecycle rules, bounds).
  Cross-entity checks live in `domain/invariants.py`.

## Errors

Categories, the code-to-category assignment and the rationale are recorded in
[ADR 0001](adr/0001-error-taxonomy-and-categories.md).

Every failure is a `RagError(code, message, retryable, context)`. Codes are stable and
listed in `domain/errors.py`; a snapshot test fails on any change.

| Category | Codes |
|---|---|
| `config` | `E_CONFIG_INVALID` |
| `source` | `E_SOURCE_NOT_FOUND`, `E_SOURCE_LOCATOR_CHANGED`, `E_MASS_DELETE_GUARD` |
| `parse` | `E_PARSE_FAILED`, `E_UNSUPPORTED_FORMAT` |
| `provider` | `E_EMBED_DIM_MISMATCH`, `E_PROVIDER_TIMEOUT`*, `E_PROVIDER_UNAVAILABLE`*, `E_RATE_LIMITED`*, `E_CONTEXT_TRUNCATED`, `E_DEGENERATE_ANSWER`, `E_CITATION_INVALID` |
| `resource` | `E_CONTEXT_BUDGET`, `E_RESOURCE_GATE`, `E_TIMEOUT`* |
| `storage` | `E_STORE_LOCKED`*, `E_MIGRATION_REQUIRED` |
| `integrity` | `E_EMBED_MODEL_MISMATCH`, `E_INTEGRITY` |
| `user` | `E_FILTER_INVALID`, `E_QUERY_INVALID` |

\* retryable by default. `context` holds scalars and nested sequences/mappings of scalars (deeply immutable,
restored intact by `copy` and `pickle`) and must never contain document
text, prompts or secrets. Violated domain invariants raise `E_INTEGRITY`.

## Ports

`SourceAdapter`, `Parser`, `Chunker`, `EmbeddingProvider`, `Store` (with `MetadataReader`,
`MetadataWriter` and `UnitOfWork`), `EmbeddingCache`, `VectorIndex`, `LexicalIndex`,
`Reranker`, `LLMProvider`, `TokenCounter` and `Clock` are runtime-checkable protocols with
no behaviour. Network-bound providers (`EmbeddingProvider`, `LLMProvider`)
are `async`; CPU- and SQLite-bound ports are synchronous and are called from worker
threads by the services that use them ([ADR 0003](adr/0003-async-provider-ports.md)).
The store boundaries are in [ADR 0004](adr/0004-store-boundaries-for-phase-2.md).

## Configuration

`config.load_settings(path)` merges, from highest priority: `RAG_*` environment variables
(nested keys with `__`), an optional TOML file, defaults. `.env` files are not a source.
Validation is strict (unknown keys are rejected) and every problem is reported as
`E_CONFIG_INVALID` naming the field and reason, never the value. The `low` resource
profile is the only supported one ([ADR 0002](adr/0002-single-resource-profile.md)); its limits (`llm_concurrency`, `embed_concurrency`, `embed_batch`,
`ingest_workers`, `max_loaded_vectors_mb`) can be overridden explicitly. An enabled
embedding provider requires `provider` and `model`; an enabled LLM additionally requires
`context_window`. Secrets are referenced by environment-variable name (`api_key_env`);
a key placed in a file is rejected.

## Observability

`structlog` with JSON output. `request_scope(request_id)` binds the request id to every
event inside it and `stage_scope(...)` adds the stage name, timing and failure codes.
Processors, in order: event-name check (names must be constants such as `stage.done`;
anything else is replaced by `log.event_redacted` with a digest and length), exception
summary (`exc_info` becomes `exception_type`, `exception_chain`, `RagError` code,
category and retryable flag, and traceback frames as `file:line in function`; the
exception message is never logged), then redaction of values under sensitive keys (text,
prompts, answers, tokens, keys, passwords, `exception`, `stack`) and replacement of any
string longer than 200 characters. Use `text_digest()` to correlate content without
logging it.

## Test doubles

`rag_engine.testing` provides `HashingEmbedder` (signed feature hashing over words and
character trigrams: deterministic, meaningful lexical similarity, no model), `ScriptedLLM`
(replays scripted replies and failures and records calls), `FakeClock`, and
`InMemorySource`. Each is verified by the contract test of its port.

## Storage (`rag_engine.storage`)

SQLite is the system of record ([data model](DATA_MODEL.md)). Files: `sqlite/core.py` (store,
opening, migration, index-state bootstrap), `sqlite/uow.py` (unit of work),
`sqlite/connection.py`, `sqlite/repository.py` (SQL), `sqlite/codec.py` (row mapping),
`backup.py`, `migrations/`.

- **Connections.** One connection per reading thread (`query_only`, never blocked by a
  writer thanks to WAL) and one writer connection. WAL mode is required: the store refuses
  to open where it is unavailable (`E_RESOURCE_GATE`), as it does for SQLite older than 3.35.
  `foreign_keys=ON`, `synchronous=NORMAL`, `busy_timeout` 5 s; `SqliteStore.close()` closes
  every connection. The database file is created owner-only (`0600`).
- **Single writer.** A unit of work first takes an exclusive `flock` on
  `<database>.writer.lock` and then runs `BEGIN IMMEDIATE`. A second writer, in this or
  another process, gets `E_STORE_LOCKED` (retryable); `writer_timeout_s` lets a caller wait
  instead. The operating system releases the lock when the holder dies.
- **Unit of work.** `Store.unit_of_work().begin()` yields the only object that can change
  data. It commits on normal exit and rolls back on any exception; nesting is refused. The
  store never reads the clock: timestamps are passed in. Writing a document checks that its
  current version exists, belongs to it, and is never withdrawn once set.
- **Migrations.** Numbered SQL files in `storage/migrations/`, applied in order, each in its
  own transaction together with its `schema_migrations` row (version, name, SHA-256 checksum,
  time). Applied migrations are re-verified on every start: a changed script is
  `E_INTEGRITY`, a database from a newer engine is `E_MIGRATION_REQUIRED`, a failing script is
  rolled back and named in the error. A script that rebuilds tables declares
  `-- migrate: foreign_keys=off`; the runner then disables foreign keys outside the
  transaction, runs `foreign_key_check` before committing and re-enables them. The schema
  version is owned by `schema_migrations`; `index_state.schema_version` mirrors it and no
  script edits it. Before migrating a database that already has a version, the runner takes a
  verified backup `<db>.pre-migration-v<from>-to-v<to>.bak` (never overwritten, owner-only).
  Schema v1 is published and immutable (its checksum is pinned in a test); v2 corrects it
  ([ADR 0005](adr/0005-documents-versions-and-schema-v2.md)); Phase 4 starts at v3.
- **Read-only open.** `open_store_read_only` never migrates, creates files or takes the writer
  lock; `--dry-run` uses it.
- **Backup.** `backup_database` runs `VACUUM INTO` a temporary file, verifies
  `integrity_check` and per-table row counts, and only then moves it into place.
- **Lexical index.** Schema v3 keeps FTS5 external-content rows for indexed leaves of
  active, current document versions. Publication, replacement and deletion update FTS in
  the writer transaction. The `snowball-ru-en/1` analyzer applies NFC, case folding,
  `ё` normalization and RU/EN Snowball stemming to both stored text and query tokens.
  MATCH expressions contain only generated quoted tokens; search uses FTS5 BM25 with
  headings weighted twice as much as body text and filters candidates before its limit.
- **Retention and rebuild.** `rag index gc` keeps the configured number of complete
  versions per document (three by default), retaining older version metadata while
  removing their text and chunks. It removes embeddings with no reference from any retained
  chunk; historical versions pin their cached vectors until their chunk data is purged.
  `rag index rebuild --lexical` replaces analyzed text and FTS rows in one writer
  transaction. `rag index switch-model` stages a candidate model's vectors in the
  content-addressed cache, verifies every visible chunk, and changes active model and
  chunk embedding references in a single transaction. Interrupted staging can be
  repeated without changing the previously active model.

## Ingestion (`rag_engine.ingestion`)

One scan, `IngestService.run(source)`:

1. `FilesystemSource.scan` lists the files below the root: sorted, NFC keys, include/exclude
   globs, symlinks skipped unless the policy follows them (and then only inside the root),
   oversized files skipped, unreadable directories reported as unreadable *subtrees*.
2. `ChangeDetector` compares each entry with its `SyncState`. If `(size, mtime)` equals the
   stored signal the file is not read; otherwise it is hashed (SHA-256) and its MIME type and
   encoding are detected from the content. Result per entry: `unchanged`, `new`, `modified`,
   `deleted` or `inaccessible`.
3. Moves are recognised when exactly one vanished and one new entry share a content hash
   that no other document holds; the `DocumentId` is kept. Anything ambiguous stays separate.
4. The mass-delete guard runs on the vanished set (below).
5. If anything changed, one unit of work writes documents, sync states, tombstones and the
   `ingest_doc` jobs together. If nothing changed, no transaction is opened.
6. Tombstones older than the source's grace period are garbage-collected.

Each stage logs `stage.start/done/failed` under one `request_id`; events contain numbers and
codes, never paths or text. Parsing, chunking and embedding are later phases; the queued jobs
wait for them. A new document is `active` with no version: it is not searchable until its first
commit.

**Identity.** A document is `(source, key)`; its `DocumentId` never changes while the key
exists. A source's root is part of its identity: registering the same name with another root
is `E_SOURCE_LOCATOR_CHANGED`, and the only way to move a source is `rag source rebase`
(previewable with `--dry-run`), which keeps ids for keys that still exist and for unique
content matches.

**Mass-delete protection.** Nothing is changed, and `E_MASS_DELETE_GUARD` is raised, when a
scan finds no document at all while some are known (`empty_scan`), when every known document
vanished (`all_known_gone`), or when more than `max(10, 20 %)` of the known documents vanished
(`threshold`). `--accept-mass-delete` confirms. An unreadable directory never produces
deletions and a missing root is `E_SOURCE_NOT_FOUND`. Deletions are tombstones (`missing`):
reversible while the grace period (`delete_grace_days`, default 30) runs; a reappearing file
returns with the same id.

**Job queue.** `JobQueue`/`Worker` over the `jobs` table: idempotent enqueue, lease with
heartbeat, retry with exponential backoff and jitter, then `dead`; non-retryable errors go
`dead` immediately, so a poison file never blocks the queue. Expired leases are recovered on
every claim; a worker that lost its lease is fenced off by the attempt number.
`available_at` is the time before which a job must not be claimed.

### Parameters

See [ADR 0006](adr/0006-ingestion-safety-and-queue.md) for the rules behind these values.
Queue parameters are settings (`[ingest]`); safety rules are fixed.

| Parameter | Value | Where |
|---|---|---|
| Mass-delete threshold | `max(10, 20 %)`, plus `empty_scan` and `all_known_gone` | `ingestion/lifecycle.py` |
| Grace period | 30 days (per source) | `SourcePolicy.delete_grace_days` |
| Maximum file size | 50 MiB (per source) | `SourcePolicy.max_file_bytes` |
| Attempts / base / factor / cap / jitter / lease | 5 / 30 s / 2 / 1 h / 20 % / 300 s | `[ingest]` |

## Parsing and chunking (`rag_engine.parsing`, `rag_engine.chunking`)

`DocumentProcessor` (in `ingestion`) reads a document within its size limit, identifies the
format from the content, picks the parser, and chunks the result; nothing is stored. The
contracts are in [ADR 0008](adr/0008-parsing-and-chunking-contracts.md).

- **One text per version.** A parser returns a `ParsedDocument`: `normalized_text` plus typed
  blocks (heading, paragraph, list item, quote, code, table) with offsets into that text, their
  heading path and, for PDF, their page. `TextAssembler` builds text, offsets and heading paths
  together, so `normalized_text[start:end]` equals the block text by construction.
- **Normalisation** is idempotent (NFC, line breaks, whitespace, invisible characters); markup
  entities are decoded once by the parser that owns the markup; per-source cleaners drop
  whole lines and are part of `normalizer_version`.
- **Parsers.** Markdown, HTML and plain text are built in; PDF (`pdf` extra) reads the text
  layer page by page, joins lines into paragraphs and drops running titles and page numbers;
  DOCX (`docx` extra) reads styles, lists, tables, quotes and code after checking the archive
  against entry, size and ratio limits. Every refusal is `E_PARSE_FAILED` with a reason
  (`corrupt`, `encrypted`, `no_text_layer`, `too_large`, `too_many_pages`, `too_many_blocks`,
  `too_deep`, `zip_bomb`, `timeout`, `memory`, `worker_failed`, `encoding`) or
  `E_UNSUPPORTED_FORMAT`.
- **Isolation.** PDF, DOCX and HTML run in one long-lived worker process by default: minimal
  environment, data-segment cap, no core dumps, JSON messages, a kill at the time limit and a
  fresh worker after any failure.
- **Token counting.** `TokenCounter` is a port with a script-aware heuristic and an exact
  counter over a local `tokenizer.json`; the counter id is part of the chunker fingerprint.
- **Chunking.** A chunk is an exact slice of the text. The packer joins sentences, lines and
  word groups up to the target size, repeats whole sentences as overlap between prose chunks,
  keeps tables and code intact when they fit and never exceeds `max_tokens`. Strategies:
  `structure_aware` (sections cut at headings, parent chunks for sections with several leaves,
  tiny sections merged), `recursive`, `fixed`; `auto` chooses by document structure. Repeated
  leaf texts inside a version are marked duplicates and not indexed. Chunk ids derive from the
  version, the fingerprint and the ordinal.
- **Duplicates between documents.** After every ingest that changed something, documents of a
  source with identical raw bytes are linked (`duplicate_of`) to the active one with the
  smallest id, in the same transaction.

Parsing and chunking are pure computation: they depend on the domain and the ports only, and
the import contracts keep them independent of each other.

## Retrieval (`rag_engine.retrieval`)

One call of `RetrievalService.search` is a deterministic function of the query and the published
index. The service sees only ports (`VectorIndex`, `LexicalIndex`, `ChunkFilter`, `ChunkCatalog`,
`EmbeddingProvider`) and an injected snapshot callable, so refreshing the vector matrix and
reading the generation stay in the composition root.

```
query text
 -> QueryProcessor    NFC, control characters, token limit (E_QUERY_INVALID), script language
 -> ChunkFilter       metadata filters to a set of searchable chunk ids (bound parameters only)
 -> dense channel     query prefix + embed_query, exact cosine over the fp16 matrix, candidate_k
 -> lexical channel   FTS5 / BM25 with the Russian/English analyzer, candidate_k
 -> fusion            RRF (k = 60, weights 1.0 / 0.5) or min-max weighted scores
 -> diversity         exact duplicate texts collapse, optional cap per document
 -> top_k hits        HitScores (dense, lexical, fused) + diagnostics
```

- Only indexed, non-duplicate chunks of active documents' current versions can be returned.
- Fusion keeps every raw score and per-channel rank; ties are broken by chunk id. The fused score
  orders results inside one query and is not a probability.
- Fallbacks are reported, not raised: `filter_matched_zero`, `no_index`, `dense_empty`,
  `lexical_empty`. A query embedded by another model than the index's is `E_EMBED_MODEL_MISMATCH`.
- The default mode is `dense` ([ADR 0013](adr/0013-default-retrieval-mode.md)); measured latency
  and the evaluation method are in [`RETRIEVAL_EVALUATION.md`](RETRIEVAL_EVALUATION.md).
- `rag_engine.evaluation` (above `ingestion` and `retrieval` in the layering) holds the metrics,
  seeded bootstrap intervals, the dataset schema and the runner that scores rankings.

## Command line

`rag source add|list|rebase` (`add` takes repeatable `--cleaner` line patterns),
`rag ingest [--dry-run] [--accept-mass-delete]`, `rag status`, `rag chunks DOC [--source NAME]
[--cleaner RE] [--strategy S] [--full]` (parse and chunk one document, nothing stored);
`--json` prints one JSON document (errors included). Commands contain no logic: they call
services from the injected `Container`. Exit codes follow the error category: `0` ok, `1`
unexpected, `2` user/usage, `3` config, `4` source, `5` parse, `6` provider, `7` resource,
`8` storage, `9` integrity; the exact error code is in the output.
