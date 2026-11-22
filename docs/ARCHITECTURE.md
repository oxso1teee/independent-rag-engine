# Architecture

Independent RAG Engine is a **modular monolith** built as **hexagonal architecture**:
a pure core talks to the outside world only through `Protocol` ports, and concrete
adapters are injected at a single composition root.

## Layers and the dependency rule

```
cli                                                       application entry points (thin)
ingestion                                                 use cases of the ingestion pipeline
storage | system | testing | observability | config       adapters and cross-cutting services
ports                                                     Protocol contracts
domain                                                    identifiers, models, invariants, errors
```

A layer may import only from layers below it; layers on the same row are independent.
`ingestion` additionally may not import `storage`, `system`, `sqlite3`, `structlog`,
`pydantic` or the CLI framework: it receives its store, clock and adapters through
constructors and sees them only as ports. Later pipeline modules (parsing, chunking,
embedding, retrieval, rerank, context, generation, orchestrator, evaluation) follow the
same rule. `cli` is the composition root.

The rules are executable: [`.importlinter`](../.importlinter) declares them and CI runs
`lint-imports`. They also forbid the domain from touching I/O or third-party libraries,
and the core from importing the CLI or the test doubles. A negative test proves that
violations make the linter fail.

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
| `user` | `E_FILTER_INVALID` |

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

SQLite is the system of record ([data model](DATA_MODEL.md)).

- **Connections.** One connection per reading thread (`query_only`, never blocked by a
  writer thanks to WAL) and one writer connection. WAL mode is required: the store refuses
  to open where it is unavailable (`E_RESOURCE_GATE`), as it does for SQLite older than 3.35.
  `busy_timeout` is 5 s; every connection is closed by `SqliteStore.close()`.
- **Single writer.** A unit of work first takes an exclusive `flock` on
  `<database>.writer.lock` and then runs `BEGIN IMMEDIATE`. A second writer, in this or
  another process, gets `E_STORE_LOCKED` (retryable); `writer_timeout_s` lets a caller wait
  instead. The operating system releases the lock when the holder dies.
- **Unit of work.** `Store.unit_of_work().begin()` yields the only object that can change
  data. It commits on normal exit and rolls back on any exception; nesting is refused. The
  store never reads the clock: timestamps are passed in.
- **Migrations.** Numbered SQL files in `storage/migrations/`, applied in order, each in its
  own transaction together with its `schema_migrations` row (version, name, SHA-256 checksum,
  time). Applied migrations are re-verified on every start: a changed script is
  `E_INTEGRITY`, a database from a newer engine is `E_MIGRATION_REQUIRED`, a failing script is
  rolled back and named in the error.
- **Read-only open.** `open_store_read_only` never migrates, creates files or takes the writer
  lock; `--dry-run` uses it.
- **Backup.** `backup_database` runs `VACUUM INTO` a temporary file, verifies
  `integrity_check` and per-table row counts, and only then moves it into place.

## Ingestion (`rag_engine.ingestion`)

One scan, `IngestService.run(source)`:

1. `FilesystemSource.scan` lists the files below the root: sorted, NFC keys, include/exclude
   globs, symlinks skipped unless the policy follows them (and then only inside the root),
   oversized files skipped, unreadable directories reported as unreadable *subtrees*.
2. `ChangeDetector` compares each entry with its `SyncState`. If `(size, mtime)` equals the
   stored signal the file is not read; otherwise it is hashed (SHA-256) and its MIME type
   detected from the content. Result per entry: `unchanged`, `new`, `modified`, `deleted` or
   `inaccessible`.
3. Moves are recognised when exactly one vanished and one new entry share a content hash
   that no other document holds; the `DocumentId` is kept. Anything ambiguous stays separate.
4. The mass-delete guard runs on the vanished set (below).
5. If anything changed, one unit of work writes documents, sync states, tombstones and the
   `ingest_doc` jobs together. If nothing changed, no transaction is opened.
6. Tombstones older than the source's grace period are garbage-collected.

Parsing, chunking and embedding are later phases; the queued jobs wait for them.

**Identity.** A document is `(source, key)`; its `DocumentId` never changes while the key
exists. A source's root is part of its identity: registering the same name with another root
is `E_SOURCE_LOCATOR_CHANGED`, and the only way to move a source is `rag source rebase`
(previewable with `--dry-run`), which keeps ids for keys that still exist and for unique
content matches.

**Mass-delete protection.** Nothing is changed, and `E_MASS_DELETE_GUARD` is raised, when a
scan finds no documents at all while some are tracked, or when at least two tracked
documents (more than 20 %) vanished at once. An unreadable directory never produces
deletions, and a missing root is `E_SOURCE_NOT_FOUND`. `--allow-mass-delete` overrides the
guard. Deletions are tombstones (`missing`): reversible while the grace period
(`delete_grace_days`, default 30) runs; a reappearing file returns with the same id.

**Job queue.** `JobQueue`/`Worker` over the `jobs` table: idempotent enqueue, lease with
heartbeat, retry with exponential backoff (30 s, 60 s, ... capped at 1 h), at most 5 attempts,
then `dead`. Non-retryable errors go `dead` immediately, so a poison file never blocks the
queue. Expired leases are recovered on every claim; a worker that lost its lease is fenced
off by the attempt number.

### Assumed parameters

The specification documents available to this repository state only the 20 % threshold.
These values are therefore assumptions, kept as named constants and covered by tests:

| Parameter | Value | Where |
|---|---|---|
| Mass-delete ratio / minimum count | > 20 % and ≥ 2 documents | `ingestion/lifecycle.py` |
| Grace period | 30 days (per source) | `SourcePolicy.delete_grace_days` |
| Maximum file size | 50 MiB (per source) | `SourcePolicy.max_file_bytes` |
| Job attempts / backoff / lease | 5 / 30 s × 2, cap 1 h / 300 s | `ingestion/jobs.py` `RetryPolicy` |
| Index state placeholders | `none` until schema v2 | `storage/sqlite/store.py` |

## Command line

`rag source add|list|rebase`, `rag ingest [--dry-run]`, `rag status`; `--json` prints one JSON
document. Commands contain no logic: they call services obtained from the injected
`Container`. Exit codes are listed in `cli/exit_codes.py` (`3` configuration, `4` source not
found, `5` root changed, `6` mass-delete guard, `7` store locked, `8` migration required,
`9` integrity, `10` resource, `11` unsupported format; `2` usage; `1` unexpected).
