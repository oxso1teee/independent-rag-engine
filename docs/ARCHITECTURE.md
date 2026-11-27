# Architecture

Independent RAG Engine is a **modular monolith** built as **hexagonal architecture**:
a pure core talks to the outside world only through `Protocol` ports, and concrete
adapters are injected at a single composition root.

## Layers and the dependency rule

```
container                                                 composition root: configuration, adapters, entry point
cli                                                       command line (thin)
evaluation                                                metrics, datasets and runners that measure the pipeline
pipeline                                                  query orchestration: stages, time budget, final status
generation                                                answer generation: prompt, model call, citation checks
ingestion | retrieval | context                           use cases: the ingestion pipeline, search and context assembly
storage | system | testing | observability | config | parsing | chunking | embedding | llm
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
- Optional reranking (`RerankStage`, `OnnxCrossEncoder`) reorders a pool larger than `top_k` and
  can never fail a search: any problem keeps the fused order and reports `rerank_skipped:<reason>`
  ([ADR 0014](adr/0014-reranker-off-by-default.md)). `SufficiencyGate` compares the best dense
  cosine or reranker logit with thresholds calibrated per embedding model.
- `rag_engine.evaluation` (above `ingestion` and `retrieval` in the layering) holds the metrics,
  seeded bootstrap intervals, the dataset schema and the runner that scores rankings.

## Context (`rag_engine.context`)

`ContextBuilder.build(hits, question_tokens, system_tokens, limits, policy)` returns a `Context`:
numbered `Evidence` plus a `ContextReport`. It reads chunk content through the
`ChunkContentReader` port and counts tokens with the same `TokenCounter` as the rest of the
pipeline.

- **Budget.** `evidence_budget = window - system - question - max_new - margin`, every term
  measured, `margin = max(256, 10 % of the window)`; each passage also pays a per-item overhead for
  its marker and locator line. A prompt that leaves no room, or a budget that cannot hold the
  required evidence, is `E_CONTEXT_BUDGET` with a hint (smaller `top_k`, larger window, other model).
- **Selection**, best hit first: a missing or blank chunk is dropped (`missing`); an exact repeat
  of a text is dropped (`duplicate`); an overlapping neighbour of the same version is folded into
  the exact union of the two slices (`merged`) or dropped when it adds nothing (`overlap`); a
  document may hold at most `max_chunks_per_document` places (`doc_cap`); what does not fit the
  remaining budget is dropped (`budget`), except the top hit, which is cut at a sentence end and
  marked `truncated`. With small-to-big a leaf is replaced by its section when the section fits;
  the evidence then lists the retrieved leaves it stands for.
- **Accounting.** Every retrieved chunk is shown, merged into a shown one, or dropped with a
  reason; nothing is lost silently (NFR-07). The report carries tokens used against the budget and
  a hash of the evidence and the budget, so a context can be compared and logged.
- **Provenance.** Evidence text is an exact slice of the normalised document text; the locator
  gives source, key, title, heading path, pages and character span, which is what citation
  validation in the generation phase checks against.

## Generation (`rag_engine.generation`, `rag_engine.llm`)

`AnswerGenerator.answer(request_id, question, context)` turns a `Context` into a
`GeneratedAnswer` that is either **answered** (text plus resolved citations) or **refused**. The
prompt is `answer.v1`: a pinned system prompt, the numbered evidence (`[E1] source key`, then the
passage) and the question; marker-like text inside a document is rewritten to `(E1)` so a document
cannot pose as evidence, and the question is flattened to one line. The model refuses by replying
with exactly the sentinel `<<INSUFFICIENT>>`, which is language-neutral and never mixed with prose.

- **Before the call.** The prompt and the answer allowance are measured with the provider's own
  counter; if they do not fit its window the call is not made (`E_CONTEXT_BUDGET`). An empty
  context is refused without calling the model.
- **After the call.** A prompt the provider reports as exactly as long as its window was cut
  silently and is `E_CONTEXT_TRUNCATED`; a gap of more than 15 % between the counted and the
  reported prompt tokens is a `token_count_drift` warning; a reply cut by the length limit carries
  `answer_truncated`.
- **Degenerate replies.** Fewer than `min_answer_chars` characters (20 by default, as specified;
  citation markers excluded; `[answer] min_answer_chars`), fewer than two letters or digits, a lone
  marker, a loop of repeated words, or the sentinel mixed with prose is degenerate. It gets one retry at
  temperature 0.3, then `E_DEGENERATE_ANSWER`.
- **Citations.** Level 1: every `[E<n>]` must name evidence of this context and an answer must
  cite something; otherwise one retry, then `E_CITATION_INVALID`. A source that was not provided
  can therefore never reach the caller. Level 2: each cited sentence must share at least half of
  its content terms with the evidence it cites. This is a **lexical overlap screen, not a check
  that the answer is true or entailed by the evidence**: a paraphrase that reverses the meaning
  passes and a correct paraphrase in different words can fail. Sentences that cannot be judged
  (too few terms, or written in another script than the evidence, as when a Russian answer cites
  an English passage) are reported as unjudged, never as failures. A low overlap is reported by
  default and fails the answer only with `strict_overlap`. Entailment-level faithfulness is
  measured in the evaluation phase, not asserted here.
- **Streaming.** `stream()` yields `EvidenceEvent`, `DeltaEvent`*, `CitationsEvent`, `DoneEvent`.
  Citations can only be checked on the finished text, so live deltas are **provisional**
  (`DeltaEvent.provisional`): if validation fails, or the provider fails after text was sent, a
  `RetractEvent` is emitted and then the error is raised, and a client must discard the draft.
  `DoneEvent.answer` is the only validated text. With `verified_only=True` the reply is collected
  and validated first and sent once as a non-provisional delta, so unvalidated text never reaches
  the client, at the price of the first-token latency. A refusal never streams text, because the
  start of the reply is held back while it could still be the sentinel. A stream reports no usage,
  so its token counts are estimated (`usage_estimated`); the Ollama adapter itself fails a stream
  whose closing event shows a cut prompt.

`ResilientLLM` wraps any provider: a per-call timeout (`E_PROVIDER_TIMEOUT`, not retried), bounded
retries with backoff only for errors that are `retryable` and are "unreachable/5xx" or "slow
down", a circuit breaker (three consecutive failures open it for 60 s, then one trial call), and
serialised calls (`limits.llm_concurrency`, 1 by default). A cancelled call frees its slot and is
neither success nor failure; a stream is retried only before its first delta.

The adapters in `rag_engine.llm` are `OllamaChat` (native `/api/chat` with `num_ctx`, `num_predict`,
`think: false`, `prompt_eval_count`, `done_reason`) and `OpenAICompatChat` (`/chat/completions`,
mandatory `context_window`, key in the `Authorization` header only and removed from error
excerpts). Ollama has no tokenizer endpoint, so `count_tokens` uses the same deliberately high
heuristic counter as the context builder. An out-of-memory failure is `E_RESOURCE_GATE` and is never
retried. `build_chat` in the composition root reads `[llm]`; the API key is read from the
environment variable named by `api_key_env`, never from the file.

### Resource budget of the query path

The target machine has about 7.8 GiB, shared with a desktop, and often 1.5 to 3.7 GiB available.
Measured or file-size figures for the components a query can load:

| Component | Resident | Source |
|---|---:|---|
| E5-small, ONNX, in the CLI process | 1.23 GiB peak | `EMBEDDING_BAKEOFF.md` |
| Cross-encoder (`bge-reranker-v2-m3-int8`), in the process, off by default | 2.5 GiB peak | ADR 0014 |
| Chat model through Ollama (`qwen3.5:4b`) | 3.1 GiB file + cache and projector, **not measured** | `ollama list`; no real run was possible, see `TESTING.md` |

Together they exceed what is usually free, so the rules are: Ollama's `keep_alive` defaults to `0`
(the model is unloaded as soon as the answer is done, so the next query's embedder never competes
with a resident model); configuration is **refused** when the reranker is enabled together with an
Ollama model that stays loaded (`keep_alive` other than `0`); and a pipeline releases the
in-process models before it calls a local generator (Phase 8). `keep_alive = "5m"` is therefore
valid only without the reranker and on a machine that has the memory for an embedder beside the
resident model, which the owner decides; it was the first default and was withdrawn because
the sum (about 1.2 + 3.5 GiB without the reranker) already exceeds the memory usually available.
The price of `0` is a cold model load (a few seconds) on every question.

## Query pipeline (`rag_engine.pipeline`)

`RagPipeline.ask(query)` and `.stream(query)` run retrieval, context assembly, the sufficiency gate
and generation under **one time budget** and always end in one of four statuses. There is no fifth
way out: stage failures are statuses, only cancellation propagates.

| Status | Meaning | Text | Citations | Model called |
|---|---|---|---|---|
| `answered` | the model answered and every citation resolves to evidence of this request | yes | at least one | yes |
| `insufficient_evidence` | no hits, an empty context, the gate refused, or the model said the evidence lacks the answer | no | none | not when the gate or the search settled it |
| `unverified` | the model produced text but its citations stayed invalid after one retry | no (kept apart as `raw_text`) | none | yes |
| `error` | a stage failed; `error.code` and `error.stage` say which (`E_TIMEOUT`, `E_CONTEXT_BUDGET`, `E_DEGENERATE_ANSWER`, provider errors) | no | none | maybe |

The table is enforced by `PipelineAnswer` itself: an `answered` result without a real answer (at
least two letters or digits; the generator additionally enforces the 20-character default),
without a citation, or citing evidence that is not among its sources cannot be built
(NFR-08, tested on one-character, empty, punctuation-only and marker-only replies).

- **Time.** `TimeBudget` splits the total allowance (`[answer] total_timeout_s`): retrieval may use
  at most its share (25 % by default), context assembly its share (5 %), generation whatever is
  left. Time comes from the injected monotonic clock. The budget is cooperative: it bounds every
  await and is checked between stages, but cannot interrupt a synchronous call that never yields.
- **Gate.** The sufficiency gate reads the calibration stored in the index on every request; with
  none stored it lets the question through and says `uncalibrated`. A refusal by the gate answers
  without any model call (tested with a spy: zero calls).
- **Streaming.** `stream(query, verified_only=True)` sends evidence, then the validated text once,
  then citations, then one `FinalEvent`. `verified_only=False` streams provisional drafts and
  announces a failed validation with `RetractEvent` before the final event. A stream never raises
  a pipeline failure; it ends with a final event of status `unverified` or `error`.
- **Concurrency and cancellation.** `limits.llm_concurrency` (1 on the `low` profile) bounds
  requests in flight. A cancelled request frees its slot and the model provider's slot and is not
  an error status; the other requests are unaffected (tested with two concurrent requests, one
  cancelled).
- **Memory.** In one-shot use (`rag ask`) the in-process embedder and reranker are released before
  a local model is called (`[answer] release_models`). Releasing is not assumed to free memory:
  measured on the real E5-small model (`eval/scripts/release_memory.py`, one run), dropping the
  model and collecting returned 383 of 1160 MiB of the process' resident set, and
  `malloc_trim` after it returned 760 MiB; the hand-over therefore also calls `trim_heap()`. The
  cross-encoder was not measured (loading it needs about 2.5 GiB). A long-running service keeps its
  models and does not release them.
- **Observability (NFR-07).** `pipeline.context` logs how many chunks were included, cut, dropped
  and merged; the same counts are in `diagnostics.context`. Every pipeline event carries the
  `request_id`; document text, questions and answers are never logged.

`rag ask QUESTION [--stream [--live]] [--json] [--sources] [--show-unverified]` is a thin command
over the pipeline. Exit codes: 0 answered; 10 insufficient evidence; 11 unverified; other values
follow the error category (3 configuration, 6 provider, 7 resource, ...). `--stream --json` prints
one event per line, ending with `final`.

## Command line

`rag source add|list|rebase` (`add` takes repeatable `--cleaner` line patterns),
`rag ingest [--dry-run] [--accept-mass-delete]`, `rag status`, `rag chunks DOC [--source NAME]
[--cleaner RE] [--strategy S] [--full]` (parse and chunk one document, nothing stored);
`--json` prints one JSON document (errors included). Commands contain no logic: they call
services from the injected `Container`. Exit codes follow the error category: `0` ok, `1`
unexpected, `2` user/usage, `3` config, `4` source, `5` parse, `6` provider, `7` resource,
`8` storage, `9` integrity; the exact error code is in the output.
