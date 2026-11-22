# Architecture

Independent RAG Engine is a **modular monolith** built as **hexagonal architecture**:
a pure core talks to the outside world only through `Protocol` ports, and concrete
adapters are injected at a single composition root.

## Layers and the dependency rule

```
cli                                  application entry points (thin)
testing | observability | config     adapters and cross-cutting services
ports                                Protocol contracts
domain                               identifiers, models, invariants, errors
```

A layer may import only from layers below it; layers on the same row are independent.
Future pipeline modules (ingestion, parsing, chunking, embedding, storage, retrieval,
rerank, context, generation, orchestrator, evaluation) will sit between `ports` and the
entry points, depend on `domain` and `ports` only, and receive collaborators through their
constructors.

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

`SourceAdapter`, `Parser`, `Chunker`, `EmbeddingProvider`, `Store`, `VectorIndex`,
`LexicalIndex`, `Reranker`, `LLMProvider`, `TokenCounter` and `Clock` are runtime-checkable
protocols with no behaviour. Network-bound providers (`EmbeddingProvider`, `LLMProvider`)
are `async`; CPU- and SQLite-bound ports are synchronous and are called from worker
threads by the services that use them ([ADR 0003](adr/0003-async-provider-ports.md)).
The store boundaries planned for Phase 2 are in [ADR 0004](adr/0004-store-boundaries-for-phase-2.md).

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
