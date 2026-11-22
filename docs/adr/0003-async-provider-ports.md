# 0003. Asynchronous provider ports, synchronous storage ports

Status: Accepted (2026-11-22)

## Context

Embedding and LLM providers are network- or model-bound, need timeouts, cancellation,
streaming and bounded concurrency (semaphores). Storage and index ports wrap SQLite and
in-process computation; SQLite is a synchronous driver with a single writer.

## Decision

- `EmbeddingProvider` and `LLMProvider` are asynchronous (`async def`; `LLMProvider.stream`
  returns an `AsyncIterator`).
- `SourceAdapter`, `Parser`, `Chunker`, `Store`, `VectorIndex`, `LexicalIndex`, `Reranker`,
  `TokenCounter` and `Clock` are synchronous. Services that need them from async code call
  them through a bounded worker pool (`run_in_executor`).

This is consistent with the architecture: asyncio for the API and network providers,
a synchronous SQLite driver in a dedicated writer thread plus reader pool, CPU-bound work
in a bounded executor.

## Consequences

- Provider adapters can use native async HTTP clients and cancellation.
- Synchronous adapters (for example an in-process embedder) wrap blocking work themselves.
- Contract tests drive async ports with `asyncio.run`; unit tests allow only local
  (AF_UNIX) sockets, which asyncio needs; network sockets stay blocked.
- Making another port asynchronous later would be a breaking change and needs a new record.
