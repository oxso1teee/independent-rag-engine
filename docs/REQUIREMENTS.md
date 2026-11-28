# Requirements

What the engine is required to do (functional) and to be (non-functional). The numbers are the
identifiers used in tests, ADRs and the changelog (`NFR-04`, `FR-14`); [AUDIT.md](AUDIT.md) states, for
each, whether it is met and on what evidence. The non-functional targets were set before the
measurements of the retrieval and storage work and were not changed to fit them.

## Functional

| ID | Requirement | Scope |
|---|---|---|
| FR-01 | Index files from the local file system by glob policy; upload text or a file through the API | in |
| FR-02 | Identify a document stably by (source, relative key); moving a source root only by an explicit command | in |
| FR-03 | Detect changes (size and mtime, then SHA-256); do not parse or embed the unchanged again | in |
| FR-04 | Parse Markdown, text, HTML, then PDF (per page) and DOCX; extract headings, pages and metadata | in |
| FR-05 | Structure-aware chunking with `heading_path`, offsets and a parent section; fixed and recursive as fallbacks | in |
| FR-06 | Text cleaning: one normalisation, noise-removal rules, de-duplication of chunks inside a document | in |
| FR-07 | Embeddings through providers (Ollama, OpenAI-compatible, ONNX, fake); batches, a cache by (text hash, model), retries, limits | in |
| FR-08 | One transactional database for documents, versions, chunks, embeddings, full text and sync state; migrations; integrity check | in |
| FR-09 | Dense (exact cosine), lexical (BM25 with a language analyser) and hybrid (RRF) search with metadata filters | in |
| FR-10 | Cross-encoder reranking as an option with a fallback and a candidate limit; on by default only with a proven gain | in (off by default) |
| FR-11 | Context assembly with a token budget, de-duplication, ordering and a report of what was dropped or cut | in |
| FR-12 | Answer generation through LLM providers (Ollama native, OpenAI-compatible, scripted), streaming and not | in |
| FR-13 | Verifiable citations: markers and cited passages are validated | in |
| FR-14 | Refuse on insufficient evidence (a gate before the model, a refusal marker, a check after); distinguish `answered`, `insufficient_evidence`, `unverified`, `error` | in |
| FR-15 | Immutable document versions, soft deletion, garbage collection by retention | in |
| FR-16 | A command line and a REST API with OpenAPI and stable error codes | in |
| FR-17 | An evaluation framework: retrieval and generation metrics, regression gates | in |
| FR-18 | `doctor`: resource gates (memory, swap, disk), provider availability, integrity checks | in |
| FR-19 | Backup and restore as one file, with verification | in |
| FR-20 | An MCP adapter for agents | out of the first release |
| FR-21 | An ANN index (LanceDB, sqlite-vec) above about 10⁵ chunks | out of the first release |
| FR-22 | Multi-step agentic search and query rewriting | out of the first release |

## Non-functional

| ID | Area | Requirement |
|---|---|---|
| NFR-01 | Memory | an idle process at most 150 MiB; search over 10⁴ chunks at most 300 MiB; over 10⁵ chunks (1024 dimensions, fp16) at most 600 MiB |
| NFR-02 | Disk | an environment without models at most 300 MB (no torch, no document-conversion suite); a database of about 3 times the source text plus vectors |
| NFR-03 | Search latency | p95 at most 100 ms at 10⁴ chunks and 400 ms at 10⁵ (without embedding the query; 4 vCPU) |
| NFR-04 | Rescan | 1000 unchanged files in at most 5 s with no parser or embedder call |
| NFR-05 | Increment | editing one paragraph of a 100-chunk document re-embeds at most 3 chunks |
| NFR-06 | Reliability | `kill -9` at any point of ingestion leaves a database that opens, passes the integrity check and finishes on re-run |
| NFR-07 | Transparency | every truncation or dropped context item is in the answer's diagnostics and the log; no silent truncation |
| NFR-08 | Answer correctness | a one-character or empty reply is never an `answered` result |
| NFR-09 | Testability | the unit suite runs in under 60 s with no network and no model, deterministically |
| NFR-10 | Security | no non-loopback bind without a token; filters only through a typed DSL, never raw SQL; secrets only from the environment; logs without document text by default |
| NFR-11 | Reproducibility | a clean clone and `uv sync --frozen` give the same versions; the lock file is checked in CI |
| NFR-12 | Compatibility | Python 3.12 and later; Linux first, macOS best effort |
