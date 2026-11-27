# Independent RAG Engine

Modular, local-first RAG engine for reliable knowledge retrieval, evidence-grounded
generation, and reproducible evaluation.

> **Status: early development.** The repository contains the core foundation (domain model,
> error taxonomy, ports, configuration, logging, test doubles, CI) and document
> ingestion: a SQLite store, source discovery, change detection, safe deletion and a
> persistent job queue, driven by the `rag` command, and document processing: parsers for
> Markdown, HTML, text, PDF and DOCX and three chunking strategies, previewed with
> `rag chunks`. With a pinned Ollama or local ONNX embedding model, `rag ingest` now
> publishes chunks, embeddings and an FTS5 index atomically, and `rag search` retrieves from it
> (dense, lexical or hybrid, with metadata filters). Reranking and generation are later phases.

## Design goals

- **Local-first.** A laptop-class CPU is enough; SQLite is the system of record.
- **Modular monolith, hexagonal.** A pure domain, `Protocol` ports, and replaceable adapters.
  Boundaries are enforced in CI.
- **Deterministic.** Identifiers, chunking and tests are reproducible; non-determinism is
  confined to model providers.
- **Explicit failures.** One error taxonomy with stable codes; nothing is truncated or
  skipped silently.
- **Safe by default.** Document text, prompts and secrets are never written to logs;
  secrets are read from environment variables only.
- **Testable without models or network.** Deterministic fakes ship in `rag_engine.testing`.

## Requirements

- Python `>=3.12,<3.14`
- [uv](https://docs.astral.sh/uv/)

## Supported documents

Markdown, HTML and plain text work out of the box. Optional formats are separate extras:

| Format | Extra | Notes |
|---|---|---|
| PDF | `uv sync --extra pdf` | text layer only; a scan without text is refused (`no_text_layer`) |
| DOCX | `uv sync --extra docx` | headings, lists, tables, quotes and code from styles; password-protected files are refused |
| exact token counting | `uv sync --extra tokenizers` | counts with a local `tokenizer.json` |
| ONNX embeddings | `uv sync --extra onnx` | in-process, offline; the recommended default model is `intfloat/multilingual-e5-small` ([ADR 0012](docs/adr/0012-embedding-default-evaluation.md)) |

## Quick start

```bash
git clone https://github.com/puchkovs/independent-rag-engine.git
cd independent-rag-engine
uv sync --frozen
uv run rag --version
make check          # lint, types, architecture, tests
```

Register a folder and preview parsing and chunking. With embedding enabled in the
configuration, `rag ingest` processes the queue and publishes each complete version;
without it, jobs remain queued.

```bash
uv run rag source add notes ~/Documents/notes
uv run rag ingest --dry-run
uv run rag ingest
uv run rag status
uv run rag chunks ~/Documents/notes/guide.md
uv run rag index gc                 # purge old versions and unused embeddings
uv run rag index rebuild --lexical  # rebuild FTS5 in one transaction
uv run rag doctor --integrity        # inspect SQLite, published FTS and vectors read-only
uv run rag search "how are chunks stored" --mode hybrid --top-k 5
uv run rag search "ZX4471" --mode lexical --source notes --path-glob '**/*.md' --json
# after changing the configured embedding model:
uv run rag index switch-model      # stage every visible vector, then activate atomically
uv run python eval/scripts/chunk_grid.py   # chunk shapes at 192/256/384 tokens; ranks nothing
```

### Embedding model

Embedding is off until configured; nothing is downloaded automatically. The recommended default
is `intfloat/multilingual-e5-small` (MIT, 384 dimensions, 512-token window) run offline through
ONNX Runtime; [ADR 0012](docs/adr/0012-embedding-default-evaluation.md) records the measurements
behind that choice and its limits. Provision the model files yourself, set `provider = "onnx"` and
`model_dir`, and pin `revision` to the fingerprint of the directory (see `rag.example.toml`).
Ollama models remain supported when the machine can host them.

The local development environment used for Phase 4 measurements occupied 368 MB on disk
(including optional dependencies and development tools, excluding model weights). Install size
depends on selected extras and platform.

## Repository layout

```
src/rag_engine/
  domain/          identifiers, models, invariants, error taxonomy   (pure, no I/O)
  ports/           Protocol contracts: sources, parsing, chunking, embeddings, storage,
                   indexes, reranking, LLM, tokens, clock
  ingestion/       filesystem source, change detection, identity, lifecycle, duplicates, job
                   queue, MIME, document processing (parse and chunk)
  parsing/         normaliser, cleaners, parsers (md, html, txt, pdf, docx), isolated worker
  chunking/        token counters, splitting, packing, the fixed, recursive and
                   structure-aware chunkers
  retrieval/       query processing, fusion, diversity and the search service
  evaluation/      metrics, bootstrap intervals, dataset schema and the retrieval runner
  storage/         SQLite store (WAL, single writer, unit of work) and SQL migrations
  lexical/         RU/EN Snowball analyzer shared by publication and FTS queries
  system/          real clock and identifier entropy
  config/          validated settings and resource profiles
  observability/   structured logging, request correlation, timing
  testing/         deterministic fakes (embedder, LLM, clock, in-memory source)
  cli/             the `rag` command (receives its collaborators, no infrastructure)
  container.py     composition root and entry point
tests/             unit, contract, architecture, integration, e2e, chaos, integrity and perf tests
docs/              architecture and development notes
eval/scripts/      reproducible reports (chunk grid, embedding bake-off, retrieval evaluation, benchmarks)
eval/corpus/       the mini corpus of the retrieval evaluation
eval/datasets/     versioned question sets
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data model](docs/DATA_MODEL.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)
- [Phase 4 embedding measurements](docs/EMBEDDING_BAKEOFF.md)
- [Retrieval evaluation](docs/RETRIEVAL_EVALUATION.md)
- [Reviewing an evaluation dataset](docs/DATASET_REVIEW.md)
- [Changelog](CHANGELOG.md)
- [Architecture decision records](docs/adr/README.md)

## Configuration

Settings come from an optional TOML file and `RAG_*` environment variables
(`RAG_LLM__CONTEXT_WINDOW=8192`); environment values win. See
[`rag.example.toml`](rag.example.toml). Secrets are referenced by the *name* of an
environment variable (`api_key_env`) and never stored in files.
