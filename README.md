# Independent RAG Engine

Modular, local-first RAG engine for reliable knowledge retrieval, evidence-grounded
generation, and reproducible evaluation.

> **Status: early development.** The repository contains the core foundation (domain model,
> error taxonomy, ports, configuration, logging, test doubles, CI) and document
> ingestion: a SQLite store, source discovery, change detection, safe deletion and a
> persistent job queue, driven by the `rag` command. Parsing, chunking, embeddings,
> retrieval and generation are built on top in later phases.

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

## Quick start

```bash
git clone https://github.com/oxso1teee/independent-rag-engine.git
cd independent-rag-engine
uv sync --frozen
uv run rag --version
make check          # lint, types, architecture, tests
```

Register a folder and see what ingestion finds (documents that need processing are queued),
then preview how a document is parsed and split into chunks (nothing is stored):

```bash
uv run rag source add notes ~/Documents/notes
uv run rag ingest --dry-run
uv run rag ingest
uv run rag status
uv run rag chunks ~/Documents/notes/guide.md
uv run python eval/scripts/chunk_grid.py   # chunk shapes at 192/256/384 tokens; ranks nothing
```

The installed environment (including development tools) is about 190 MB; the runtime
dependencies alone take about 60 MB.

## Repository layout

```
src/rag_engine/
  domain/          identifiers, models, invariants, error taxonomy   (pure, no I/O)
  ports/           Protocol contracts: sources, parsing, chunking, embeddings, storage,
                   indexes, reranking, LLM, tokens, clock
  ingestion/       filesystem source, change detection, identity, lifecycle, job queue, MIME
  storage/         SQLite store (WAL, single writer, unit of work) and SQL migrations
  system/          real clock and identifier entropy
  config/          validated settings and resource profiles
  observability/   structured logging, request correlation, timing
  testing/         deterministic fakes (embedder, LLM, clock, in-memory source)
  cli/             the `rag` command (receives its collaborators, no infrastructure)
  container.py     composition root and entry point
tests/             unit, contract, architecture, integration, e2e, chaos, integrity and perf tests
docs/              architecture and development notes
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data model](docs/DATA_MODEL.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)
- [Changelog](CHANGELOG.md)
- [Architecture decision records](docs/adr/README.md)

## Configuration

Settings come from an optional TOML file and `RAG_*` environment variables
(`RAG_LLM__CONTEXT_WINDOW=8192`); environment values win. See
[`rag.example.toml`](rag.example.toml). Secrets are referenced by the *name* of an
environment variable (`api_key_env`) and never stored in files.
