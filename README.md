# Independent RAG Engine

Modular, local-first RAG engine for reliable knowledge retrieval, evidence-grounded
generation, and reproducible evaluation.

> **Status: early development.** The repository currently contains the core foundation
> (domain model, error taxonomy, ports, configuration, logging, test doubles, CI).
> Ingestion, retrieval and generation are built on top of it in later phases.

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

## Quick start

```bash
git clone https://github.com/oxso1teee/independent-rag-engine.git
cd independent-rag-engine
uv sync --frozen
uv run rag --version
make check          # lint, types, architecture, tests
```

The installed environment (including development tools) is about 190 MB; the runtime
dependencies alone take about 60 MB.

## Repository layout

```
src/rag_engine/
  domain/          identifiers, models, invariants, error taxonomy   (pure, no I/O)
  ports/           Protocol contracts: sources, parsing, chunking, embeddings, storage,
                   indexes, reranking, LLM, tokens, clock
  config/          validated settings and resource profiles
  observability/   structured logging, request correlation, timing
  testing/         deterministic fakes (embedder, LLM, clock, in-memory source)
  cli/             the `rag` command
tests/             unit, contract and architecture tests
docs/              architecture and development notes
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Development guide](docs/DEVELOPMENT.md)

## Configuration

Settings come from an optional TOML file and `RAG_*` environment variables
(`RAG_LLM__CONTEXT_WINDOW=8192`); environment values win. See
[`rag.example.toml`](rag.example.toml). Secrets are referenced by the *name* of an
environment variable (`api_key_env`) and never stored in files.
