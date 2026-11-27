# Development

## Setup

```bash
uv sync --frozen        # reproducible environment from uv.lock
```

Changing dependencies: edit `pyproject.toml`, run `uv lock`, commit `uv.lock`. CI runs
`uv lock --check`. Add a dependency only when the current phase needs it.

## Commands

| Command | Purpose |
|---|---|
| `make lint` | Ruff lint and format check |
| `make format` | Apply Ruff fixes and formatting |
| `make typecheck` | Mypy in strict mode (source and tests) |
| `make arch` | import-linter contracts |
| `make test` | Unit tests only (`pytest -m unit`) |
| `make test-all` | All tests except `perf`, with coverage and the domain coverage gate (90 %) |
| `make perf` | Performance budgets (NFR-04), run separately |
| `make audit` | Lock check and `pip-audit` |
| `make check` | Everything above except the audit |

## Using the command line

```bash
uv run rag source add notes ~/Documents/notes --exclude '.git/**'
uv run rag ingest --dry-run      # what would change; writes nothing
uv run rag ingest                # scan and queue the documents that need processing
uv run rag ingest --accept-mass-delete   # confirm a disappearance the guard stopped
uv run rag status                # documents per state, job queue
uv run rag source rebase notes /new/location --dry-run
uv run rag source add wiki ~/wiki -C '^Last edited:.*$'   # drop matching whole lines from every text
uv run rag chunks ~/wiki/page.md          # parse and chunk one file; writes nothing
uv run rag chunks --source wiki page.md   # a document of a source, with its cleaners applied
uv run rag chunks page.md --strategy recursive --full --json
uv run rag search "query text" --mode hybrid --top-k 5 --source wiki --tag draft --json
uv run rag search "query text" --language ru --kind table --modified-after 2026-01-01 --debug
uv run rag context "query text" --window 4096 --max-new 512   # evidence, token budget, what was left out
```

The database path comes from `[storage] db_path` in `rag.toml` (default `data/rag.db`),
`RAG_STORAGE__DB_PATH` or `--db`. Queue retry and lease parameters are the `[ingest]` section.
Add `--json` to any command for machine-readable output.

`rag chunks` runs the parser and the chunker on one document and prints the chunks (`--full` for
whole texts, `--cleaner/-C` for extra line cleaners, `--strategy` to override
`[chunking] strategy`). It reads the `[parsing]` and `[chunking]` sections of `rag.toml`; PDF,
DOCX and HTML are parsed in an isolated worker process unless `[parsing] isolate = []`.
`eval/scripts/chunk_grid.py` prints how a corpus is cut at several target sizes (counts, token
distribution, overlap, time) without ranking them. Storing versions and chunks, embedding and
search are later phases: queued jobs stay queued for now.

## Tests

See [TESTING.md](TESTING.md). Short version: `tests/unit` is pure logic; `tests/contract` has
one suite per port, run against every implementation (add new adapters to the factory table
in the corresponding file); `tests/architecture` proves the import contracts reject
violations; `tests/integration`, `tests/e2e` and `tests/performance` use a real SQLite file
and real directories under `tmp_path`.
- No test uses the network (`pytest-socket`) or a model. Hypothesis runs with a fixed,
  derandomised profile, so failures reproduce.
- Time and identifiers are injected (`FakeClock`, `Uuid7Generator`); never call the
  system clock from domain or application code.
- Write the failing test first, then the implementation.

## Conventions

- Immutable domain objects (`frozen=True, slots=True`); return new objects instead of
  mutating.
- Raise `RagError` with a stable code for every expected failure; never swallow errors.
- Keep log fields free of document text, prompts and secrets (see `observability`).
  Event names are constants (`ingest.document_parsed`); put variable data in fields,
  never into the event name or an exception message that may be logged.
- Architecture decisions are recorded in `docs/adr/`.
- Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
