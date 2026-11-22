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
| `make test-all` | All tests with coverage and the domain coverage gate (90 %) |
| `make audit` | Lock check and `pip-audit` |
| `make check` | Everything above except the audit |

## Tests

- `tests/unit` — pure logic; `tests/contract` — one suite per port, run against every
  implementation (add new adapters to the factory table in the corresponding file);
  `tests/architecture` — proves the import contracts reject violations.
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
