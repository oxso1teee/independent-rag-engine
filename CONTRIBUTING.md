# Contributing

Thank you for looking. The project is small and strict; this page says what a change needs.

## Setup

```bash
uv sync --frozen --extra api
make check            # ruff, mypy (strict), import boundaries, tests: all offline
```

Models are never needed to run the tests; real-model tests are opt-in (`RAG_TEST_E5_DIR`,
`RAG_TEST_RERANK_DIR`, `RAG_TEST_OLLAMA_CHAT_MODEL`) and skip themselves when the memory is not there.

## What a change needs

1. **Tests first.** A behaviour change comes with a test that fails without it; a bug fix with the
   test that reproduces it. Pure logic is unit-tested, ports have contract tests, flows have e2e tests.
2. **The gates pass:** `ruff check`, `ruff format --check`, `mypy`, `lint-imports` and the tests, on
   Python 3.12 and 3.13. The import boundaries ([ARCHITECTURE.md](docs/ARCHITECTURE.md)) are
   enforced, not advisory.
3. **Small and reversible.** Reuse what exists; add a dependency only when the work needs it, with
   its licence checked (`THIRD_PARTY_NOTICES.md` is generated). One concern per pull request.
4. **Decisions are written down.** A choice with a trade-off gets an ADR in `docs/adr/`; a
   measurement gets its numbers, its conditions and its limits in the docs, never only in a message.
5. **No claim without evidence.** Do not report a result that was not run, do not mark an evaluation
   label as reviewed unless a person reviewed it, and do not weaken a criterion to make it pass: say
   it is open.

## Workflow

Branch from `main`, commit with a conventional message (`feat:`, `fix:`, `docs:`, `test:`,
`chore:`), open a pull request, wait for all CI jobs, squash-merge. History is not rewritten.

## Security

Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in public issues.
