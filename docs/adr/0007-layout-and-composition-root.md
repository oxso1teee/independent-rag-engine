# 0007. Layout, composition root and boundaries

Status: Accepted (2026-11-22).

## Context

The composition root must be the single place that reads configuration and builds adapters, the
application layer must not know infrastructure, and file and test directory names must stay
predictable as the product grows.

## Decisions

1. **Composition root at `rag_engine/container.py`.** It reads configuration, builds the
   adapters (SQLite store, system clock, entropy, filesystem source) and is the entry point
   (`rag = "rag_engine.container:main"`). The command line (`cli/`) receives a `Container` (a
   protocol in `cli/context.py`) and imports no infrastructure. Layer order: `container > cli >
   ingestion > storage | system | testing | observability | config > ports > domain`.
   `cli.main(argv, container=...)` requires a container, so tests inject one with a fake clock.
2. **Storage files:** `storage/sqlite/core.py` (store, opening, migration, index-state
   bootstrap), `storage/sqlite/uow.py` (unit of work), `connection.py`, `repository.py` (SQL),
   `codec.py` (row mapping), `storage/backup.py`, `storage/migrations/`.
3. **Test directories and markers:** `tests/unit`, `contract`, `architecture`, `integration`,
   `e2e`, `chaos`, `integrity`, `perf`.
4. **Performance budgets are not part of the pull-request CI.** The PR CI runs
   `-m "not perf"`; budgets are measured separately (`make perf`) before a phase is closed and
   the results are recorded in `docs/TESTING.md`.

## Design choices worth knowing

| Choice | Reason |
|---|---|
| Writer lock file is `<database>.writer.lock` | two databases in one directory must not block each other |
| Readers use one connection per thread; the writer connection is used by whichever thread holds the lock | one writer and concurrent readers without a dedicated thread; simple to test |
| Store ports are `MetadataReader`, `MetadataWriter`, `UnitOfWork`, `Store` | read-only access cannot mutate state |
| `--json` and category exit codes exist from Phase 2 | the CLI needs stable contracts to be testable |
| `ingestion` may import `rag_engine.observability` although that wraps `structlog` | logging goes through the project's own wrapper; third-party modules stay forbidden as direct imports |
