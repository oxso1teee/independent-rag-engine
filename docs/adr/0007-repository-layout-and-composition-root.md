# 0007. Repository layout, composition root and deliberate deviations

Status: Accepted (2026-11-22).

## Context

`REPOSITORY_STRUCTURE` and `ARCHITECTURE` §2 place the composition root (`container.py`) above
the application layer and fix file and test directory names. The first Phase 2 implementation
deviated in several places. The owner asked to conform wherever feasible and to record every
remaining deviation with its reason.

## Decisions

1. **Composition root at `rag_engine/container.py`.** It is the only module that reads
   configuration and builds adapters (SQLite store, system clock, entropy, filesystem source)
   and it is the entry point (`rag = "rag_engine.container:main"`). The command line
   (`cli/`) receives a `Container` (a protocol in `cli/context.py`) and imports no
   infrastructure. Layer order: `container > cli > ingestion > storage | system | testing |
   observability | config > ports > domain`. Consequences: `cli.main(argv, container=...)`
   requires a container (tests inject one with a fake clock); `rag_engine.cli:main` is no
   longer the console entry point.
2. **Storage files** follow the structure: `storage/sqlite/core.py` (store, opening,
   migration, index-state bootstrap), `storage/sqlite/uow.py` (unit of work),
   `storage/backup.py`; connection handling, SQL and row mapping stay in `connection.py`,
   `repository.py`, `codec.py`.
3. **Test directories and markers:** `tests/perf` (`perf`), `tests/chaos` (`chaos`),
   `tests/integrity` (`integrity`) next to `unit`, `contract`, `integration`, `e2e`,
   `architecture`.
4. **NFR-04 is not part of the pull-request CI.** As in `TESTING_STRATEGY` the PR CI runs
   `-m "not perf"`. NFR-04 is measured separately (`make perf`) before a phase is closed and the
   results are recorded in `docs/TESTING.md`.

## Deviations that remain (with reasons)

| Deviation | Reason |
|---|---|
| Writer lock file is `<database>.writer.lock`, not `writer.lock` | two databases in one directory must not block each other |
| Readers use one connection per thread and the writer connection is used by whichever thread holds the lock, instead of "one writer thread + reader pool" | same guarantees (one writer, concurrent readers) without a dedicated thread; simpler to test |
| Store ports are `MetadataReader`, `MetadataWriter`, `UnitOfWork`, `Store` instead of one `MetadataStore` | requested by the owner for Phase 2 (read-only access cannot mutate) |
| `--json` and category exit codes exist in Phase 2 (the roadmap lists them in P9.6) | needed to test the CLI; they are stable contracts now |
| `ingestion` may import `rag_engine.observability` although that wraps `structlog` | logging goes through the project's own wrapper; third-party modules stay forbidden as direct imports |
