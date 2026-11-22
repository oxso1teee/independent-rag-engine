# Testing

All tests run without network access (`pytest-socket`), without models and without real AI
providers. Time and identifiers are injected (`FakeClock`, `make_ids`); Hypothesis uses a
fixed, derandomised profile.

| Directory | Marker | What it proves |
|---|---|---|
| `tests/unit` | `unit` | Pure logic: domain models and invariants, glob rules, MIME table, errors, config, CLI basics |
| `tests/contract` | `contract` | One suite per port, run against every adapter (`SourceAdapter` runs against the in-memory and the filesystem source) |
| `tests/architecture` | `architecture` | The import contracts reject violations |
| `tests/integration` | `integration` | Real SQLite and real directories: unit of work, migrations, WAL and single-writer behaviour, change detection, identity and rebase, mass-delete protection, job queue, crash recovery, repository operations |
| `tests/e2e` | `e2e` | The `rag` command in-process (injected container, fake clock) and as a real process |
| `tests/performance` | `performance` | NFR-04 on 1000 files, measured |

## Failure injection and recovery

- A child process is killed with `os._exit` in the middle of a transaction
  (`test_state_survives_an_interrupted_transaction`) and in the middle of a job
  (`test_a_crash_in_the_middle_of_processing_is_recovered_after_restart`); the next process
  finds consistent state, a released lock and the job still leased, recovers it after the
  lease expires and processes it exactly once more.
- A migration that fails halfway is rolled back; a tampered migration is detected.
- Handlers that raise, workers that lose their lease, and jobs that crash their worker until
  they are declared dead.

## NFR-04

`tests/performance/test_nfr04_rescan.py` ingests 1000 files, drains the queue through a spy
parser, then rescans. It asserts: at most 5 s, 0 parser calls, 0 jobs, and an identical hash of
the logical database dump (`tests/workspace.py: logical_dump`). Run it with
`uv run pytest tests/performance -s` to see the measured times.

## Running

```bash
make test        # unit tests only
make test-all    # everything, with coverage and the domain coverage gate (90 %)
make check       # lint, types, architecture, all tests
```

Permission-based tests are skipped when run as root (root ignores file modes).
