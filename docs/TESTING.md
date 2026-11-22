# Testing

All tests run without network access (`pytest-socket`), without models and without real AI
providers. Time and identifiers are injected (`FakeClock`, `make_ids`); Hypothesis uses a
fixed, derandomised profile.

| Directory | Marker | What it proves |
|---|---|---|
| `tests/unit` | `unit` | Pure logic: domain models and invariants, glob rules, MIME table (UTF-8, cp1251, koi8-r, UTF-16, binaries), guard and retry rules, exit codes, errors, config, CLI basics |
| `tests/contract` | `contract` | One suite per port, run against every adapter (`SourceAdapter` runs against the in-memory and the filesystem source) |
| `tests/architecture` | `architecture` | The import contracts reject violations |
| `tests/integration` | `integration` | Real SQLite and real directories: unit of work, migrations (including an upgrade of a real v1 database), WAL and single-writer behaviour, change detection, identity and rebase, mass-delete protection, job queue, repository operations, ingest log events |
| `tests/e2e` | `e2e` | The `rag` command in-process (injected container, fake clock) and as a real process |
| `tests/chaos` | `chaos` | `SIGKILL` at every stage (below) |
| `tests/integrity` | `integrity` | Data-model invariants and a model-based test of ingestion |
| `tests/perf` | `perf` | NFR-04 on 1000 files, measured; **not** part of the pull-request CI |

The pull-request CI runs `pytest -m "not perf"`. `make test-all` does the same with coverage.
`make perf` runs the budgets separately.

## Failure injection (`tests/chaos`)

A child process (`tests/chaos/child.py`) kills itself with `SIGKILL` at one point; the test
then checks that the database opens, `integrity_check` and `foreign_key_check` pass, the
interrupted work is entirely absent or entirely present, and a second run reaches exactly the
state of an uninterrupted run (compared on a backup copy). Points: after scan, at the start
of apply, in the middle of apply (half of the documents written), before commit, after commit
before garbage collection, in the middle of garbage collection, in the middle of a job
(lease recovered, no duplicate, no job processed twice), and in the middle of migration `0002`
(old schema intact, retried cleanly, both backups present). `tests/integration` additionally
covers a kill inside a plain transaction.

## Data integrity (`tests/integrity`)

`checks.py` holds the invariants as `check_*` functions (no orphans, version ownership,
unversioned documents are not searchable, sync state and documents agree, one active job per
document, index state mirrors the migrations). They run at teardown of every test that uses the
`workspace` fixture. `test_ingestion_model.py` is a Hypothesis state machine: random sequences
of create/modify/delete/rename/wait/rescan (including steps around the 30-day grace boundary)
are compared with a naive model of the ingestion rules after every rescan; identities must not
change. Changing the grace period in the model by one day makes it fail.

## Parsing and chunking (Phase 3)

- **Contracts.** `tests/contract/test_parser_contract.py` runs every parser (md, html, txt, pdf,
  docx) over valid samples, empty and oversized input, arbitrary bytes and mutated samples;
  `test_chunker_contract.py` runs every chunking strategy over the same documents, policies and
  random documents; `test_token_counter_contract.py` covers both counters.
- **Corpus.** `tests/fixtures/corpus_mini` holds eighteen original Russian, English and mixed
  documents (Markdown, HTML, text, a cp1251 file, PDF and DOCX). `generate_corpus_mini.py`
  recreates them byte for byte.
- **Golden snapshot.** `tests/fixtures/golden/corpus_mini.json` records, for each document, the
  parser, versions, text hash, and every chunk (level, parent, kind, tokens, offsets, text hash,
  heading path, language, pages, duplicate flag). Any change in normalisation, parsing or chunk
  boundaries is a diff. After an intended change run
  `uv run python tests/fixtures/update_golden.py` and review the diff.
- **Properties.** `tests/unit/test_pipeline_properties.py` generates Markdown, HTML and text and
  random size policies; every strategy must refuse a document with a documented error or return
  chunks that satisfy the data-model invariants, cover the text, respect `max_tokens` and are
  identical on a second run; without overlap the leaves rebuild the text.
- **Fuzzing.** `tests/integration/test_pipeline_fuzz.py` mutates every corpus file with a fixed
  seed (flipped bytes, truncation, deleted, inserted, duplicated and zeroed ranges) and checks
  the same outcome rule; the isolated worker must give the outcome of in-process parsing.
- **Isolation.** `tests/integration/test_parser_isolation.py` runs parsers that hang, crash,
  exhaust memory, recurse, print to the protocol channel, read the environment or raise.

## NFR-04

`tests/perf/test_nfr04_rescan.py` ingests 1000 files, drains the queue through a spy parser,
then rescans. It asserts: at most 5 s, 0 parser calls, 0 jobs, and an identical hash of the
logical database dump (`tests/workspace.py: logical_dump`).

Measured on 2026-11-22 with `make perf` (4 vCPU, Python 3.12.3, SQLite 3.45.1, ext4, three runs
on the code of the Phase 2 compliance PR):

| Run | First scan, 1000 files | Rescan, 1000 unchanged (service) | Rescan through the CLI |
|---|---|---|---|
| 1 | 0.315 s | 0.099 s | 0.048 s |
| 2 | 0.358 s | 0.093 s | 0.065 s |
| 3 | 0.311 s | 0.102 s | 0.068 s |

Budget: 5 s. Parser calls on rescan: 0. Jobs created: 0. Logical database dump: unchanged.

## Running

```bash
make test        # unit tests only
make test-all    # everything except perf, with coverage and the domain coverage gate (90 %)
make perf        # NFR-04, run before a phase is closed
make check       # lint, types, architecture, all tests
```

Permission-based tests are skipped when run as root (root ignores file modes).
