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
| `tests/perf` | `perf` | NFR-04 on 1000 files and the Phase 3 parsing and chunking budgets, measured; **not** part of the pull-request CI |

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
covers a kill inside a plain transaction and an injected failure between chunk insertion and FTS
publication. A child is killed after FTS writes during indexed publication; its retry publishes
one complete version. The indexed e2e test kills a model switch inside its final transaction,
verifies that the previous model and generation remain active, then resumes the switch.

## Data integrity (`tests/integrity`)

`rag doctor --integrity` opens a read-only SQLite snapshot. It checks SQLite and foreign keys,
document/version ownership, current pointers, compressed text and chunk offsets, content hashes,
deterministic chunk IDs, the FTS row set, active-model coverage, vector dimensions, finite norms,
model IDs, and index generation. Deliberately corrupted copies exercise each class of error and
verify that inspection does not alter the database. The model-based ingestion test runs doctor
after every operation in 80 sequences of 22 steps (1760 steps total); the indexed e2e test also
checks doctor after 1000 deterministic lexical, rebuild, GC and search operations.

The one-paragraph NFR-05 scenario publishes at least 100 indexed chunks with stable section
headings, edits one paragraph, and counts actual provider inputs. It requires 1–3 new embeddings
and a sound store. These section boundaries prevent unrelated chunk boundaries from shifting.
Long unsectioned paragraphs can still shift later boundaries and exceed this budget.

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
- **Corpus.** `tests/fixtures/corpus_mini` holds twenty-one original Russian, English and mixed
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

## Phase 3 performance budgets

`tests/perf/test_phase3_pipeline.py` generates its inputs (no stored binaries) and prints one
line per measurement. The budgets are multiples (about five times, ten for the worker round
trip) of the measurements below; they exist to catch an accidental quadratic step, and like all
initial budgets they may be revised with new measurements.

Measured on 2026-11-22 with `make perf` (4 vCPU, Python 3.12.3, three runs on the code of the
hardening PR; seconds, heuristic token counter, default chunk policy 256/384/48/12 %). Sizes
in the table are *characters* of generated text: "1 MB" of Cyrillic Markdown is 1.8 MB on disk.):

| Measurement | Run 1 | Run 2 | Run 3 | Budget |
|---|---|---|---|---|
| Count tokens of 6 MB of text | 1.28 | 1.29 | 1.38 | 7 |
| Markdown, 1 MB, parse + chunk (structure-aware) | 3.49 | 3.34 | 3.28 | 15 |
| HTML, 1 MB, parse + chunk | 1.78 | 1.74 | 1.79 | 10 |
| Plain text, 1 MB, parse + chunk (recursive) | 1.00 | 1.04 | 1.06 | 6 |
| Chunk 1 MB, structure-aware | 1.42 | 1.31 | 1.33 | 15 |
| Chunk 1 MB, recursive | 1.01 | 1.04 | 0.92 | 15 |
| Chunk 1 MB, fixed | 2.96 | 3.15 | 3.27 | 15 |
| PDF, 100 pages, parse + chunk | 0.61 | 0.62 | 0.75 | 5 |
| DOCX, 2000 paragraphs, parse + chunk | 2.08 | 2.21 | 2.17 | 10 |
| Isolated worker: start + first document | 0.95 | 0.79 | 0.74 | 5 |
| Isolated worker: one more 20 KB document | 0.004 | 0.005 | 0.004 | 0.5 |
| Peak Python memory (`tracemalloc`), Markdown of 2 million characters (3.6 MB) parse + chunk | 25 MB | 25 MB | 25 MB | 150 MB |

The worker adds under a second once per process and a few milliseconds per document.

## Memory

Measured on 2026-11-22 (4 vCPU, Python 3.12.3, Linux) with generated documents, each parsed in a
fresh process (scripts are not part of the repository; the perf test above repeats the
`tracemalloc` case). Sizes are millions of characters of normalised text (Mchar).

**Python-level peak, parse + chunk (`tracemalloc`), MB per Mchar**

| Format | 0.5 Mchar | 1 Mchar | 2 Mchar |
|---|---|---|---|
| Plain text | 5.0 | 5.0 | n/a |
| HTML | 11.0 | 9.4 | n/a |
| DOCX | 12.0 | 10.0 | 9.4 |
| Markdown (Cyrillic) | 12.9 | 12.4 | 12.4 |
| PDF | about 2 MB in total for up to 100 pages (the text is small) | | |

**Growth of the peak resident set (RSS) over the 70 MB of a process that has only imported the
code, at 4 Mchar (2 Mchar for DOCX), MB per Mchar:** Markdown 9.0, HTML 5.8, DOCX 5.4, plain text
1.8 for parsing and 3.5 for parsing and chunking, PDF about 0.

**Smallest worker cap (`isolation_memory_mb`, `RLIMIT_DATA`) that still parsed the document**,
among 192, 256, 320, 384, 448, 512 and larger:

| Document | Result |
|---|---|
| any format up to 4 Mchar (Markdown 7 MB on disk) | 256 MB suffices; 192 MB did not |
| HTML, 16.3 Mchar (16.5 MB) | 384 MB (320 MB failed) |
| Markdown, 16.4 Mchar (28.6 MB) | 448 MB (384 MB failed) |
| plain text, 19.0 Mchar (19 MB) | 320 MB |

What this means, and what it does not:

- The figures are not interchangeable. `tracemalloc` sees Python allocations only, adds its own
  overhead and misses native memory (the HTML parser's C library, the allocator); RSS growth
  misses memory that the process had already mapped and freed, so it under-reports small
  documents; the worker cap counts *virtual* data segment, including about 200 MB that exist
  before the first document. A ratio such as "N times the text" is only meaningful together
  with its metric and its unit (characters, not bytes).
- **Conservative budget.** Main process: 15 MB per Mchar on top of a 70 MB baseline (the worst
  measured Python-level peak is 12.9). Worker: 250 MB plus 14 MB per Mchar (the worst measured
  need is 448 MB at 16.4 Mchar, which this rule covers); the default text limit of 20 Mchar
  gives about 530 MB against the default cap of 1024 MB. Do not set `isolation_memory_mb` below
  512 MB when documents of that size are expected.
- Not measured: DOCX and PDF documents near the archive and page limits, several workers or
  concurrent documents, other Python versions, another allocator or kernel.
- **Known limitation.** When a cap is too small, the HTML parser's native allocation can fail
  inside the C library; the worker then reports `corrupt` (`SelectolaxError`) instead of
  `memory` (seen at 256 MB with a 16 Mchar page). The failure is still contained and clean, but
  its reason is misleading.

## Running

```bash
make test        # unit tests only
make test-all    # everything except perf, with coverage and the domain coverage gate (90 %)
make perf        # NFR-04 and the Phase 3 budgets, run before a phase is closed
make check       # lint, types, architecture, all tests
```

Permission-based tests are skipped when run as root (root ignores file modes).
