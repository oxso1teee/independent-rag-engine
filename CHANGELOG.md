# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/); versions follow the phases of the
project until a release is cut.

## [Unreleased]

### Phase 5: retrieval (in progress)
- `QueryProcessor`: NFC, control characters to spaces, invisible format characters dropped,
  whitespace collapsed, a character cap before tokenising and a token limit (a query over it is
  refused, never truncated), and script-based language detection with an optional hint. New error
  code `E_QUERY_INVALID` (category `user`, exit code 2) for blank, unusable or over-long queries.
- `rag_engine.retrieval.fusion`: reciprocal rank fusion (`k = 60`, dense weight 1.0, lexical
  weight 0.5 as starting values) and min-max weighted-score fusion of the dense and lexical
  channels. Every component (raw scores, per-channel ranks) is kept; ties are broken by chunk id,
  so the order depends only on the inputs. A chunk absent from a channel adds nothing for it.
- Metadata filters (`RetrievalFilters`) resolve to the set of searchable chunks through
  parameterised SQL (`storage/filters.py`, port `ChunkFilter`): sources, case-sensitive path
  prefix, path glob (the source-policy grammar, now `domain/globs.py`), tags any/all, chunk
  language and kind, modification window, document include/exclude. There is no raw SQL: values
  are bound parameters, lists are capped at 1000 values, a malformed glob is `E_FILTER_INVALID`.
  Only indexed, non-duplicate chunks of active documents' current versions can match. A
  `ChunkCatalog` port gives chunk identity and text hash for diversity and result building.
- `rag_engine.evaluation`: Recall@k, Precision@k, hit rate (coverage), reciprocal rank and graded
  nDCG@k for one ranking, and seeded percentile bootstrap intervals, plain and paired over queries.
  Values are checked against hand-computed rankings; the same seed reproduces an interval exactly.

### Phase 4: embeddings and storage (in progress)
- Decision D-03 closed ([ADR 0012](docs/adr/0012-embedding-default-evaluation.md)): the default
  embedding model is `intfloat/multilingual-e5-small` through ONNX, chosen by an explicit rule
  (resource gate, paired quality interval, tie-breaks). Evidence in `docs/EMBEDDING_BAKEOFF.md`: NoMIRACL judged-pool ranking in
  English and Russian with paired bootstrap intervals, uncached latency, throughput and memory of
  E5-small and Qwen3 0.6B, exact dense search at 10k and 100k vectors, and the memory of loading a
  real 100k-vector index (peak 276 MiB). Qwen3 4B quality could not be measured on the 7.8 GiB
  development machine and is recorded as such. Scripts: `eval/scripts/bakeoff_nomiracl.py`,
  `bakeoff_report.py`, `benchmark_dense.py`, `benchmark_refresh.py`.
- Phase 4 acceptance criteria and their evidence are tabulated in `docs/TESTING.md`.
- Acceptance tests for Phase 4 failure paths: sixteen injected corruptions that `doctor
  --integrity` must report (and never repair), every fencing guard of the atomic publisher,
  damaged stored vectors and model metadata at index load, indexing-handler refusals, Ollama
  response and connection errors, and the exact token counter. Total coverage rose from 93 % to
  94 %; `integrity.py` 79 → 96 %, `publishing.py` 81 → 97 %, `vector_flat.py` 83 → 98 %.
- ONNX embeddings support the pinned `intfloat/multilingual-e5-small` snapshot offline (384
  dimensions, 512 tokens, `query: ` / `passage: ` prefixes, mean pooling). The adapter counts
  tokens with the model's own `tokenizer.json`, special tokens included, and refuses an input over
  the window with `E_RESOURCE_GATE` instead of letting the inference library truncate it. Inference
  runs in-process (`parallel=None`) so a second concurrent call is still refused, not queued.
  Opt-in tests with the real weights run with `RAG_TEST_E5_DIR` set (marker `external`).
- `rag doctor --integrity` checks the read-only SQLite snapshot, foreign keys, model identity,
  vector shape and norm, current versions, chunk provenance, FTS consistency and index generation.
- Replaced-version chunks lose their indexed flag when unpublished. Embedding GC retains vectors
  referenced by any retained chunk and removes orphan cache entries after version data is purged.
- NFR-05 checks actual re-embedding after one paragraph edit in a 100-plus-chunk document;
  model-based random operations and failure injection exercise integrity and rollback.

### Phase 3: parsing and chunking (complete)
- Domain: `Block.attrs`, UTC check for `ParsedDocument.created`, `detect_language`,
  `check_block_provenance`, `check_duplicate_graph`; ADR 0008 defines the contracts.
- Ports: `ParseLimits`, `ParseHints.encoding/limits`, `Parser.normalizer_version`; domain
  `ParseReason` and `parse_failed`; settings sections `[parsing]` and `[chunking]`.
- Parsing: idempotent text normalisation (prose and code modes), exactly-once entity decoding,
  validated per-source cleaners that version the normaliser, and `TextAssembler`, which builds
  `normalized_text`, block offsets and heading paths together.
- Plain-text parser (declared or detected encoding, blank-line paragraphs, empty documents) and a
  shared `Parser` contract suite.
- Markdown parser: headings (ATX and setext), paragraphs, nested list items, quotes, fenced and
  indented code, tables, raw HTML blocks reduced to text, front-matter metadata (title, author,
  date) read without a YAML library; entities decoded once. Dependencies: `markdown-it-py`,
  `selectolax`.
- HTML parser: main-content selection, page furniture and hidden elements removed, headings,
  paragraphs, nested lists, tables, code and quotes, title, author and date from the page head.
- Token counting: `HeuristicTokenCounter` (script-aware, at least one token per word, errs on the
  high side) and `HfTokenCounter` for a local `tokenizer.json` (optional extra `tokenizers`).
- `DocumentProcessor`: size-limited reading, format identification from content, parser
  selection, provenance of a version (parser, normaliser, MIME decision, content hash) and
  clean failures (`E_PARSE_FAILED` with a reason, `E_UNSUPPORTED_FORMAT`); parser registry.
- Text splitting for the chunkers: sentences (Russian and English abbreviations, initials,
  decimals, list markers, ellipsis), words, lines and size-bounded hard splits that never separate
  a combining mark from its letter; every function reassembles its input exactly.
- Chunking core: a packer that joins sentences, lines and word groups up to the target size,
  repeats the tail of the previous prose chunk as overlap (never around tables or code), merges a
  too-small last chunk and guarantees `max_tokens` by recursive halving; a chunk assembler that
  builds identified, language-tagged chunks as exact slices of the normalised text and marks
  repeated leaf text as a non-indexed duplicate.
- `fixed` and `recursive` chunkers with a shared `Chunker` contract suite (bounds, coverage,
  provenance, determinism, overlap, duplicates, random documents).
- `structure_aware` chunker: sections cut at headings, tiny sections merged while they fit the
  target, headings without text joined to the section they introduce, a parent chunk (level 1)
  for every heading-led group with two or more leaves; the heading belongs to the parent, or to
  the leaf when the group has one leaf. Shares the `Chunker` contract suite.
- `ChunkerRegistry` and the `ChunkerSelector` port: `auto` picks `structure_aware` for documents
  with headings, `recursive` for several blocks and `fixed` for a single block; `[chunking]
  strategy` forces one. `DocumentProcessor.process` parses and chunks a file and reports the
  chunker, its version and its fingerprint; each stage is logged without document text.
- `corpus_mini` fixtures: sixteen original Russian, English and mixed documents (Markdown, HTML,
  plain text, a cp1251 file, tiny sections, empty headings, tables, code) produced by a
  deterministic generator, with a corpus test that checks parser and strategy choice, the chunk
  invariants under two policies, determinism, language, heading paths and parents.
- `rag chunks DOC`: parses and chunks a file (or, with `--source`, a document of a registered
  source with that source's cleaners) and prints the chunks as text or `--json`; `--cleaner`,
  `--strategy` and `--full` options; no database is needed for a plain file. `rag source add
  --cleaner` stores validated line cleaners with the source; `SourceRegistry.open` reads a
  document through its source; the container builds counter, limits, strategy and parsers from
  `[parsing]` and `[chunking]` (`Processing`).
- Architecture contracts: `parsing` and `chunking` join the independent adapter row and may not
  import storage, configuration, ingestion, the network or each other; negative tests run the
  project's own configuration against planted violations.
- PDF parser (optional extra `pdf`, `pypdf`): text layer to page-numbered paragraphs; lines joined
  by layout heuristics, hyphenated line ends and ligatures repaired, running titles and page
  numbers removed, title, author and date from the file; password-protected files
  (`encrypted`), scans (`no_text_layer`), damaged files, page, text and block limits and a time
  limit end as `E_PARSE_FAILED` with a reason. Source cleaners run on the original lines. A
  test helper builds PDFs without fixtures or network; two PDF reports join `corpus_mini`.
- DOCX parser (optional extra `docx`, `python-docx`): headings (a Title puts the Heading styles
  one level below it), paragraphs, nested list items, quotes and code from styles, tables as
  rows of cells, tracked insertions kept and deletions and text boxes left out, title, author and
  date from the file. The archive is checked first (entries, uncompressed size, compression
  ratio); password-protected files and damaged archives end as `E_PARSE_FAILED` with a reason.
  Test builders create DOCX files; two handbooks join `corpus_mini` (byte-identical on
  regeneration).
- Parser isolation: PDF, DOCX and HTML are parsed in a long-lived worker process by default
  (`[parsing] isolate`). The worker starts with a minimal environment, a data-segment cap
  (`isolation_memory_mb`) and no core dumps; messages are length-prefixed JSON, never pickle. A
  hang is killed at the time limit (`timeout`), exhausted memory ends as `memory`, a crash as
  `worker_failed`; the next document starts a fresh worker, which is also replaced every 200
  documents. Results equal those of in-process parsing (tested for the real parsers).
- `duplicate_of`: after ingestion changes are applied, in the same transaction, documents of one
  source with identical raw content are linked to the active one with the smallest
  `DocumentId` (no chains; links follow additions, changes, renames, disappearances and
  restorations; non-active documents hold none). `rag ingest` reports the links it updated. The
  integrity checks verify the graph exactly, and the model-based test now creates copies.
- Hardening: a golden snapshot of how every `corpus_mini` document is parsed and chunked (with an
  update script), property tests of parse-then-chunk for generated Markdown, HTML and text under
  random size policies, and mutation fuzzing of every corpus file through the whole pipeline,
  including equality of isolated and in-process outcomes.
- Performance budgets (`make perf`): token counting, parsing and chunking of 1 MB Markdown, HTML
  and text, a 100-page PDF, a 2000-paragraph DOCX, the isolated worker and peak memory, with
  three measured runs recorded in `docs/TESTING.md`.
- `eval/scripts/chunk_grid.py`: reports how a corpus is chunked at target sizes 192, 256 and 384
  tokens (chunk and section counts, token distribution, small and near-limit shares, overlap,
  duplicates, time) as Markdown or JSON. It describes shapes only and ranks nothing.
- `corpus_mini` gains a service-line and repeated-disclaimer document; the fixtures are excluded
  from the formatter so they stay byte-identical to the generator's output.
- Memory estimate corrected: memory is measured per million characters, with the Python-level
  peak, the resident-set growth and the worker cap kept apart; a conservative budget (15 MB per
  million characters, and 250 MB plus 14 MB per million characters for the worker) and the
  limits of the measurement are in `docs/TESTING.md` and ADR 0008.

### Phase 2 refinements
- Schema v2 (`0002_align_data_model.sql`, forward only; `0001` untouched): `document_versions`,
  `normalized_text`, `UNIQUE (document_id, content_hash)`, no `pending` status. Documents
  without a version are `active` and not searchable until their first commit. Phase 4's
  migration becomes v3.
- Automatic verified backup before migrating a populated database; foreign-key-safe table
  rebuilds in the migration runner; the schema version is mirrored from `schema_migrations`.
- Mass-delete guard: `max(10, 20 %)` plus `empty_scan` and `all_known_gone`; the flag is
  `--accept-mass-delete` (`--allow-mass-delete` is removed).
- CLI exit codes follow the error category.
- Retry delays have jitter; retry and lease parameters are `[ingest]` settings.
- MIME detection on `filetype` and `charset-normalizer` (cp1251, koi8-r, cp866, UTF-16 with BOM);
  the encoding is stored with the job.
- Ingestion emits structured stage events (numbers only).
- Composition root moved to `rag_engine/container.py` (new entry point); storage split into
  `core.py`, `uow.py`, `backup.py`; tests reorganised into `perf`, `chaos`, `integrity`; PR CI
  excludes `perf`.
- New tests: `SIGKILL` at every stage, invariants at teardown, a model-based ingestion test.
- ADRs 0005-0007; ADR 0004 refined.

## Phase 2: SQLite core and document ingestion

### Added
- SQLite store: WAL, single writer lock (`E_STORE_LOCKED`), unit of work with rollback,
  read-only opening, verified `VACUUM INTO` backup.
- Numbered, checksummed SQL migrations; schema v1 (sources, documents, versions, sync state,
  jobs, index state).
- `FilesystemSource` with include/exclude globs, symlink policy, root boundary, NFC keys.
- Change detection (size/mtime first, SHA-256 only when needed), content-based MIME detection.
- Source identity, explicit rebase, content-proven move detection.
- Tombstones with a per-source grace period, garbage collection and the mass-delete guard
  (`E_MASS_DELETE_GUARD`).
- Persistent job queue: idempotent enqueue, leases, heartbeats, exponential backoff, dead
  letters, crash recovery.
- `rag source add|list|rebase`, `rag ingest [--dry-run]`, `rag status` (Typer), JSON output and
  stable exit codes.
- Domain: `SyncState`, `IngestionJob`, `IndexMetadata`, `JobId`, `DocumentStatus.PENDING`;
  `SourceRevision` moved into the domain.
- Documentation: data model, testing guide; ADR 0004 accepted with amendments.

### Changed
- The Phase 1 `Store` port is replaced by `MetadataReader`, `MetadataWriter`, `UnitOfWork`
  and `Store`; `SourceAdapter` gained `scan`.
- The command line is built with Typer instead of `argparse`.
- Import contracts: `ingestion` sits above the adapters and may not import them.

### Not included
Parsing, chunking, embeddings, indexes, retrieval, generation, API (later phases).

## [0.0.1] - Phase 1: core foundation
Domain model, error taxonomy, ports, configuration, logging, test doubles, CI.
