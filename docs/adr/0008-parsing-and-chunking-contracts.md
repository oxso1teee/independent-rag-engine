# 0008. Parsing and chunking contracts

Status: Accepted (2026-11-22). Implemented; see the architecture document and the changelog.

## Context

Documents arrive as bytes in several formats. Retrieval needs one clean text per document
version, structure and provenance for every piece of it, and chunks that can always be traced
back to exact character offsets. These contracts define that path up to, and not including,
persistence of chunks.

## Decisions

1. **Boundary.** `DocumentProcessor` turns bytes into a `ParsedDocument` and chunks in memory;
   `rag chunks DOC` shows the result. Versions, `normalized_text` and chunks are stored in one
   atomic commit together with the full-text and vector indexes (schema v3); jobs are closed by
   that commit, never by parsing alone.
2. **One text per version.** `normalized_text` is the only text; every block and chunk offset
   refers to it, and `normalized_text[char_start:char_end]` equals the block or chunk text.
   Normalisation is idempotent: NFC, line endings to `\n`, runs of spaces and tabs to one
   space, at most one blank line between paragraphs, no control characters, no zero-width
   space, word joiner, byte-order mark or soft hyphen (joiners that carry meaning in some
   scripts are kept). Text offsets are produced while the text is assembled, so the invariant
   holds by construction.
3. **Markup entities** are decoded exactly once, by the parser that owns the markup (HTML and
   Markdown). Plain text is never unescaped.
4. **Cleaners** are per-source line patterns; a line that matches one entirely is removed. They
   are validated when a source is registered (length, no nested unbounded repetition) and
   applied per line with a line-length cap. The set of cleaners is part of `normalizer_version`
   because it changes the text.
5. **Blocks and provenance.** Block kinds: heading, paragraph, list item, table, code, quote. A
   block's `heading_path` lists the headings above it (a heading's own path ends with itself);
   `page` is set for paged formats and never decreases. Tables are rows of cells joined with
   ` | `, one row per line.
6. **Parsers.** Plain text, Markdown and HTML are built in; PDF (`pypdf`, page by page, outline
   as headings, repeated page furniture removed) and DOCX (`python-docx`, heading styles, lists,
   tables) are optional extras. A parser either returns a valid document or raises
   `E_PARSE_FAILED` (reason: `corrupt`, `encrypted`, `no_text_layer`, `too_large`,
   `too_many_pages`, `zip_bomb`, `timeout`, `memory`) or `E_UNSUPPORTED_FORMAT` (also when an
   extra is not installed). An empty document is valid and has no blocks.
7. **Limits and isolation.** Input size, text length, block count, page count, archive size and
   ratio, nesting depth and wall time are limits with defaults in `[parsing]`; exceeding one is
   an error, never silent truncation. PDF, DOCX and HTML are parsed in a separate, long-lived
   worker process (`[parsing] isolate`), so a malformed file cannot take the main process down:
   the worker is a fresh interpreter with a minimal environment (no secrets), a data-segment
   cap (`isolation_memory_mb`, `RLIMIT_DATA`) and no core dumps. Requests and replies are
   length-prefixed JSON (never pickle); the document is rebuilt through the domain
   constructors, which check every invariant again. A parse that exceeds `timeout_s` (plus a
   short grace period) gets the worker killed (`timeout`); memory exhaustion is reported as
   `memory` and a crash as `worker_failed`; the next document starts a fresh worker, which is
   also replaced after 200 documents. This limits resources, it is not a syscall sandbox.
   Markdown and plain text are parsed in process.
8. **Token counting.** `TokenCounter` is a port. The default `HeuristicTokenCounter` is
   script-aware, deterministic and errs on the high side; `HfTokenCounter` counts with a local
   `tokenizer.json` (optional extra). The counter id is part of the chunker fingerprint.
9. **Chunking.** Strategies: `structure_aware` (headed documents), `recursive` (paragraph,
   sentence, word) and `fixed` (unstructured text). Leaves target 256 tokens and never exceed
   384: a sentence, table row, code line or word that is longer is cut at word or character
   boundaries, so the `oversize` flag stays false for these strategies (it remains part of the
   model for strategies that cannot split). A too-small last chunk is merged into its
   neighbour. Overlap (12 % of the target) repeats whole sentences, or words when no sentence
   fits, of the previous chunk, only between prose chunks of one section, never across a
   heading and never next to a table or code. Tables and code are not cut inside a row or line
   when they fit. Chunks are exact slices of the normalised text. Chunk ids are
   `ChunkId.derive(version, fingerprint, ordinal)`: the same input and policy give the same
   chunks.
10. **Two levels.** A section starts at a heading of any level and ends at the next heading;
    sections do not nest, so each character belongs to one parent. A heading without text of its
    own joins the section it introduces; a section smaller than `min_tokens` (or following one)
    is merged with its neighbour when the result fits the target and the next heading is not of
    a higher level. A group with two or more leaves gets a parent chunk (level 1) that holds the
    whole group including its heading, and the leaves exclude that heading; a group with one
    leaf keeps its heading in the leaf and has no parent. Parents are not indexed and keep the
    whole section text for expansion at answer time. Text without a heading above it, and
    documents without headings, have leaves only.
11. **Duplicates inside a version.** A repeated chunk text keeps its first occurrence
    indexed; later ones are marked `is_duplicate` and not indexed.
12. **Language.** `detect_language` classifies by script: Cyrillic `ru`, Latin script `en`,
    a substantial mix `mixed`, otherwise `und`. It is a pure function in the domain.
13. **`duplicate_of`.** Between documents of one source whose raw `content_hash` is identical,
    the active document with the smallest `DocumentId` is the original and keeps `duplicate_of`
    empty; the others point to it. There are no chains. When the original disappears a new one
    is chosen; when a group shrinks to one document its link is cleared. The resolver runs
    after ingestion changes were applied and only when something changed. Duplicates are
    marked here and excluded from indexing by the commit that stores versions.
14. **The chunk hierarchy has no module of its own.** Grouping blocks into sections and choosing
    parents is part of the structure-aware chunker (`chunking/structure.py`): it decides the
    leaves and the parent of each group together, and the other strategies have no parents.
    `chunking/assemble.py` builds chunks for every strategy (ids, language, pages, duplicate
    marking); a separate hierarchy module would only split one decision across two files.
15. **The chunk-size grid is a script, not a notebook.** `eval/scripts/chunk_grid.py` runs from
    the command line, prints Markdown or JSON, is deterministic, and is linted, type-checked and
    tested with the rest of the code, which a notebook would not be. It reports the shape of the
    chunks at several target sizes and ranks nothing: choosing a size needs retrieval quality
    measurements.

## Consequences

- Dependencies: Markdown (`markdown-it-py`) and HTML (`selectolax`) parsing are in the core;
  `pypdf` and `python-docx` are extras (`[pdf]`, `[docx]`); `tokenizers` is the extra
  `[tokenizers]` because it brings network-capable packages that exact counting does not need.
- **Memory grows with the number of characters of the normalised text**, not with the size of
  the file (a Python string takes 1, 2 or 4 bytes per character; Cyrillic Markdown is about 1.8
  bytes per character on disk). Three different quantities are involved and must not be
  mixed: the Python-level peak (`tracemalloc`: every Python allocation, no native one), the
  growth of the peak resident set (native memory included, but memory freed earlier in the
  process is reused, so small documents show almost no growth) and the data segment that the
  worker cap limits (`RLIMIT_DATA`: virtual, about 200 MB before any document is read).
  Measured, parsing and chunking one document needs about 5 MB per million characters for
  plain text and 9 to 13 MB for HTML, DOCX and Markdown (Python-level peak); a PDF costs
  little, its text is small. **Conservative budget: 15 MB per million characters plus the
  interpreter baseline.** For the worker, plan 250 MB plus 14 MB per million characters: the
  default text limit (20 million characters) then needs about 530 MB, and the default cap
  (`isolation_memory_mb = 1024`) leaves a margin of about two. The measurements and their
  limits are in [TESTING.md](../TESTING.md#memory).
- Chunk and version storage, embeddings and indexes are not part of these contracts.
