# 0008. Parsing and chunking contracts

Status: Accepted (2026-11-22). Implemented task by task; the changelog lists what is in place.

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
   worker process with a memory limit and a timeout, so a malformed file cannot take the main
   process down; Markdown and plain text are parsed in process.
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

## Consequences

- Dependencies: Markdown (`markdown-it-py`) and HTML (`selectolax`) parsing are in the core;
  `pypdf` and `python-docx` are extras (`[pdf]`, `[docx]`); `tokenizers` is the extra
  `[tokenizers]` because it brings network-capable packages that exact counting does not need.
- Peak memory of one document is about six times its text; the default text limit (20 million
  characters) keeps it well under the isolation memory limit.
- Chunk and version storage, embeddings and indexes are not part of these contracts.
