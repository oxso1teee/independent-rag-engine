# 0022. A slimmer database (planned after the first release)

Status: Proposed (2026-11-28). Not implemented. The owner decided that this migration follows the
first release; the NFR-02 database target (about 3× the source text plus vectors) stays unchanged
and is reported as not met ([AUDIT.md](../AUDIT.md), F-2).

## Context

Measured on 3.5 MB of real prose with 384-dimension vectors (`eval/scripts/db_size.py`): 7.0× the
source text, 5.6× without vectors. The `chunks` table alone is 3.5×: leaf text 0.88, section
("parent") text 0.64, the analysed text for full-text search 0.74, ids and hashes 0.29, headings
and metadata 0.13. Embeddings are 1.79× (theory 1.34×), the full-text index 0.46×, the compressed
document text 0.39×, secondary indexes about 0.7×.

## Plan (each step is its own migration and its own pull request)

1. **Parents without text.** A section chunk keeps only its offsets; its text is cut from the
   document version's normalised text when read. Saves about 0.64×. Needs the context builder and
   the chunk presenter to read through one accessor, and a migration that drops the text of level-1
   rows (their text is reproducible from the offsets, which a test must prove before the drop).
2. **Analysed text only in the full-text index.** Move to a contentless or self-contained FTS5
   table so `body_an` is not also a column of `chunks`. Saves about 0.74× but changes how snippets
   and the rebuild of the index work (`rebuild_lexical_index` must re-analyse from the text).
3. **A narrower embeddings row.** Key the vector by an integer id and keep the model id once, not
   per row. Saves part of the 0.45× above theory.
4. **Fewer secondary indexes** where the query plan does not need them (check each with
   `EXPLAIN QUERY PLAN`).

Expected result: about 4× with vectors removed, still above 3×. The target should then be restated
with the measurement rather than quietly changed.

## Rules for the work

- Migrations are forward-only and tested on a real database of the previous schema (the existing
  upgrade tests are the template); `rag backup` before and `rag doctor --integrity` after.
- No retrieval metric may change: the retrieval evaluation and the regression baseline run before
  and after, and the paired comparison must show no difference.
- Each step reports the size with `db_size.py` on the same corpus.
