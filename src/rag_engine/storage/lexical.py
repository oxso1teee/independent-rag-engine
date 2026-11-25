"""RU/EN Snowball analysis and safe FTS5 BM25 search over published chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence, Set
from pathlib import Path

from rag_engine.domain import ChunkId, ErrorCode, RagError, ScoredChunk
from rag_engine.lexical.analysis import ANALYZER_ID, safe_match
from rag_engine.ports import LexicalDocument
from rag_engine.storage.sqlite.connection import connect
from rag_engine.storage.sqlite.core import SqliteStore


class SqliteLexicalIndex:
    """FTS5 adapter. Publication owns writes; this port reads active current versions."""

    def __init__(self, database: Path) -> None:
        self._database = database

    def __len__(self) -> int:
        conn = connect(self._database, busy_timeout_ms=5000, read_only=True)
        try:
            return int(
                conn.execute(
                    "SELECT count(*) FROM chunks AS c"
                    " JOIN document_versions AS v ON v.id = c.document_version_id"
                    " JOIN documents AS d ON d.id = v.document_id"
                    " WHERE d.status = 'active' AND d.current_version_id = v.id"
                    " AND c.indexed = 1 AND c.is_duplicate = 0"
                ).fetchone()[0]
            )
        finally:
            conn.close()

    def upsert(self, documents: Sequence[LexicalDocument]) -> None:
        with (
            SqliteStore(self._database, writer_timeout_s=5.0) as store,
            store.unit_of_work().begin() as writer,
        ):
            writer.upsert_lexical_documents(documents)

    def remove(self, ids: Sequence[ChunkId]) -> None:
        with (
            SqliteStore(self._database, writer_timeout_s=5.0) as store,
            store.unit_of_work().begin() as writer,
        ):
            writer.remove_lexical_chunks(ids)

    def search(
        self, query: str, k: int, allowed: Set[ChunkId] | None = None
    ) -> Sequence[ScoredChunk]:
        expression = safe_match(query)
        if k <= 0 or expression is None or (allowed is not None and not allowed):
            return []
        conn = connect(self._database, busy_timeout_ms=5000, read_only=True)
        try:
            # Run a single snapshot; filtering happens before LIMIT.
            conn.execute("BEGIN")
            metadata = conn.execute(
                "SELECT lexical_analyzer_id FROM index_state WHERE id = 1"
            ).fetchone()
            if metadata is None or metadata[0] != ANALYZER_ID:
                raise RagError(ErrorCode.E_INTEGRITY, "lexical analyzer does not match the index")
            sql = (
                "SELECT c.id, -bm25(chunks_fts, 1.0, 2.0) AS score"
                " FROM chunks_fts"
                " JOIN chunks AS c ON c.chunk_rowid = chunks_fts.rowid"
                " JOIN document_versions AS v ON v.id = c.document_version_id"
                " JOIN documents AS d ON d.id = v.document_id"
                " WHERE chunks_fts MATCH ? AND d.status = 'active'"
                " AND d.current_version_id = v.id AND c.indexed = 1 AND c.is_duplicate = 0"
            )
            hits: list[ScoredChunk] = []
            if allowed is None:
                rows = conn.execute(
                    sql + " ORDER BY score DESC, c.id LIMIT ?", (expression, k)
                ).fetchall()
                hits.extend(ScoredChunk(ChunkId(row["id"]), float(row["score"])) for row in rows)
            else:
                candidates = sorted(str(chunk_id) for chunk_id in allowed)
                for start in range(0, len(candidates), 800):
                    batch = candidates[start : start + 800]
                    placeholders = ",".join("?" for _ in batch)
                    rows = conn.execute(
                        sql
                        + f" AND c.id IN ({placeholders})"
                        + " ORDER BY score DESC, c.id LIMIT ?",
                        (expression, *batch, k),
                    ).fetchall()
                    hits.extend(
                        ScoredChunk(ChunkId(row["id"]), float(row["score"])) for row in rows
                    )
            return sorted(hits, key=lambda hit: (-hit.score, str(hit.chunk_id)))[:k]
        except sqlite3.Error as exc:
            raise RagError(ErrorCode.E_INTEGRITY, "FTS search failed") from exc
        finally:
            conn.close()
