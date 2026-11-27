"""Chunk text, position and provenance for evidence and reranking."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from rag_engine.domain import (
    ChunkId,
    ChunkRef,
    DocumentId,
    DocumentVersionId,
    ErrorCode,
    RagError,
)
from rag_engine.ports import ChunkContent
from rag_engine.storage.sqlite.connection import connect

_BATCH = 500


class SqliteChunkContentReader:
    def __init__(self, database: Path) -> None:
        self._database = database

    def read(self, ids: Sequence[ChunkId]) -> Mapping[ChunkId, ChunkContent]:
        if not ids:
            return {}
        found: dict[ChunkId, ChunkContent] = {}
        conn = connect(self._database, busy_timeout_ms=5000, read_only=True)
        try:
            conn.execute("BEGIN")
            for start in range(0, len(ids), _BATCH):
                batch = [str(item) for item in ids[start : start + _BATCH]]
                rows = conn.execute(
                    "SELECT c.id, c.document_version_id, v.document_id, c.text, c.token_count,"  # noqa: S608
                    " c.level, c.parent_id, c.ordinal, c.char_start, c.char_end,"
                    " c.heading_path_json, c.page_start, c.page_end,"
                    " s.name AS source_name, d.source_key, d.title"
                    " FROM chunks AS c"
                    " JOIN document_versions AS v ON v.id = c.document_version_id"
                    " JOIN documents AS d ON d.id = v.document_id"
                    " JOIN sources AS s ON s.id = d.source_id"
                    f" WHERE c.id IN ({','.join('?' for _ in batch)})",
                    batch,
                )
                for row in rows:
                    chunk_id = ChunkId(row["id"])
                    found[chunk_id] = ChunkContent(
                        ref=ChunkRef(
                            chunk_id,
                            DocumentId.parse(row["document_id"]),
                            DocumentVersionId.parse(row["document_version_id"]),
                        ),
                        text=row["text"],
                        token_count=int(row["token_count"]),
                        level=int(row["level"]),
                        parent_id=None if row["parent_id"] is None else ChunkId(row["parent_id"]),
                        ordinal=int(row["ordinal"]),
                        char_start=int(row["char_start"]),
                        char_end=int(row["char_end"]),
                        heading_path=tuple(json.loads(row["heading_path_json"])),
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        source_name=row["source_name"],
                        source_key=row["source_key"],
                        title=row["title"],
                    )
            return found
        except (sqlite3.Error, ValueError) as exc:
            raise RagError(ErrorCode.E_INTEGRITY, "the chunk content lookup failed") from exc
        finally:
            conn.close()
