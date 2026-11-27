"""Metadata filters and chunk identity lookups over the published SQLite index.

A filter is turned into a parameterised ``WHERE`` clause built only from the fixed fragments
below; every caller-supplied value travels as a bound parameter, so there is no raw SQL to
inject into. Path globs run through a registered SQL function that uses the same glob grammar
as the source policies, not SQLite's own ``GLOB``.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag_engine.domain import (
    ChunkId,
    ChunkRef,
    DocumentId,
    DocumentVersionId,
    ErrorCode,
    RagError,
    RetrievalFilters,
    TextHash,
)
from rag_engine.domain.globs import compile_glob
from rag_engine.ports import ChunkInfo
from rag_engine.storage.sqlite.connection import connect

MAX_VALUES = 1000  # per list: far below SQLite's variable limit, far above any real filter
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Only chunks a search may return: indexed, not duplicates, of an active document's current version.
_SEARCHABLE = (
    " FROM chunks AS c"
    " JOIN document_versions AS v ON v.id = c.document_version_id"
    " JOIN documents AS d ON d.id = v.document_id AND d.current_version_id = v.id"
    " JOIN sources AS s ON s.id = d.source_id"
    " WHERE d.status = 'active' AND c.indexed = 1 AND c.is_duplicate = 0"
)


def _invalid(message: str) -> RagError:
    return RagError(ErrorCode.E_FILTER_INVALID, message)


def _marks(values: Sequence[object], name: str) -> str:
    if len(values) > MAX_VALUES:
        raise _invalid(f"too many values in the '{name}' filter (at most {MAX_VALUES})")
    return ",".join("?" for _ in values)


def _nanoseconds(moment: datetime) -> int:
    return (moment - _EPOCH) // timedelta(microseconds=1) * 1000


def _where(filters: RetrievalFilters) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if filters.sources:
        clauses.append(f"s.name IN ({_marks(filters.sources, 'sources')})")
        params.extend(filters.sources)
    if filters.path_prefix is not None:
        clauses.append("substr(d.source_key, 1, ?) = ?")  # exact and case-sensitive, no LIKE
        params.extend((len(filters.path_prefix), filters.path_prefix))
    if filters.path_glob is not None:
        clauses.append("rag_path_matches(d.source_key)")
    if filters.tags_any:
        marks = _marks(filters.tags_any, "tags_any")
        clauses.append(
            f"EXISTS (SELECT 1 FROM json_each(d.tags_json) AS t WHERE t.value IN ({marks}))"  # noqa: S608
        )
        params.extend(filters.tags_any)
    if filters.tags_all:
        wanted = tuple(dict.fromkeys(filters.tags_all))
        marks = _marks(wanted, "tags_all")
        clauses.append(
            "(SELECT count(DISTINCT t.value) FROM json_each(d.tags_json) AS t"  # noqa: S608
            f" WHERE t.value IN ({marks})) = ?"
        )
        params.extend((*wanted, len(wanted)))
    if filters.language is not None:
        clauses.append("c.language = ?")
        params.append(filters.language.value)
    if filters.kinds:
        clauses.append(f"c.kind IN ({_marks(filters.kinds, 'kinds')})")
        params.extend(kind.value for kind in filters.kinds)
    if filters.modified_after is not None:
        clauses.append("v.mtime_ns > ?")
        params.append(_nanoseconds(filters.modified_after))
    if filters.modified_before is not None:
        clauses.append("v.mtime_ns < ?")
        params.append(_nanoseconds(filters.modified_before))
    if filters.doc_ids:
        clauses.append(f"d.id IN ({_marks(filters.doc_ids, 'doc_ids')})")
        params.extend(str(item) for item in filters.doc_ids)
    if filters.exclude_doc_ids:
        clauses.append(f"d.id NOT IN ({_marks(filters.exclude_doc_ids, 'exclude_doc_ids')})")
        params.extend(str(item) for item in filters.exclude_doc_ids)
    return "".join(f" AND {clause}" for clause in clauses), params


def _is_empty(filters: RetrievalFilters) -> bool:
    return filters == RetrievalFilters()


class SqliteChunkFilter:
    def __init__(self, database: Path) -> None:
        self._database = database

    def resolve(self, filters: RetrievalFilters) -> frozenset[ChunkId] | None:
        if _is_empty(filters):
            return None
        clause, params = _where(filters)
        matcher = None
        if filters.path_glob is not None:
            try:
                pattern = compile_glob(filters.path_glob)
            except RagError as exc:
                raise _invalid("the path_glob filter is not a valid glob pattern") from exc

            def matcher(key: str) -> int:
                return 1 if pattern.fullmatch(unicodedata.normalize("NFC", key)) else 0

        conn = connect(self._database, busy_timeout_ms=5000, read_only=True)
        try:
            if matcher is not None:
                conn.create_function("rag_path_matches", 1, matcher, deterministic=True)
            conn.execute("BEGIN")
            rows = conn.execute("SELECT c.id" + _SEARCHABLE + clause, params)
            return frozenset(ChunkId(row[0]) for row in rows)
        except sqlite3.Error as exc:
            raise RagError(ErrorCode.E_INTEGRITY, "the filter query failed") from exc
        finally:
            conn.close()


class SqliteChunkCatalog:
    def __init__(self, database: Path) -> None:
        self._database = database

    def describe(self, ids: Sequence[ChunkId]) -> Mapping[ChunkId, ChunkInfo]:
        if not ids:
            return {}
        found: dict[ChunkId, ChunkInfo] = {}
        conn = connect(self._database, busy_timeout_ms=5000, read_only=True)
        try:
            conn.execute("BEGIN")
            for start in range(0, len(ids), MAX_VALUES):
                batch = [str(item) for item in ids[start : start + MAX_VALUES]]
                rows = conn.execute(
                    "SELECT c.id, c.document_version_id, v.document_id, c.text_hash"  # noqa: S608
                    " FROM chunks AS c JOIN document_versions AS v"
                    " ON v.id = c.document_version_id"
                    f" WHERE c.id IN ({','.join('?' for _ in batch)})",
                    batch,
                )
                for row in rows:
                    chunk_id = ChunkId(row["id"])
                    found[chunk_id] = ChunkInfo(
                        ChunkRef(
                            chunk_id,
                            DocumentId.parse(row["document_id"]),
                            DocumentVersionId.parse(row["document_version_id"]),
                        ),
                        TextHash(row["text_hash"]),
                    )
            return found
        except sqlite3.Error as exc:
            raise RagError(ErrorCode.E_INTEGRITY, "the chunk lookup failed") from exc
        finally:
            conn.close()
