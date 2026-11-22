"""A database exactly as released with schema v1 (migration 0001 only), filled through SQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rag_engine.domain import Sha256Hex
from rag_engine.storage.migrations import Migrator, load_packaged_migrations
from tests.factories import T0, make_ids

PUBLISHED_0001_SHA256 = "b04ed778d1ae9e802297a1235ff3bf2cca89b1ae22074ea08064053142ab748b"


def _conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


class V1Data:
    """Identifiers of the rows written into a schema-v1 database."""

    def __init__(self) -> None:
        ids = make_ids()
        self.source = str(ids.source_id())
        self.docs = {key: str(ids.document_id()) for key in "abcde"}
        self.version = str(ids.document_version_id())
        self.jobs = [str(ids.job_id()), str(ids.job_id())]
        self.hash = "a" * 64
        self.statuses = {
            "a": "pending",
            "b": "active",
            "c": "missing",
            "d": "failed",
            "e": "deleted",
        }


def build_v1_database(path: Path) -> tuple[V1Data, dict[str, int]]:
    """A database exactly as released with schema v1, filled through plain SQL."""
    data = V1Data()
    conn = _conn(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        Migrator(load_packaged_migrations()[:1]).apply(conn, T0)
        stamp = "2026-01-01T00:00:00.000000+00:00"
        conn.execute(
            "INSERT INTO sources VALUES (?, 'notes', 'fs', '/n', ?, ?, ?, ?)",
            (
                data.source,
                Sha256Hex.of_text("/n").hex,
                '{"include":["**/*"],"exclude":[],"max_file_bytes":1000,"cleaners":[],'
                '"follow_symlinks":false,"delete_grace_days":30}',
                stamp,
                stamp,
            ),
        )
        for key, status in data.statuses.items():
            conn.execute(
                "INSERT INTO documents (id, source_id, source_key, current_version_id, status,"
                " missing_since, tags_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?)",
                (
                    data.docs[key],
                    data.source,
                    f"{key}.md",
                    data.version if key == "b" else None,
                    status,
                    stamp if status == "missing" else None,
                    stamp,
                    stamp,
                ),
            )
        conn.execute(
            "INSERT INTO versions VALUES (?, ?, 1, ?, 5, 6, 'text/plain', 'content',"
            " 'p', '1', '1', 'fp', 0, '[]', ?)",
            (data.version, data.docs["b"], data.hash, stamp),
        )
        conn.execute(
            "INSERT INTO sync_state VALUES (?, 'a.md', 1, 2, ?, ?, 'changed', 0, NULL, NULL)",
            (data.source, data.hash, stamp),
        )
        for job_id, key in zip(data.jobs, "ab", strict=True):
            conn.execute(
                "INSERT INTO jobs VALUES (?, 'ingest_doc', ?, ?, 'discovered', 'queued', NULL,"
                ' 0, NULL, \'{"mime":"text/plain"}\', ?, ?, NULL)',
                (job_id, data.source, data.docs[key], stamp, stamp),
            )
        conn.execute(
            "INSERT INTO index_state VALUES (1, 1, 7, NULL, 'none', 'none', 'none', '{}', ?, NULL)",
            (stamp,),
        )
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in ("sources", "documents", "versions", "sync_state", "jobs", "index_state")
        }
        return data, counts
    finally:
        conn.close()
