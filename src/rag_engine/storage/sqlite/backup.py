"""Consistent online backup through ``VACUUM INTO`` with verification of the result."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rag_engine.domain import ErrorCode, RagError
from rag_engine.storage.sqlite.connection import connect

_TABLES = ("sources", "documents", "versions", "sync_state", "jobs", "index_state")


def backup_database(source: Path, destination: Path) -> Path:
    """Copy ``source`` into a new, compact database file and verify the copy.

    The destination must not exist. The copy is built under a temporary name and moved
    into place only after ``integrity_check`` passes and the row counts match, so a
    failed backup never leaves a plausible-looking file behind.
    """
    if destination.exists():
        raise RagError(
            ErrorCode.E_INTEGRITY,
            "the backup destination already exists",
            context={"destination": str(destination)},
        )
    if not source.is_file():
        raise RagError(
            ErrorCode.E_SOURCE_NOT_FOUND,
            "the database to back up does not exist",
            context={"path": str(source)},
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    origin = connect(source, busy_timeout_ms=5000, read_only=True)
    try:
        try:
            # mode=ro already protects the source; VACUUM INTO needs query_only to be off
            origin.execute("PRAGMA query_only = OFF")
            origin.execute("VACUUM INTO ?", (str(partial),))
        except sqlite3.Error as exc:
            partial.unlink(missing_ok=True)
            raise RagError(
                ErrorCode.E_INTEGRITY,
                "the backup could not be created",
                context={"sqlite_error": type(exc).__name__},
            ) from exc
        _verify(origin, partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        origin.close()
    partial.replace(destination)
    return destination


def _verify(origin: sqlite3.Connection, copy_path: Path) -> None:
    copy = connect(copy_path, busy_timeout_ms=5000, read_only=True)
    try:
        verdict = copy.execute("PRAGMA integrity_check").fetchall()
        if [row[0] for row in verdict] != ["ok"]:
            raise RagError(ErrorCode.E_INTEGRITY, "the backup failed its integrity check")
        for table in _TABLES:
            expected = origin.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            actual = copy.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            if expected != actual:
                raise RagError(
                    ErrorCode.E_INTEGRITY,
                    "the backup does not match the original",
                    context={"table": table, "expected": expected, "actual": actual},
                )
    finally:
        copy.close()
