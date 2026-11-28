"""Consistent online backup through ``VACUUM INTO`` with verification of the result."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.domain import ErrorCode, RagError

if TYPE_CHECKING:
    from rag_engine.storage.integrity import IntegrityReport


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
    from rag_engine.storage.sqlite.connection import connect  # lazy: avoids an import cycle

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
    partial.chmod(0o600)
    partial.replace(destination)
    return destination


def _verify(origin: sqlite3.Connection, copy_path: Path) -> None:
    from rag_engine.storage.sqlite.connection import connect  # lazy: avoids an import cycle

    copy = connect(copy_path, busy_timeout_ms=5000, read_only=True)
    try:
        verdict = copy.execute("PRAGMA integrity_check").fetchall()
        if [row[0] for row in verdict] != ["ok"]:
            raise RagError(ErrorCode.E_INTEGRITY, "the backup failed its integrity check")
        tables = [
            row[0]
            for row in origin.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                " ORDER BY name"
            )
        ]
        for table in tables:
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


def verify_backup(path: Path) -> IntegrityReport:
    """Check a backup file the way ``doctor --integrity`` checks a live database."""
    from rag_engine.storage.integrity import check_integrity  # lazy: avoids an import cycle

    if not path.is_file():
        raise RagError(
            ErrorCode.E_SOURCE_NOT_FOUND, "the backup does not exist", context={"path": str(path)}
        )
    return check_integrity(path)


def restore_backup(backup: Path, destination: Path, *, replace: bool = False) -> Path | None:
    """Put a verified backup in place of ``destination``; return the file it displaced, if any.

    The backup is verified first and copied under a temporary name, so a damaged backup never
    reaches the destination and a failed copy leaves the old database as it was. An existing
    destination is refused unless ``replace`` is set, and even then it is kept next to the new one
    under a ``.before-restore`` name rather than deleted. Write-ahead-log files of the displaced
    database are moved with it, so the restored file is never opened with a stale log.
    """
    from rag_engine.storage.integrity import check_integrity  # lazy: avoids an import cycle

    report = verify_backup(backup)
    if not report.ok:
        raise RagError(
            ErrorCode.E_INTEGRITY,
            "the backup failed verification and was not restored",
            context={"errors": list(report.errors)[:5]},
        )
    displaced: Path | None = None
    if destination.exists():
        if not replace:
            raise RagError(
                ErrorCode.E_INTEGRITY,
                "the destination database exists; pass --replace to keep it aside and restore",
                context={"destination": str(destination)},
            )
        displaced = destination.with_name(destination.name + ".before-restore")
        if displaced.exists():
            raise RagError(
                ErrorCode.E_INTEGRITY,
                "an earlier displaced database is still there; move it away first",
                context={"path": str(displaced)},
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".restore-partial")
    partial.unlink(missing_ok=True)
    try:
        shutil.copyfile(backup, partial)
        partial.chmod(0o600)
        if not check_integrity(partial).ok:
            raise RagError(ErrorCode.E_INTEGRITY, "the copy failed verification")
        if displaced is not None:
            destination.replace(displaced)
        for suffix in ("-wal", "-shm"):
            stale = destination.with_name(destination.name + suffix)
            if not stale.exists():
                continue
            if displaced is not None:
                stale.replace(displaced.with_name(displaced.name + suffix))
            else:
                stale.unlink()
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return displaced
