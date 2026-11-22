"""Numbered SQL migrations with checksums.

Files are named ``NNNN_name.sql`` and applied strictly in order, each inside one
transaction together with its bookkeeping row. Applied migrations are verified by checksum
on every start, and a database created by a newer version is refused.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

from rag_engine.domain import ErrorCode, RagError

_NAME_RE = re.compile(r"(\d{4})_([a-z0-9_]+)\.sql")


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """SHA-256 of the script with normalised line endings."""
        normalised = self.sql.replace("\r\n", "\n").encode("utf-8")
        return hashlib.sha256(normalised).hexdigest()


def load_packaged_migrations() -> tuple[Migration, ...]:
    """Migrations shipped inside this package, validated to be numbered 1..N without gaps."""
    found: list[Migration] = []
    for entry in resources.files(__package__).iterdir():
        match = _NAME_RE.fullmatch(entry.name)
        if match is None:
            if entry.name.endswith(".sql"):
                raise RagError(
                    ErrorCode.E_INTEGRITY,
                    "migration file has an invalid name",
                    context={"file": entry.name},
                )
            continue
        found.append(Migration(int(match[1]), match[2], entry.read_text(encoding="utf-8")))
    found.sort(key=lambda m: m.version)
    _validate_sequence(found)
    return tuple(found)


def _validate_sequence(migrations: Sequence[Migration]) -> None:
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise RagError(
            ErrorCode.E_INTEGRITY,
            "migrations must be numbered 1..N without gaps or duplicates",
            context={"versions": versions},
        )


def split_statements(script: str) -> list[str]:
    """Split a script into complete statements (triggers with inner ``;`` stay whole)."""
    statements: list[str] = []
    current = ""
    for line in script.splitlines(keepends=True):
        current += line
        if sqlite3.complete_statement(current):
            statements.append(current.strip())
            current = ""
    if current.strip() and not all(
        part.strip().startswith("--") or not part.strip() for part in current.splitlines()
    ):
        raise RagError(ErrorCode.E_INTEGRITY, "migration ends with an incomplete statement")
    return statements


class Migrator:
    """Brings a database to the latest version of ``migrations``."""

    def __init__(self, migrations: Sequence[Migration]) -> None:
        _validate_sequence(migrations)
        self._migrations = tuple(migrations)

    @property
    def latest_version(self) -> int:
        return len(self._migrations)

    def current_version(self, conn: sqlite3.Connection) -> int:
        """Highest applied version (0 for a fresh database)."""
        if not _table_exists(conn, "schema_migrations"):
            return 0
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    def verify(self, conn: sqlite3.Connection) -> int:
        """Check the recorded history against the known migrations; return the version."""
        if not _table_exists(conn, "schema_migrations"):
            return 0
        rows = conn.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        if rows and rows[-1][0] > self.latest_version:
            raise RagError(
                ErrorCode.E_MIGRATION_REQUIRED,
                "the database was created by a newer version of the engine",
                context={"database_version": rows[-1][0], "supported_version": self.latest_version},
            )
        if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
            raise RagError(
                ErrorCode.E_INTEGRITY,
                "the migration history of the database has gaps",
                context={"versions": [r[0] for r in rows]},
            )
        for version, _name, checksum in rows:
            known = self._migrations[version - 1]
            if known.checksum != checksum:
                raise RagError(
                    ErrorCode.E_INTEGRITY,
                    "an applied migration was modified after it had been applied",
                    context={"version": version, "name": known.name},
                )
        return len(rows)

    def apply(self, conn: sqlite3.Connection, now: datetime) -> list[int]:
        """Apply every pending migration; returns the versions applied.

        The connection must be in autocommit mode (``isolation_level=None``) and the caller
        must hold the writer lock.
        """
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        done = self.verify(conn)
        applied: list[int] = []
        for migration in self._migrations[done:]:
            self._apply_one(conn, migration, now)
            applied.append(migration.version)
        return applied

    @staticmethod
    def _apply_one(conn: sqlite3.Connection, migration: Migration, now: datetime) -> None:
        try:
            conn.execute("BEGIN IMMEDIATE")
            for statement in split_statements(migration.sql):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (migration.version, migration.name, migration.checksum, now.isoformat()),
            )
            conn.execute("COMMIT")
        except (sqlite3.Error, RagError) as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if isinstance(exc, RagError):
                raise
            raise RagError(
                ErrorCode.E_INTEGRITY,
                f"migration {migration.version:04d}_{migration.name} failed and was rolled back",
                context={"version": migration.version, "sqlite_error": type(exc).__name__},
            ) from exc


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None
