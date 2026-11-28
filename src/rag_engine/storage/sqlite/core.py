"""SQLite ``Store``: connection lifecycle, opening, migration and index-state bootstrap."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import TracebackType

from rag_engine.domain import ErrorCode, IndexMetadata, RagError
from rag_engine.ports import Clock, MetadataReader, UnitOfWork
from rag_engine.storage.backup import backup_database
from rag_engine.storage.migrations import Migration, Migrator, load_packaged_migrations
from rag_engine.storage.sqlite.connection import WriterLock, connect, enable_wal
from rag_engine.storage.sqlite.repository import SqliteReader, translate
from rag_engine.storage.sqlite.uow import SqliteUnitOfWork

DEFAULT_BUSY_TIMEOUT_MS = 5000

# Placeholders of the Phase 2 index state; schema v2 replaces them with real settings.
_UNSET = "none"


class StoreNotInitialised(RagError):
    """The database file does not exist yet (nothing has been written so far)."""


class SqliteStore:
    """Handle over one database file. Thread-safe: readers use per-thread connections and
    every writer must win the process-wide writer lock."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        writer_timeout_s: float = 0.0,
        read_only: bool = False,
    ) -> None:
        self.path = path
        self.read_only = read_only
        self._busy_timeout_ms = busy_timeout_ms
        self.writer_lock = WriterLock(path.with_name(path.name + ".writer.lock"), writer_timeout_s)
        self.local = threading.local()
        self._guard = threading.Lock()
        # (thread, its read connection): a connection outlives its thread only until the next
        # thread asks for one, when the connections of finished threads are closed
        self._connections: list[tuple[threading.Thread, sqlite3.Connection]] = []
        self._writer: sqlite3.Connection | None = None
        self._closed = False

    # -- Store port --------------------------------------------------------------------

    def reader(self) -> MetadataReader:
        conn: sqlite3.Connection | None = getattr(self.local, "reader", None)
        if conn is None:
            self._ensure_open()
            conn = connect(self.path, busy_timeout_ms=self._busy_timeout_ms, read_only=True)
            self.local.reader = conn
            with self._guard:
                kept: list[tuple[threading.Thread, sqlite3.Connection]] = []
                for thread, old in self._connections:
                    if thread.is_alive():
                        kept.append((thread, old))
                    else:
                        old.close()  # nobody can use a finished thread's connection again
                kept.append((threading.current_thread(), conn))
                self._connections = kept
        return SqliteReader(conn)

    def unit_of_work(self) -> UnitOfWork:
        self._ensure_open()
        return SqliteUnitOfWork(self)

    def close(self) -> None:
        with self._guard:
            self._closed = True
            connections, self._connections = self._connections, []
            writer, self._writer = self._writer, None
        for _, conn in connections:
            conn.close()
        if writer is not None:
            writer.close()
        self.local = threading.local()

    # -- helpers -----------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RagError(ErrorCode.E_INTEGRITY, "the store is closed")

    def writer_connection(self) -> sqlite3.Connection:
        if self._writer is None:
            conn = connect(self.path, busy_timeout_ms=self._busy_timeout_ms)
            with self._guard:
                self._writer = conn
        return self._writer

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_store(
    path: Path,
    clock: Clock,
    *,
    migrations: Sequence[Migration] | None = None,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    writer_timeout_s: float = 0.0,
) -> SqliteStore:
    """Open (creating and migrating when needed) the store at ``path``."""
    migrator = Migrator(load_packaged_migrations() if migrations is None else migrations)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    store = SqliteStore(path, busy_timeout_ms=busy_timeout_ms, writer_timeout_s=writer_timeout_s)
    try:
        store.writer_lock.acquire()
        try:
            conn = store.writer_connection()
            try:
                enable_wal(conn)
                if created:
                    # metadata about private documents: owner-only; SQLite gives -wal/-shm the
                    # same mode as the database file
                    path.chmod(0o600)
                migrator.apply(conn, clock.now(), _backup_hook(path))
            except sqlite3.Error as exc:
                raise translate(exc) from exc
        finally:
            store.writer_lock.release()
        _ensure_index_state(store, clock.now(), migrator.latest_version)
    except BaseException:
        store.close()
        raise
    return store


def open_store_read_only(
    path: Path, *, migrations: Sequence[Migration] | None = None
) -> SqliteStore:
    """Open an existing, already migrated store without changing anything on disk."""
    migrator = Migrator(load_packaged_migrations() if migrations is None else migrations)
    if not path.is_file():
        raise StoreNotInitialised(
            ErrorCode.E_SOURCE_NOT_FOUND,
            "the database does not exist yet",
            context={"path": str(path)},
        )
    store = SqliteStore(path, read_only=True)
    try:
        probe = connect(path, busy_timeout_ms=DEFAULT_BUSY_TIMEOUT_MS, read_only=True)
        try:
            version = migrator.verify(probe)
        finally:
            probe.close()
        if version < migrator.latest_version:
            raise RagError(
                ErrorCode.E_MIGRATION_REQUIRED,
                "the database needs a migration; run a command that writes first",
                context={"database_version": version, "supported_version": migrator.latest_version},
            )
    except BaseException:
        store.close()
        raise
    return store


def backup_path(database: Path, from_version: int, to_version: int) -> Path:
    """A free name for the pre-migration backup, built from the real schema versions."""
    base = database.name + f".pre-migration-v{from_version}-to-v{to_version}"
    candidate = database.with_name(base + ".bak")
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = database.with_name(f"{base}.{counter}.bak")
    return candidate


def _backup_hook(database: Path) -> Callable[[int, int], None]:
    def hook(from_version: int, to_version: int) -> None:
        backup_database(database, backup_path(database, from_version, to_version))

    return hook


def _ensure_index_state(store: SqliteStore, now: datetime, schema_version: int) -> None:
    """Create the singleton row, and keep its schema version equal to the migration history.

    The version is never edited by migration scripts: the migration runner owns it, and this
    function mirrors it into ``index_state`` once migrations have been applied.
    """
    current = store.reader().get_index_metadata()
    if current is not None and current.schema_version == schema_version:
        return
    with store.unit_of_work().begin() as writer:
        existing = writer.get_index_metadata()
        if existing is not None:
            writer.update_index_metadata(replace(existing, schema_version=schema_version))
        else:
            writer.update_index_metadata(
                IndexMetadata(
                    schema_version=schema_version,
                    index_generation=0,
                    active_embedding_model_id=None,
                    lexical_analyzer_id=_UNSET,
                    chunker_default_fingerprint=_UNSET,
                    vector_store_kind=_UNSET,
                    created_at=now,
                )
            )


__all__ = [
    "SqliteStore",
    "StoreNotInitialised",
    "open_store",
    "open_store_read_only",
]
