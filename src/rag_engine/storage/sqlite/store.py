"""SQLite ``Store``: connection lifecycle, single writer, unit of work."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType

from rag_engine.domain import ErrorCode, IndexMetadata, RagError
from rag_engine.ports import Clock, MetadataReader, MetadataWriter, UnitOfWork
from rag_engine.storage.migrations import Migration, Migrator, load_packaged_migrations
from rag_engine.storage.sqlite.connection import WriterLock, connect, enable_wal
from rag_engine.storage.sqlite.repository import SqliteReader, SqliteWriter, translate

DEFAULT_BUSY_TIMEOUT_MS = 5000

# Placeholders of the Phase 2 index state; schema v2 replaces them with real settings.
_UNSET = "none"


class StoreNotInitialised(RagError):
    """The database file does not exist yet (nothing has been written so far)."""


class _SqliteUnitOfWork:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    @contextmanager
    def begin(self) -> Iterator[MetadataWriter]:
        store = self._store
        if store.read_only:
            raise RagError(ErrorCode.E_INTEGRITY, "the store was opened read-only")
        if getattr(store._local, "in_uow", False):
            raise RagError(ErrorCode.E_INTEGRITY, "nested units of work are not allowed")
        store._lock.acquire()
        store._local.in_uow = True
        try:
            conn = store._writer_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise translate(exc) from exc
            try:
                yield SqliteWriter(conn)
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise translate(exc) from exc
        finally:
            store._local.in_uow = False
            store._lock.release()


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
        self._lock = WriterLock(path.with_name(path.name + ".writer.lock"), writer_timeout_s)
        self._local = threading.local()
        self._guard = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        self._writer: sqlite3.Connection | None = None
        self._closed = False

    # -- Store port --------------------------------------------------------------------

    def reader(self) -> MetadataReader:
        conn: sqlite3.Connection | None = getattr(self._local, "reader", None)
        if conn is None:
            self._ensure_open()
            conn = connect(self.path, busy_timeout_ms=self._busy_timeout_ms, read_only=True)
            self._local.reader = conn
            with self._guard:
                self._connections.append(conn)
        return SqliteReader(conn)

    def unit_of_work(self) -> UnitOfWork:
        self._ensure_open()
        return _SqliteUnitOfWork(self)

    def close(self) -> None:
        with self._guard:
            self._closed = True
            connections, self._connections = self._connections, []
            writer, self._writer = self._writer, None
        for conn in connections:
            conn.close()
        if writer is not None:
            writer.close()
        self._local = threading.local()

    # -- helpers -----------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RagError(ErrorCode.E_INTEGRITY, "the store is closed")

    def _writer_connection(self) -> sqlite3.Connection:
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
    store = SqliteStore(path, busy_timeout_ms=busy_timeout_ms, writer_timeout_s=writer_timeout_s)
    try:
        store._lock.acquire()
        try:
            conn = store._writer_connection()
            try:
                enable_wal(conn)
                migrator.apply(conn, clock.now())
            except sqlite3.Error as exc:
                raise translate(exc) from exc
        finally:
            store._lock.release()
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


def _ensure_index_state(store: SqliteStore, now: datetime, schema_version: int) -> None:
    if store.reader().get_index_metadata() is not None:
        return
    with store.unit_of_work().begin() as writer:
        if writer.get_index_metadata() is None:
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
