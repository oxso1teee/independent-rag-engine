"""Connections, capability checks and the single-writer lock."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import time
from pathlib import Path
from typing import IO

from rag_engine.domain import ErrorCode, RagError

MIN_SQLITE_VERSION = (3, 35, 0)  # RETURNING, partial and expression indexes, VACUUM INTO


def check_capabilities(version: tuple[int, ...] | None = None) -> None:
    """Fail early when the linked SQLite library is too old."""
    found = sqlite3.sqlite_version_info if version is None else version
    if tuple(found) < MIN_SQLITE_VERSION:
        raise RagError(
            ErrorCode.E_RESOURCE_GATE,
            "the SQLite library is too old",
            context={
                "found": ".".join(map(str, found)),
                "required": ".".join(map(str, MIN_SQLITE_VERSION)),
            },
        )


def connect(path: Path, *, busy_timeout_ms: int, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection in autocommit mode with the engine's pragmas applied."""
    check_capabilities()
    try:
        if read_only:
            conn = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro",
                uri=True,
                isolation_level=None,
                check_same_thread=False,
            )
        else:
            conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys = ON")
        if read_only:
            conn.execute("PRAGMA query_only = ON")
    except sqlite3.Error as exc:
        raise RagError(
            ErrorCode.E_STORE_LOCKED if "locked" in str(exc) else ErrorCode.E_INTEGRITY,
            "the database cannot be opened",
            context={"path": str(path), "sqlite_error": type(exc).__name__},
        ) from exc
    return conn


def enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to write-ahead logging; refuse to run where WAL is unavailable."""
    mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        raise RagError(
            ErrorCode.E_RESOURCE_GATE,
            "write-ahead logging is not available for this database location",
            context={"journal_mode": mode},
        )
    conn.execute("PRAGMA synchronous = NORMAL")


class WriterLock:
    """Exclusive, process-wide writer lock backed by ``flock`` on a sidecar file.

    ``flock`` locks belong to the open file description, so a second writer is refused
    even within the same process. The operating system drops the lock when the holder
    dies, so a crashed writer never leaves a stale lock behind.
    """

    def __init__(self, path: Path, timeout_s: float = 0.0) -> None:
        self._path = path
        self._timeout_s = timeout_s
        self._handle: IO[bytes] | None = None

    def acquire(self) -> None:
        handle = self._path.open("ab")
        deadline = time.monotonic() + self._timeout_s
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise RagError(
                        ErrorCode.E_STORE_LOCKED,
                        "another writer holds the store",
                        context={"lock": self._path.name},
                    ) from None
                time.sleep(0.01)
        self._handle = handle
        # Informational only (which process holds the lock); the flock is the lock.
        os.ftruncate(handle.fileno(), 0)
        os.write(handle.fileno(), f"{os.getpid()}\n".encode())

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
