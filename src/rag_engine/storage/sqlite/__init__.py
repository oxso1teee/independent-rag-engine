"""SQLite implementation of the metadata store."""

from rag_engine.storage.sqlite.backup import backup_database
from rag_engine.storage.sqlite.connection import MIN_SQLITE_VERSION, check_capabilities
from rag_engine.storage.sqlite.store import SqliteStore, open_store, open_store_read_only

__all__ = [
    "MIN_SQLITE_VERSION",
    "SqliteStore",
    "backup_database",
    "check_capabilities",
    "open_store",
    "open_store_read_only",
]
