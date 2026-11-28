"""Read connections of threads that have ended are released.

A server answers requests on worker threads that come and go. Each thread that reads gets its own
read connection; they used to be kept until the store closed, so a long-running API process leaked
two file descriptors (database and write-ahead log) for every worker thread that ever read. Found
by the hard soak: descriptors grew from 19 to 62 in ten minutes of concurrent searches.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from rag_engine.storage.sqlite import SqliteStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc"),
]


def open_handles(path: Path) -> int:
    count = 0
    for entry in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{entry}")
        except OSError:
            continue
        count += target.startswith(str(path))
    return count


def read_once(store: SqliteStore) -> None:
    thread = threading.Thread(target=store.reader)
    thread.start()
    thread.join()


def test_connections_of_finished_threads_do_not_pile_up(store: SqliteStore, tmp_path: Path) -> None:
    store.reader()  # the main thread's own connection
    before = open_handles(tmp_path / "rag.db")
    for _ in range(30):
        read_once(store)
    assert open_handles(tmp_path / "rag.db") <= before + 3 * 2  # a few live ones, never 30


def test_a_live_threads_connection_is_kept_and_everything_closes_with_the_store(
    store: SqliteStore, tmp_path: Path
) -> None:
    ready, release = threading.Event(), threading.Event()
    seen: list[object] = []

    def hold() -> None:
        seen.append(store.reader())
        ready.set()
        release.wait(10)
        seen.append(store.reader())  # still usable after others came and went

    holder = threading.Thread(target=hold)
    holder.start()
    ready.wait(10)
    for _ in range(5):
        read_once(store)
    release.set()
    holder.join()
    assert len(seen) == 2
    store.close()
    assert open_handles(tmp_path / "rag.db") == 0
