"""A process that dies by SIGKILL at a chosen point of ingestion, a job or a migration.

Run as ``python -m tests.chaos.child <point> <database> <root> [clock_days]``. Nothing is
cleaned up when the signal arrives, exactly like ``kill -9``. The points are:

* ``after_scan``      the scan and change detection finished, nothing is written yet;
* ``apply_start``     the write transaction is open, nothing is written;
* ``apply_mid``       the transaction is open and half of the documents are written;
* ``before_commit``   everything is written, the transaction is not committed;
* ``after_commit``    committed, garbage collection has not started;
* ``gc_mid``          a garbage-collection transaction is open and one document is deleted;
* ``job_mid``         a worker holds a lease and is in the middle of the handler;
* ``migration_mid``   a schema migration is half way through its statements.
"""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable, Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any

from rag_engine.container import build_services, filesystem_adapter
from rag_engine.domain import IngestionJob
from rag_engine.ingestion import change, jobs, lifecycle, service
from rag_engine.storage import migrations
from rag_engine.storage.sqlite import open_store
from rag_engine.storage.sqlite import repository as repo
from rag_engine.testing import FakeClock
from tests.factories import make_ids


def die() -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def after_returning(owner: Any, name: str, hook: Callable[[], None]) -> None:
    original = getattr(owner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        hook()
        return result

    setattr(owner, name, wrapper)


def before_calling(owner: Any, name: str, hook: Callable[[], None], nth: int = 1) -> None:
    original = getattr(owner, name)
    count = {"n": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        count["n"] += 1
        if count["n"] == nth:
            hook()
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)


def run_worker(queue: jobs.JobQueue, hook: Callable[[], None]) -> None:
    seen = {"n": 0}

    def handler(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
        del job, heartbeat
        seen["n"] += 1
        if seen["n"] == 2:
            hook()

    jobs.Worker(queue, handler).run_until_idle()


def run_migration(database: Path, clock: FakeClock) -> None:
    original = migrations.split_statements

    def dying(script: str) -> Iterable[str]:
        for index, statement in enumerate(original(script)):
            if index == 4:  # four statements ran inside the open transaction
                die()
            yield statement

    migrations.split_statements = dying  # type: ignore[assignment]
    open_store(database, clock)


def main(point: str, database: str, clock_days: int = 0) -> None:
    clock = FakeClock()
    clock.advance(timedelta(days=clock_days))
    if point == "migration_mid":
        run_migration(Path(database), clock)
        return
    store = open_store(Path(database), clock)
    ids = make_ids(2_000_000_000_000)
    services = build_services(store, Path(database), clock, ids, filesystem_adapter)

    if point == "after_scan":
        after_returning(change.ChangeDetector, "detect", die)
    elif point == "apply_start":
        before_calling(service.IngestService, "_apply", die)
    elif point == "apply_mid":
        before_calling(repo.SqliteWriter, "put_document", die, nth=4)
    elif point == "before_commit":
        after_returning(service.IngestService, "_apply", die)
    elif point == "after_commit":
        before_calling(lifecycle.Lifecycle, "collect_garbage", die)
    elif point == "gc_mid":
        before_calling(repo.SqliteWriter, "delete_document", die, nth=2)
    elif point == "job_mid":
        run_worker(services.queue, die)
        return
    else:
        raise SystemExit(f"unknown point: {point}")
    services.ingest.run("docs")
    store.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 0)
