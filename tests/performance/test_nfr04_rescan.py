"""NFR-04: a rescan of 1000 unchanged files takes at most 5 seconds and does no work.

Measured on the real filesystem and a real SQLite database (nothing is mocked); the
parser is a spy, and the logical content of the database is hashed before and after.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_engine.cli import main
from rag_engine.domain import IngestionJob, JobStatus
from rag_engine.ingestion.jobs import Worker
from rag_engine.ingestion.service import IngestReport
from rag_engine.testing import FakeClock
from tests.e2e.test_cli_e2e import ContainerForTests
from tests.workspace import Workspace, logical_dump

FILE_COUNT = 1000
BUDGET_SECONDS = 5.0


class SpyParser:
    """Stands in for the Phase 3 parsers: counts how often a document would be parsed."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
        del job, heartbeat
        self.calls += 1


@pytest.mark.performance
def test_rescan_of_1000_unchanged_files(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(FILE_COUNT):
        workspace.write(f"section{i % 20:02d}/doc{i:04d}.md", f"# document {i}\n\nbody {i}\n")

    started = time.perf_counter()
    first = workspace.run()
    first_seconds = time.perf_counter() - started
    assert isinstance(first, IngestReport) and first.new == FILE_COUNT

    parser = SpyParser()
    outcomes = Worker(workspace.queue, parser).run_until_idle(max_jobs=FILE_COUNT + 10)
    assert len(outcomes) == FILE_COUNT and parser.calls == FILE_COUNT
    assert workspace.queue.snapshot() == {JobStatus.DONE: FILE_COUNT}

    before = logical_dump(workspace.store)
    started = time.perf_counter()
    again = workspace.run()
    rescan_seconds = time.perf_counter() - started
    assert isinstance(again, IngestReport)

    assert (again.unchanged, again.new, again.modified, again.deleted) == (FILE_COUNT, 0, 0, 0)
    assert again.jobs_enqueued == 0
    assert Worker(workspace.queue, parser).run_until_idle() == []
    assert parser.calls == FILE_COUNT  # not a single additional parser call
    assert logical_dump(workspace.store) == before  # identical logical database
    with capsys.disabled():
        print(
            f"\nNFR-04: first scan of {FILE_COUNT} files {first_seconds:.3f}s; "
            f"rescan {rescan_seconds:.3f}s (budget {BUDGET_SECONDS}s)"
        )
    assert rescan_seconds <= BUDGET_SECONDS


@pytest.mark.performance
def test_rescan_through_the_command_line_stays_within_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(FILE_COUNT):
        (root / f"doc{i:04d}.md").write_text(f"# {i}\n", encoding="utf-8")
    container = ContainerForTests(tmp_path / "state" / "rag.db", FakeClock())
    assert main(["source", "add", "corpus", str(root)], container=container) == 0
    assert main(["ingest"], container=container) == 0
    capsys.readouterr()
    started = time.perf_counter()
    assert main(["ingest"], container=container) == 0
    elapsed = time.perf_counter() - started
    assert f"{FILE_COUNT} unchanged" in capsys.readouterr().out
    with capsys.disabled():
        print(f"\nNFR-04 (CLI): rescan {elapsed:.3f}s (budget {BUDGET_SECONDS}s)")
    assert elapsed <= BUDGET_SECONDS
