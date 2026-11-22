"""P2.4: change detection against the stored sync state."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import BinaryIO

import pytest

from rag_engine.domain import (
    DocumentStatus,
    ErrorCode,
    JobKind,
    JobStatus,
    SourceKey,
    SourceRevision,
    SyncStatus,
)
from rag_engine.ingestion.service import IngestReport
from rag_engine.ports import ScanReport, SourceAdapter
from tests.integration.conftest import Workspace, logical_dump


class CountingAdapter:
    """Spy around a real adapter: counts directory scans and file reads."""

    def __init__(self, inner: SourceAdapter) -> None:
        self._inner = inner
        self.scans = 0
        self.opens: list[str] = []

    def identity(self):  # type: ignore[no-untyped-def]
        return self._inner.identity()

    def scan(self, previous: Mapping[SourceKey, SourceRevision]) -> ScanReport:
        self.scans += 1
        return self._inner.scan(previous)

    def discover(self, previous: Mapping[SourceKey, SourceRevision]):  # type: ignore[no-untyped-def]
        return self._inner.discover(previous)

    def open(self, key: SourceKey) -> BinaryIO:
        self.opens.append(key.value)
        return self._inner.open(key)


@pytest.fixture
def spied(make_workspace: Callable[..., Workspace]) -> tuple[Workspace, list[CountingAdapter]]:
    spies: list[CountingAdapter] = []

    def wrap(adapter: SourceAdapter) -> SourceAdapter:
        spy = CountingAdapter(adapter)
        spies.append(spy)
        return spy

    return make_workspace(wrap=wrap), spies


def report(ws: Workspace, **options: bool) -> IngestReport:
    result = ws.run(**options)
    assert isinstance(result, IngestReport)
    return result


def test_first_scan_finds_new_documents_and_plans_one_job_each(workspace: Workspace) -> None:
    workspace.write("a.md", "# A")
    workspace.write("b/c.txt", "text")
    result = report(workspace)
    assert (result.new, result.modified, result.unchanged, result.deleted) == (2, 0, 0, 0)
    assert result.jobs_enqueued == 2
    reader = workspace.store.reader()
    documents = reader.list_documents(workspace.source.id)
    assert [d.source_key.value for d in documents] == ["a.md", "b/c.txt"]
    assert {d.status for d in documents} == {DocumentStatus.PENDING}
    states = reader.list_sync_states(workspace.source.id)
    assert {s.status for s in states} == {SyncStatus.CHANGED}
    assert all(s.last_content_hash is not None and s.last_revision is not None for s in states)
    jobs = reader.list_jobs(JobStatus.QUEUED, JobKind.INGEST_DOC)
    assert len(jobs) == 2
    assert {j.document_id for j in jobs} == {d.id for d in documents}
    assert all(j.payload["mime"] for j in jobs)


def test_unchanged_rescan_reads_nothing_and_writes_nothing(
    spied: tuple[Workspace, list[CountingAdapter]],
) -> None:
    ws, spies = spied
    for i in range(20):
        ws.write(f"dir/{i}.md", f"# doc {i}")
    first = report(ws)
    assert first.new == 20
    before = logical_dump(ws.store)
    opens_before = sum(len(s.opens) for s in spies)

    for _ in range(3):
        again = report(ws)
        assert (again.new, again.modified, again.deleted) == (0, 0, 0)
        assert again.unchanged == 20
        assert again.jobs_enqueued == 0

    assert sum(len(s.opens) for s in spies) == opens_before  # not a single file was read again
    assert logical_dump(ws.store) == before  # and the database did not change at all


def test_unchanged_rescan_does_not_even_open_a_transaction(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.write("a.md")
    report(workspace)
    begun: list[int] = []
    uow_type = type(workspace.store.unit_of_work())
    original = uow_type.begin

    def counting(self: object):  # type: ignore[no-untyped-def]
        begun.append(1)
        return original(self)  # type: ignore[arg-type]

    monkeypatch.setattr(uow_type, "begin", counting)
    report(workspace)
    assert begun == []


def test_document_id_is_stable_across_modifications(workspace: Workspace) -> None:
    workspace.write("a.md", "one")
    report(workspace)
    original = workspace.store.reader().get_document_by_key(workspace.source.id, SourceKey("a.md"))
    assert original is not None
    for text in ("two!", "three!!"):
        workspace.write("a.md", text)
        result = report(workspace)
        assert (result.modified, result.new) == (1, 0)
        current = workspace.store.reader().get_document_by_key(
            workspace.source.id, SourceKey("a.md")
        )
        assert current is not None and current.id == original.id


def test_modification_updates_state_and_does_not_duplicate_the_job(workspace: Workspace) -> None:
    workspace.write("a.md", "one")
    report(workspace)
    workspace.write("a.md", "something else")
    result = report(workspace)
    assert result.modified == 1
    assert result.jobs_enqueued == 0  # the open job for this document already covers it
    assert len(workspace.store.reader().list_jobs()) == 1


def test_mtime_only_change_refreshes_the_signal_without_reprocessing(
    spied: tuple[Workspace, list[CountingAdapter]],
) -> None:
    ws, spies = spied
    path = ws.write("a.md", "same bytes")
    report(ws)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    opens_before = sum(len(s.opens) for s in spies)
    result = report(ws)
    assert (result.modified, result.unchanged, result.jobs_enqueued) == (0, 1, 0)
    assert sum(len(s.opens) for s in spies) == opens_before + 1  # hashed once to be sure
    state = ws.store.reader().get_sync_state(ws.source.id, SourceKey("a.md"))
    assert state is not None and state.last_revision is not None
    assert state.last_revision.mtime_ns == stat.st_mtime_ns + 5_000_000_000
    before = logical_dump(ws.store)
    report(ws)
    assert sum(len(s.opens) for s in spies) == opens_before + 1  # the new signal is trusted
    assert logical_dump(ws.store) == before


def test_deleted_file_becomes_a_tombstone(workspace: Workspace) -> None:
    for name in ("a.md", "b.md", "c.md", "d.md", "e.md", "f.md", "g.md", "h.md", "i.md", "j.md"):
        workspace.write(name)
    report(workspace)
    (workspace.root / "j.md").unlink()
    result = report(workspace)
    assert result.deleted == 1
    document = workspace.store.reader().get_document_by_key(workspace.source.id, SourceKey("j.md"))
    assert document is not None and document.status is DocumentStatus.MISSING
    assert document.missing_since == workspace.clock.now()
    state = workspace.store.reader().get_sync_state(workspace.source.id, SourceKey("j.md"))
    assert state is not None and state.status is SyncStatus.MISSING
    # idempotent: the tombstone is not touched again
    before = logical_dump(workspace.store)
    assert report(workspace).deleted == 0
    assert logical_dump(workspace.store) == before


def test_unsupported_format_is_recorded_and_never_scheduled(workspace: Workspace) -> None:
    workspace.write("image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    workspace.write("ok.txt", "fine")
    result = report(workspace)
    assert (result.unsupported, result.new, result.jobs_enqueued) == (1, 1, 1)
    reader = workspace.store.reader()
    document = reader.get_document_by_key(workspace.source.id, SourceKey("image.png"))
    assert document is not None and document.status is DocumentStatus.FAILED
    state = reader.get_sync_state(workspace.source.id, SourceKey("image.png"))
    assert state is not None
    assert state.status is SyncStatus.FAILED
    assert state.last_error_code is ErrorCode.E_UNSUPPORTED_FORMAT
    before = logical_dump(workspace.store)
    again = report(workspace)
    assert (again.unsupported, again.new, again.modified) == (0, 0, 0)
    assert logical_dump(workspace.store) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_inaccessible_file_keeps_its_document_and_recovers(workspace: Workspace) -> None:
    path = workspace.write("secret.md", "v1")
    other = workspace.write("other.md", "o")
    del other
    report(workspace)
    path.write_text("changed while locked")
    path.chmod(0o000)
    try:
        result = report(workspace)
        assert result.inaccessible == 1
        document = workspace.store.reader().get_document_by_key(
            workspace.source.id, SourceKey("secret.md")
        )
        assert document is not None and document.status is not DocumentStatus.MISSING
        state = workspace.store.reader().get_sync_state(workspace.source.id, SourceKey("secret.md"))
        assert state is not None and state.status is SyncStatus.FAILED
        before = logical_dump(workspace.store)
        report(workspace)
        assert logical_dump(workspace.store) == before  # a known failure is not re-recorded
    finally:
        path.chmod(0o644)
    recovered = report(workspace)
    assert recovered.modified == 1
    state = workspace.store.reader().get_sync_state(workspace.source.id, SourceKey("secret.md"))
    assert state is not None and state.status is SyncStatus.CHANGED


def test_reappearing_file_within_the_grace_period_returns_with_the_same_id(
    workspace: Workspace,
) -> None:
    for i in range(10):
        workspace.write(f"{i}.md", f"doc {i}")
    report(workspace)
    reader = workspace.store.reader()
    original = reader.get_document_by_key(workspace.source.id, SourceKey("3.md"))
    assert original is not None
    (workspace.root / "3.md").unlink()
    report(workspace)
    workspace.write("3.md", "doc 3")
    result = report(workspace)
    assert result.restored == 1 and result.new == 0
    back = reader.get_document_by_key(workspace.source.id, SourceKey("3.md"))
    assert back is not None
    assert back.id == original.id
    assert back.status is DocumentStatus.PENDING and back.missing_since is None
    state = reader.get_sync_state(workspace.source.id, SourceKey("3.md"))
    assert state is not None and state.status is SyncStatus.CHANGED
    assert len(reader.list_jobs(JobStatus.QUEUED)) == 10


def test_results_are_deterministic_for_the_same_input(
    make_workspace: Callable[..., Workspace],
) -> None:
    ws = make_workspace()
    for i in range(5):
        ws.write(f"f{i}.md", str(i))
    first = ws.ingest.plan(ws.source.name)
    second = ws.ingest.plan(ws.source.name)
    assert first == second


def test_dry_run_changes_nothing(workspace: Workspace) -> None:
    workspace.write("a.md")
    before = logical_dump(workspace.store)
    planned = workspace.ingest.plan(workspace.source.name)
    assert planned.dry_run and planned.new == 1 and planned.jobs_enqueued == 0
    assert logical_dump(workspace.store) == before
    assert workspace.store.reader().list_documents(workspace.source.id) == []
