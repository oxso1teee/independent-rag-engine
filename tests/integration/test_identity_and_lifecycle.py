"""P2.5 source identity / rebase and P2.6 tombstones, grace period, mass-delete guard."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from rag_engine.domain import (
    DocumentStatus,
    ErrorCode,
    RagError,
    SourceKey,
    SourcePolicy,
    SyncStatus,
)
from rag_engine.ingestion.identity import match_moves
from rag_engine.ingestion.service import IngestReport
from tests.integration.conftest import Workspace, logical_dump


def run(ws: Workspace, **options: bool) -> IngestReport:
    result = ws.run(**options)
    assert isinstance(result, IngestReport)
    return result


def populate(ws: Workspace, count: int = 10) -> None:
    for i in range(count):
        ws.write(f"dir/doc{i:02d}.md", f"# document number {i}\n")
    run(ws)


def ids_by_key(ws: Workspace) -> dict[str, object]:
    docs = ws.store.reader().list_documents(ws.source.id)
    return {d.source_key.value: d.id for d in docs}


# ---------------------------------------------------------------- source identity


def test_adding_the_same_source_twice_is_idempotent(workspace: Workspace) -> None:
    again = workspace.registry.add("docs", workspace.root)
    assert again == workspace.source
    assert len(workspace.registry.list()) == 1


def test_adding_a_name_with_another_root_requires_an_explicit_rebase(
    workspace: Workspace, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RagError) as info:
        workspace.registry.add("docs", other)
    assert info.value.code is ErrorCode.E_SOURCE_LOCATOR_CHANGED
    assert workspace.registry.get("docs").locator == workspace.source.locator


def test_unknown_source_and_missing_root_are_reported(workspace: Workspace, tmp_path: Path) -> None:
    with pytest.raises(RagError) as unknown:
        workspace.registry.get("nope")
    assert unknown.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    with pytest.raises(RagError) as missing:
        workspace.registry.add("ghost", tmp_path / "does-not-exist")
    assert missing.value.code is ErrorCode.E_SOURCE_NOT_FOUND


def test_invalid_policy_is_rejected_before_anything_is_stored(
    workspace: Workspace, tmp_path: Path
) -> None:
    other = tmp_path / "second"
    other.mkdir()
    with pytest.raises(RagError) as info:
        workspace.registry.add("second", other, SourcePolicy(include=("[broken",)))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    assert [s.name for s in workspace.registry.list()] == ["docs"]


def test_a_renamed_file_keeps_its_document_id(workspace: Workspace) -> None:
    populate(workspace)
    before = ids_by_key(workspace)
    (workspace.root / "dir" / "doc03.md").rename(workspace.root / "dir" / "renamed.md")
    result = run(workspace)
    assert (result.moved, result.new, result.deleted) == (1, 0, 0)
    after = ids_by_key(workspace)
    assert after["dir/renamed.md"] == before["dir/doc03.md"]
    assert "dir/doc03.md" not in after
    assert len(workspace.store.reader().list_documents(workspace.source.id)) == 10
    states = workspace.store.reader().list_sync_states(workspace.source.id)
    assert {s.source_key.value for s in states} == set(after)


def test_a_moved_file_keeps_its_document_id(workspace: Workspace) -> None:
    populate(workspace)
    before = ids_by_key(workspace)
    (workspace.root / "archive").mkdir()
    shutil.move(workspace.root / "dir" / "doc05.md", workspace.root / "archive" / "doc05.md")
    result = run(workspace)
    assert result.moved == 1
    assert ids_by_key(workspace)["archive/doc05.md"] == before["dir/doc05.md"]


def test_duplicate_contents_are_never_merged_automatically(workspace: Workspace) -> None:
    workspace.write("a.md", "identical")
    workspace.write("b.md", "identical")
    for i in range(8):
        workspace.write(f"x{i}.md", f"other {i}")
    run(workspace)
    before = ids_by_key(workspace)
    (workspace.root / "a.md").rename(workspace.root / "c.md")  # two docs share the content
    result = run(workspace)
    assert result.moved == 0  # ambiguous: one vanished, one appeared, but the hash is shared
    assert (result.new, result.deleted) == (1, 1)
    after = ids_by_key(workspace)
    assert after["c.md"] != before["a.md"]
    assert after["b.md"] == before["b.md"]


def test_match_moves_requires_a_unique_pair() -> None:
    from rag_engine.domain import ContentHash, DocumentId
    from tests.factories import make_ids

    ids = make_ids()
    h1, h2 = ContentHash.of_bytes(b"1"), ContentHash.of_bytes(b"2")
    d1, d2 = ids.document_id(), ids.document_id()
    del DocumentId
    found, ambiguous = match_moves(
        [(SourceKey("a"), d1, h1), (SourceKey("b"), d2, h2)],
        [(SourceKey("x"), h1), (SourceKey("y"), h2), (SourceKey("z"), h2)],
    )
    assert [(m[0].value, m[2].value) for m in found] == [("a", "x")]
    assert ambiguous == [h2]


def test_rebase_preserves_document_ids_where_provable(workspace: Workspace, tmp_path: Path) -> None:
    populate(workspace)
    before = ids_by_key(workspace)
    new_root = tmp_path / "relocated"
    shutil.copytree(workspace.root, new_root)
    shutil.rmtree(workspace.root)
    (new_root / "dir" / "doc02.md").rename(new_root / "dir" / "two.md")  # renamed meanwhile
    (new_root / "extra.md").write_text("brand new", encoding="utf-8")

    plan = workspace.registry.rebase("docs", new_root, apply=False)
    assert (plan.kept, len(plan.moves), len(plan.unmatched_new)) == (9, 1, 1)
    assert workspace.registry.get("docs").locator == workspace.source.locator  # preview only

    workspace.registry.rebase("docs", new_root, apply=True)
    rebased = workspace.registry.get("docs")
    assert rebased.id == workspace.source.id
    assert rebased.locator == str(new_root.resolve())

    result = run(workspace)
    assert (result.new, result.deleted, result.modified) == (1, 0, 0)
    after = ids_by_key(workspace)
    assert after["dir/two.md"] == before["dir/doc02.md"]
    assert all(after[k] == v for k, v in before.items() if k != "dir/doc02.md")


def test_rebase_to_an_unrelated_root_does_not_merge_anything(
    workspace: Workspace, tmp_path: Path
) -> None:
    populate(workspace)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    for i in range(3):
        (unrelated / f"n{i}.md").write_text(f"nothing in common {i}", encoding="utf-8")
    plan = workspace.registry.rebase("docs", unrelated, apply=True)
    assert plan.kept == 0 and not plan.moves
    assert len(plan.unmatched_old) == 10
    # the old documents are still all there; the next scan is stopped by the guard
    assert len(workspace.store.reader().list_documents(workspace.source.id)) == 10
    with pytest.raises(RagError) as info:
        run(workspace)
    assert info.value.code is ErrorCode.E_MASS_DELETE_GUARD


def test_rebase_to_the_same_root_is_a_noop(workspace: Workspace) -> None:
    plan = workspace.registry.rebase("docs", workspace.root, apply=True)
    assert plan.unchanged_locator


def test_rebase_to_a_missing_root_fails_cleanly(workspace: Workspace, tmp_path: Path) -> None:
    with pytest.raises(RagError) as info:
        workspace.registry.rebase("docs", tmp_path / "nowhere", apply=True)
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert workspace.registry.get("docs").locator == workspace.source.locator


def test_scan_with_a_swapped_root_is_refused(workspace: Workspace, tmp_path: Path) -> None:
    populate(workspace, 3)
    other = tmp_path / "other"
    other.mkdir()
    # a different root behind the same registered source: identity check must catch it
    from rag_engine.ingestion.sources import FilesystemSource

    adapter = FilesystemSource(other)
    with pytest.raises(RagError) as info:
        workspace.registry.verify(workspace.source, adapter)
    assert info.value.code is ErrorCode.E_SOURCE_LOCATOR_CHANGED


# ------------------------------------------------------------------- mass delete


def test_one_missing_file_among_many_is_tombstoned(workspace: Workspace) -> None:
    populate(workspace, 10)
    (workspace.root / "dir" / "doc00.md").unlink()
    assert run(workspace).deleted == 1


def test_a_few_missing_files_below_the_threshold_are_tombstoned(workspace: Workspace) -> None:
    populate(workspace, 20)
    for i in range(4):  # exactly 20 %: not "more than 20 %"
        (workspace.root / "dir" / f"doc{i:02d}.md").unlink()
    assert run(workspace).deleted == 4


def test_more_than_twenty_percent_missing_is_blocked_and_changes_nothing(
    workspace: Workspace,
) -> None:
    populate(workspace, 10)
    for i in range(3):
        (workspace.root / "dir" / f"doc{i:02d}.md").unlink()
    before = logical_dump(workspace.store)
    with pytest.raises(RagError) as info:
        run(workspace)
    assert info.value.code is ErrorCode.E_MASS_DELETE_GUARD
    assert info.value.context["vanished"] == 3
    assert logical_dump(workspace.store) == before
    with pytest.raises(RagError):
        workspace.ingest.plan("docs")  # a dry run reports the same verdict


def test_the_guard_can_be_overridden_explicitly_and_stays_reversible(
    workspace: Workspace,
) -> None:
    populate(workspace, 10)
    originals = ids_by_key(workspace)
    hidden = workspace.root / "dir"
    moved_away = workspace.root.parent / "elsewhere"
    hidden.rename(moved_away)
    workspace.write("keep.md", "keep")
    result = run(workspace, allow_mass_delete=True)
    assert result.deleted == 10
    docs = workspace.store.reader().list_documents(workspace.source.id, DocumentStatus.MISSING)
    assert len(docs) == 10
    moved_away.rename(hidden)  # the files come back within the grace period
    restored = run(workspace)
    assert restored.restored == 10 and restored.new == 0
    assert {k: v for k, v in ids_by_key(workspace).items() if k != "keep.md"} == originals


def test_an_empty_scan_never_wipes_the_corpus(workspace: Workspace) -> None:
    populate(workspace, 5)
    shutil.rmtree(workspace.root / "dir")
    with pytest.raises(RagError) as info:
        run(workspace)
    assert info.value.code is ErrorCode.E_MASS_DELETE_GUARD
    assert len(workspace.store.reader().list_documents(workspace.source.id)) == 5


def test_a_vanished_root_is_an_error_not_a_deletion(workspace: Workspace) -> None:
    populate(workspace, 5)
    shutil.rmtree(workspace.root)
    with pytest.raises(RagError) as info:
        run(workspace)
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert (
        workspace.store.reader().count_documents(workspace.source.id, DocumentStatus.MISSING) == 0
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_an_unreadable_directory_does_not_delete_its_documents(workspace: Workspace) -> None:
    populate(workspace, 10)
    workspace.write("other/keep.md", "keep")
    run(workspace)
    locked = workspace.root / "dir"
    locked.chmod(0o000)
    try:
        result = run(workspace)
    finally:
        locked.chmod(0o755)
    assert result.deleted == 0 and result.inaccessible == 10
    reader = workspace.store.reader()
    assert reader.count_documents(workspace.source.id, DocumentStatus.MISSING) == 0
    recovered = run(workspace)
    assert (recovered.deleted, recovered.modified) == (0, 10)  # failed state re-examined
    states = reader.list_sync_states(workspace.source.id)
    assert SyncStatus.FAILED not in {s.status for s in states}


# --------------------------------------------------------------- grace period / gc


def test_garbage_collection_waits_for_the_grace_period(workspace: Workspace) -> None:
    populate(workspace, 10)
    (workspace.root / "dir" / "doc01.md").unlink()
    run(workspace)  # tombstoned now
    reader = workspace.store.reader()
    tombstone = reader.get_document_by_key(workspace.source.id, SourceKey("dir/doc01.md"))
    assert tombstone is not None

    workspace.clock.advance(timedelta(days=29, hours=23))
    assert run(workspace).gc_removed == 0
    assert reader.get_document(tombstone.id) is not None

    workspace.clock.advance(timedelta(hours=2))
    assert run(workspace).gc_removed == 1
    assert reader.get_document(tombstone.id) is None
    assert reader.get_sync_state(workspace.source.id, SourceKey("dir/doc01.md")) is None
    metadata = reader.get_index_metadata()
    assert metadata is not None and metadata.last_gc_at == workspace.clock.now()
    # and it is a new document if the file ever returns
    workspace.write("dir/doc01.md", "# document number 1\n")
    assert run(workspace).new == 1


def test_gc_honours_a_per_source_grace_period(
    make_workspace: Callable[..., Workspace], tmp_path: Path
) -> None:
    ws = make_workspace()
    short = tmp_path / "short"
    short.mkdir()
    source = ws.registry.add("short", short, SourcePolicy(delete_grace_days=0))
    for i in range(6):
        (short / f"{i}.md").write_text(f"d{i}", encoding="utf-8")
    ws.ingest.run("short")
    (short / "0.md").unlink()
    result = ws.ingest.run("short")
    assert (result.deleted, result.gc_removed) == (1, 1)  # grace of zero days: gone at once
    assert ws.store.reader().count_documents(source.id) == 5


def test_idempotent_state_after_gc(workspace: Workspace) -> None:
    populate(workspace, 10)
    (workspace.root / "dir" / "doc09.md").unlink()
    run(workspace)
    workspace.clock.advance(timedelta(days=31))
    run(workspace)
    before = logical_dump(workspace.store)
    again = run(workspace)
    assert (again.gc_removed, again.deleted, again.new) == (0, 0, 0)
    assert logical_dump(workspace.store) == before
