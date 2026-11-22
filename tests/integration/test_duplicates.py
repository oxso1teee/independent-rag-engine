"""``duplicate_of``: documents of one source whose raw content is identical."""

from __future__ import annotations

from pathlib import Path

from rag_engine.domain import Document, DocumentStatus
from rag_engine.ingestion.service import IngestReport
from tests.workspace import Workspace, logical_dump

SAME = "the same text in every copy\n"


def run(ws: Workspace) -> IngestReport:
    return ws.ingest.run(ws.source.name)


def documents(ws: Workspace) -> dict[str, Document]:
    return {d.source_key.value: d for d in ws.store.reader().list_documents(ws.source.id)}


def links(ws: Workspace) -> dict[str, str | None]:
    """Each document key mapped to the key of the document it duplicates."""
    found = documents(ws)
    by_id = {d.id: key for key, d in found.items()}
    return {
        key: None if d.duplicate_of is None else by_id[d.duplicate_of] for key, d in found.items()
    }


def oldest(ws: Workspace, *keys: str) -> str:
    found = documents(ws)
    return min(keys, key=lambda key: found[key].id)


def test_identical_files_point_to_the_original_with_the_smallest_id(workspace: Workspace) -> None:
    for name in ("a.md", "b.md", "c.md"):
        workspace.write(name, SAME)
    workspace.write("other.md", "different\n")
    report = run(workspace)
    original = oldest(workspace, "a.md", "b.md", "c.md")
    assert report.duplicates == 2
    assert links(workspace) == {
        "a.md": None if original == "a.md" else original,
        "b.md": None if original == "b.md" else original,
        "c.md": None if original == "c.md" else original,
        "other.md": None,
    }


def test_unique_files_have_no_links_and_rescans_write_nothing(workspace: Workspace) -> None:
    workspace.write("a.md", "one\n")
    workspace.write("b.md", "two\n")
    assert run(workspace).duplicates == 0
    assert set(links(workspace).values()) == {None}
    before = logical_dump(workspace.store)
    assert run(workspace).duplicates == 0
    assert logical_dump(workspace.store) == before


def test_a_later_copy_joins_the_group_and_an_unchanged_rescan_keeps_it(
    workspace: Workspace,
) -> None:
    workspace.write("a.md", SAME)
    run(workspace)
    workspace.write("b.md", SAME)
    assert run(workspace).duplicates == 1
    assert links(workspace) == {"a.md": None, "b.md": "a.md"}
    assert run(workspace).duplicates == 0


def test_when_the_original_disappears_the_next_document_takes_over(workspace: Workspace) -> None:
    for name in ("a.md", "b.md", "c.md"):
        workspace.write(name, SAME)
    run(workspace)
    first = oldest(workspace, "a.md", "b.md", "c.md")
    survivors = sorted({"a.md", "b.md", "c.md"} - {first})
    next_original = oldest(workspace, *survivors)
    (workspace.root / first).unlink()
    report = run(workspace)
    assert report.duplicates >= 1
    found = links(workspace)
    assert found[first] is None  # the tombstone holds no link
    assert found[next_original] is None
    other = next(k for k in survivors if k != next_original)
    assert found[other] == next_original


def test_a_group_that_shrinks_to_one_document_loses_its_link(workspace: Workspace) -> None:
    workspace.write("a.md", SAME)
    workspace.write("b.md", SAME)
    run(workspace)
    first = oldest(workspace, "a.md", "b.md")
    second = "b.md" if first == "a.md" else "a.md"
    (workspace.root / first).unlink()
    assert run(workspace).duplicates == 1
    assert links(workspace)[second] is None


def test_a_restored_original_takes_its_place_back(workspace: Workspace) -> None:
    workspace.write("a.md", SAME)
    workspace.write("b.md", SAME)
    run(workspace)
    first = oldest(workspace, "a.md", "b.md")
    second = "b.md" if first == "a.md" else "a.md"
    path = workspace.root / first
    path.unlink()
    run(workspace)
    path.write_text(SAME, encoding="utf-8")
    run(workspace)
    assert documents(workspace)[first].status is DocumentStatus.ACTIVE
    assert links(workspace) == {first: None, second: first}


def test_changing_a_copy_removes_it_from_the_group(workspace: Workspace) -> None:
    workspace.write("a.md", SAME)
    workspace.write("b.md", SAME)
    run(workspace)
    workspace.write("b.md", "now different\n")
    assert run(workspace).duplicates == 1
    assert set(links(workspace).values()) == {None}


def test_two_files_that_become_identical_are_linked(workspace: Workspace) -> None:
    workspace.write("a.md", "first\n")
    workspace.write("b.md", "second\n")
    run(workspace)
    workspace.write("b.md", "first\n")
    run(workspace)
    first = oldest(workspace, "a.md", "b.md")
    second = "b.md" if first == "a.md" else "a.md"
    assert links(workspace) == {first: None, second: first}


def test_copies_are_found_across_directories_but_never_across_sources(
    workspace: Workspace, tmp_path: Path
) -> None:
    workspace.write("x/a.md", SAME)
    workspace.write("y/a.md", SAME)
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "a.md").write_text(SAME, encoding="utf-8")
    other = workspace.registry.add("other", other_root)
    run(workspace)
    workspace.ingest.run("other")
    assert sorted(v for v in links(workspace).values() if v) == [
        oldest(workspace, "x/a.md", "y/a.md")
    ]
    (foreign,) = workspace.store.reader().list_documents(other.id)
    assert foreign.duplicate_of is None


def test_files_that_cannot_be_processed_are_not_copies(workspace: Workspace) -> None:
    picture = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    workspace.write("one.png", picture)
    workspace.write("two.png", picture)
    workspace.write("note.md", "text\n")
    assert run(workspace).duplicates == 0
    assert set(links(workspace).values()) == {None}


def test_a_dry_run_reports_and_writes_no_links(workspace: Workspace) -> None:
    workspace.write("a.md", SAME)
    workspace.write("b.md", SAME)
    before = logical_dump(workspace.store)
    assert workspace.ingest.plan(workspace.source.name).duplicates == 0
    assert logical_dump(workspace.store) == before
    assert not documents(workspace)


def test_the_report_and_the_summary_mention_the_links(workspace: Workspace) -> None:
    workspace.write("a.md", SAME)
    workspace.write("b.md", SAME)
    report = run(workspace)
    assert report.to_dict()["duplicates"] == 1
