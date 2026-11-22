"""P2.3: FilesystemSource on a real directory tree."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from rag_engine.domain import ErrorCode, RagError, SourceKey, SourceKind, SourcePolicy
from rag_engine.ingestion.sources import FilesystemSource, locator_fingerprint, normalize_locator


def write(root: Path, name: str, data: str | bytes = "x") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def keys(source: FilesystemSource) -> list[str]:
    return [e.key.value for e in source.scan({}).entries]


def test_discovers_files_recursively_in_sorted_order(tmp_path: Path) -> None:
    for name in ("b/z.md", "a.txt", "b/a/deep.md", "Ж.md"):
        write(tmp_path, name)
    assert keys(FilesystemSource(tmp_path)) == sorted(["a.txt", "b/a/deep.md", "b/z.md", "Ж.md"])


def test_revision_reflects_size_and_mtime(tmp_path: Path) -> None:
    path = write(tmp_path, "a.txt", "hello")
    os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_123_456_789))
    (entry,) = FilesystemSource(tmp_path).scan({}).entries
    assert entry.revision.size_bytes == 5
    assert entry.revision.mtime_ns == 1_700_000_000_123_456_789


def test_include_and_exclude_rules(tmp_path: Path) -> None:
    for name in ("a.md", "b.txt", "drafts/c.md", "node_modules/d.md", "sub/e.md", "sub/f.tmp.md"):
        write(tmp_path, name)
    policy = SourcePolicy(include=("**/*.md",), exclude=("drafts/**", "node_modules", "*.tmp.md"))
    assert keys(FilesystemSource(tmp_path, policy)) == ["a.md", "sub/e.md"]


def test_invalid_glob_in_the_policy_is_rejected_at_construction(tmp_path: Path) -> None:
    with pytest.raises(RagError) as info:
        FilesystemSource(tmp_path, SourcePolicy(include=("[bad",)))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_large_files_are_skipped_not_read(tmp_path: Path) -> None:
    write(tmp_path, "small.txt", "ok")
    write(tmp_path, "big.bin", b"x" * 5000)
    report = FilesystemSource(tmp_path, SourcePolicy(max_file_bytes=1000)).scan({})
    assert [e.key.value for e in report.entries] == ["small.txt"]
    assert [(i.path, i.reason) for i in report.skipped] == [("big.bin", "too_large")]


def test_symlinks_are_skipped_by_default(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write(root, "real.txt")
    write(tmp_path, "outside.txt", "secret")
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    (root / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    report = FilesystemSource(root).scan({})
    assert [e.key.value for e in report.entries] == ["real.txt"]
    reasons = {i.path: i.reason for i in report.skipped}
    assert reasons == {"link.txt": "symlink_skipped", "dirlink": "symlink_skipped"}


def test_followed_symlinks_must_stay_inside_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write(root, "target/inner.txt", "inside")
    write(tmp_path, "outside.txt", "secret")
    (root / "alias.txt").symlink_to(root / "target" / "inner.txt")
    (root / "escape.txt").symlink_to(tmp_path / "outside.txt")
    (root / "escape_dir").symlink_to(tmp_path, target_is_directory=True)
    source = FilesystemSource(root, SourcePolicy(follow_symlinks=True))
    report = source.scan({})
    found = [e.key.value for e in report.entries]
    assert "alias.txt" in found
    assert "escape.txt" not in found
    assert not any(k.startswith("escape_dir") for k in found)
    assert {"escape.txt", "escape_dir"} <= {
        i.path for i in report.skipped if i.reason == "symlink_escape"
    }
    with pytest.raises(RagError):
        source.open(SourceKey("escape.txt"))


def test_symlink_cycles_do_not_hang_the_scan(tmp_path: Path) -> None:
    write(tmp_path, "a/file.txt")
    (tmp_path / "a" / "loop").symlink_to(tmp_path / "a", target_is_directory=True)
    report = FilesystemSource(tmp_path, SourcePolicy(follow_symlinks=True)).scan({})
    assert [e.key.value for e in report.entries] == ["a/file.txt"]
    assert any(i.reason == "directory_cycle" for i in report.skipped)


def test_keys_are_nfc_normalised_even_for_decomposed_names(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "café.md")
    composed = unicodedata.normalize("NFC", "café.md")
    assert decomposed != composed
    write(tmp_path, decomposed, "content")
    source = FilesystemSource(tmp_path)
    (entry,) = source.scan({}).entries
    assert entry.key.value == composed
    with source.open(entry.key) as handle:
        assert handle.read() == b"content"
    # opening works even without a prior scan (the on-disk form is looked up)
    with FilesystemSource(tmp_path).open(SourceKey(composed)) as handle:
        assert handle.read() == b"content"


def test_names_that_normalise_to_the_same_key_are_not_guessed(tmp_path: Path) -> None:
    write(tmp_path, unicodedata.normalize("NFC", "é.md"), "one")
    write(tmp_path, unicodedata.normalize("NFD", "é.md"), "two")
    write(tmp_path, "ok.md")
    report = FilesystemSource(tmp_path).scan({})
    assert [e.key.value for e in report.entries] == ["ok.md"]
    assert [i.reason for i in report.skipped] == ["key_collision"]


def test_names_that_are_not_valid_unicode_are_skipped(tmp_path: Path) -> None:
    bad = os.fsencode(tmp_path) + b"/bad\xff\xfe.txt"
    with open(bad, "wb") as handle:
        handle.write(b"x")
    write(tmp_path, "good.txt")
    report = FilesystemSource(tmp_path).scan({})
    assert [e.key.value for e in report.entries] == ["good.txt"]
    assert [i.reason for i in report.skipped] == ["invalid_name"]


def test_names_with_backslashes_are_skipped(tmp_path: Path) -> None:
    write(tmp_path, "we\\ird.txt")
    write(tmp_path, "good.txt")
    report = FilesystemSource(tmp_path).scan({})
    assert [e.key.value for e in report.entries] == ["good.txt"]
    assert [i.reason for i in report.skipped] == ["invalid_name"]


def test_special_files_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, "a.txt")
    os.mkfifo(tmp_path / "pipe")
    report = FilesystemSource(tmp_path).scan({})
    assert [e.key.value for e in report.entries] == ["a.txt"]
    assert [(i.path, i.reason) for i in report.skipped] == [("pipe", "special_file")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_unreadable_directories_are_reported_as_subtrees_not_deletions(tmp_path: Path) -> None:
    write(tmp_path, "ok.txt")
    write(tmp_path, "locked/inner.txt")
    (tmp_path / "locked").chmod(0o000)
    try:
        report = FilesystemSource(tmp_path).scan({})
    finally:
        (tmp_path / "locked").chmod(0o755)
    assert [e.key.value for e in report.entries] == ["ok.txt"]
    assert [(i.path, i.subtree) for i in report.unreadable] == [("locked", True)]
    assert report.covers(SourceKey("locked/inner.txt"))
    assert not report.covers(SourceKey("ok.txt"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_unreadable_files_are_reported_and_open_fails_cleanly(tmp_path: Path) -> None:
    path = write(tmp_path, "secret.txt")
    path.chmod(0o000)
    try:
        source = FilesystemSource(tmp_path)
        # stat still works, so the entry is listed; reading it is what fails
        assert keys(source) == ["secret.txt"]
        with pytest.raises(RagError) as info:
            source.open(SourceKey("secret.txt"))
    finally:
        path.chmod(0o644)
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert info.value.context["reason"] == "permission_denied"


def test_missing_root_is_reported_as_source_not_found(tmp_path: Path) -> None:
    source = FilesystemSource(tmp_path / "gone")
    with pytest.raises(RagError) as info:
        source.scan({})
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert str(tmp_path) not in info.value.message


def test_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "file.txt")
    with pytest.raises(RagError) as info:
        FilesystemSource(path).scan({})
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND


def test_open_cannot_leave_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write(root, "a.txt")
    write(tmp_path, "outside.txt", "secret")
    source = FilesystemSource(root)
    for hostile in ("../outside.txt", "/etc/passwd", "a/../../outside.txt"):
        with pytest.raises(RagError):  # rejected while building the key
            source.open(SourceKey(hostile))
    with pytest.raises(RagError) as info:
        source.open(SourceKey("missing.txt"))
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND


def test_open_refuses_a_path_that_became_a_symlink_since_the_scan(tmp_path: Path) -> None:
    root = tmp_path / "root"
    path = write(root, "a.txt")
    write(tmp_path, "outside.txt", "secret")
    source = FilesystemSource(root)
    source.scan({})
    path.unlink()
    path.symlink_to(tmp_path / "outside.txt")
    with pytest.raises(RagError) as info:
        source.open(SourceKey("a.txt"))
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND


def test_a_directory_is_not_an_entry(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    source = FilesystemSource(tmp_path)
    with pytest.raises(RagError):
        source.open(SourceKey("dir"))


def test_identity_is_stable_for_equivalent_roots(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "docs", target_is_directory=True)
    direct = FilesystemSource(tmp_path / "docs").identity()
    via_link = FilesystemSource(link).identity()
    dotted = FilesystemSource(tmp_path / "docs" / ".." / "docs").identity()
    assert direct == via_link == dotted
    assert direct.kind is SourceKind.FS
    assert direct.locator_fingerprint == locator_fingerprint(normalize_locator(tmp_path / "docs"))


def test_scan_is_repeatable(tmp_path: Path) -> None:
    for name in ("a.md", "b/c.md"):
        write(tmp_path, name)
    source = FilesystemSource(tmp_path)
    assert source.scan({}) == source.scan({})
