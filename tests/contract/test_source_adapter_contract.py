"""Contract every ``SourceAdapter`` implementation must satisfy."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_engine.domain import ErrorCode, RagError, SourceKey
from rag_engine.ports import SourceAdapter, SourceRevision
from rag_engine.testing import InMemorySource

FILES = {"b/note.md": "# B", "a.txt": "alpha", "Папка/файл.md": "содержимое"}

FACTORIES: dict[str, Callable[[], SourceAdapter]] = {
    "in-memory": lambda: InMemorySource(FILES),
}


@pytest.fixture(params=sorted(FACTORIES))
def source(request: pytest.FixtureRequest) -> SourceAdapter:
    return FACTORIES[request.param]()


def test_satisfies_the_protocol(source: SourceAdapter) -> None:
    assert isinstance(source, SourceAdapter)


def test_discovery_is_sorted_and_complete(source: SourceAdapter) -> None:
    keys = [entry.key.value for entry in source.discover({})]
    assert keys == sorted(keys)
    assert set(keys) == set(FILES)


def test_discovery_is_repeatable(source: SourceAdapter) -> None:
    assert tuple(source.discover({})) == tuple(source.discover({}))


def test_previous_state_never_hides_entries(source: SourceAdapter) -> None:
    previous = {e.key: e.revision for e in source.discover({})}
    assert len(tuple(source.discover(previous))) == len(FILES)


def test_open_returns_the_exact_bytes(source: SourceAdapter) -> None:
    for entry in source.discover({}):
        with source.open(entry.key) as handle:
            data = handle.read()
        assert len(data) == entry.revision.size_bytes


def test_missing_entry_raises_the_documented_error(source: SourceAdapter) -> None:
    with pytest.raises(RagError) as info:
        source.open(SourceKey("does/not/exist.md"))
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert info.value.context["key"] == "does/not/exist.md"


def test_in_memory_source_tracks_changes() -> None:
    source = InMemorySource({"a.md": "one"})
    (before,) = source.discover({})
    source.put("a.md", "one!")
    (after,) = source.discover({})
    assert after.revision != before.revision
    assert after.revision.size_bytes == 4
    assert after.revision.mtime_ns > before.revision.mtime_ns
    source.delete("a.md")
    source.delete("a.md")  # idempotent
    assert source.discover({}) == ()
    assert source.identity().locator_fingerprint is None


def test_in_memory_source_normalises_keys_and_accepts_bytes() -> None:
    source = InMemorySource()
    key = source.put("dir\\café.bin", b"\x00\x01")
    assert key.value == "dir/café.bin"
    with source.open(key) as handle:
        assert handle.read() == b"\x00\x01"


def test_revision_rejects_negative_values() -> None:
    with pytest.raises(RagError):
        SourceRevision(-1, 0)
