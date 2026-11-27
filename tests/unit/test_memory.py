"""``trim_heap`` is best effort: it never raises and reports whether it could ask the allocator."""

from __future__ import annotations

import ctypes

import pytest

from rag_engine.system import trim_heap

pytestmark = pytest.mark.unit


def test_trim_heap_runs_and_reports_availability() -> None:
    assert isinstance(trim_heap(), bool)


def test_a_platform_without_malloc_trim_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", missing)
    assert trim_heap() is False
