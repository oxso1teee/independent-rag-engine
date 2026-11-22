from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from hypothesis import HealthCheck, settings
from structlog.contextvars import clear_contextvars

from rag_engine.domain import IdFactory
from rag_engine.storage.sqlite import SqliteStore, open_store
from rag_engine.testing import FakeClock
from tests.factories import make_ids
from tests.workspace import make_workspace, workspace

settings.register_profile(
    "deterministic",
    derandomize=True,
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("deterministic")

_MARKERS_BY_DIR = {
    "unit": "unit",
    "contract": "contract",
    "architecture": "architecture",
    "integration": "integration",
    "e2e": "e2e",
    "performance": "performance",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        parent = Path(str(item.fspath)).parent.name
        marker = _MARKERS_BY_DIR.get(parent)
        if marker is not None:
            item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture(autouse=True)
def _isolated_logging() -> Iterator[None]:
    clear_contextvars()
    yield
    clear_contextvars()
    structlog.reset_defaults()


@pytest.fixture(autouse=True)
def _clean_rag_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("RAG_"):
            monkeypatch.delenv(name)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ids() -> IdFactory:
    return make_ids()


@pytest.fixture
def store(tmp_path: Path, clock: FakeClock) -> Iterator[SqliteStore]:
    with open_store(tmp_path / "rag.db", clock) as opened:
        yield opened


__all__ = ["make_workspace", "workspace"]  # fixtures shared by integration and performance tests
