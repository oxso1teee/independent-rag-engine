from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from hypothesis import HealthCheck, settings
from structlog.contextvars import clear_contextvars

settings.register_profile(
    "deterministic",
    derandomize=True,
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("deterministic")

_MARKERS_BY_DIR = {"unit": "unit", "contract": "contract", "architecture": "architecture"}


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
