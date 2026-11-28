"""The log can go to a size-rotated file that only its owner can read."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from rag_engine.observability import RotatingStream, configure_logging, get_logger

pytestmark = pytest.mark.unit


def test_events_are_written_as_lines_to_a_file_readable_by_the_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "rag.log"
    stream = RotatingStream(path, max_bytes=100_000, backups=2)
    configure_logging("INFO", stream=stream)
    get_logger("test").info("demo.event", count=3)
    stream.flush()
    (line,) = path.read_text().splitlines()
    assert json.loads(line)["event"] == "demo.event"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    stream.close()


def test_the_file_rotates_by_size_and_keeps_a_bounded_number_of_backups(tmp_path: Path) -> None:
    path = tmp_path / "rag.log"
    stream = RotatingStream(path, max_bytes=400, backups=2)
    configure_logging("INFO", stream=stream)
    logger = get_logger("test")
    for number in range(60):
        logger.info("demo.event", number=number)
    stream.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["rag.log", "rag.log.1", "rag.log.2"]  # never more than the backups asked for
    assert all(p.stat().st_size <= 600 for p in tmp_path.iterdir())
