"""Contract of the ``Clock`` port, exercised on ``FakeClock``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rag_engine.domain import RagError, Uuid7Generator
from rag_engine.ports import Clock
from rag_engine.testing import FakeClock


def test_satisfies_the_protocol() -> None:
    assert isinstance(FakeClock(), Clock)


def test_time_only_moves_when_advanced() -> None:
    clock = FakeClock(datetime(2026, 5, 1, 12, 0, tzinfo=UTC))
    start_now, start_mono = clock.now(), clock.monotonic_ns()
    assert clock.now() == start_now
    assert clock.monotonic_ns() == start_mono
    clock.advance(timedelta(seconds=2, milliseconds=500))
    assert clock.now() == start_now + timedelta(seconds=2.5)
    assert clock.monotonic_ns() - start_mono == 2_500_000_000
    assert clock.now().utcoffset() == timedelta(0)


def test_cannot_go_backwards_or_start_naive() -> None:
    with pytest.raises(RagError):
        FakeClock().advance(timedelta(seconds=-1))
    with pytest.raises(RagError):
        FakeClock(datetime(2026, 1, 1))  # noqa: DTZ001


def test_feeds_deterministic_identifiers() -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    assert clock.unix_ms() == 1_767_225_600_000
    generator = Uuid7Generator(clock.unix_ms, lambda size: bytes(size))
    first = generator.next()
    clock.advance(timedelta(milliseconds=5))
    second = generator.next()
    assert first.int >> 80 == 1_767_225_600_000
    assert second.int >> 80 == 1_767_225_600_005
    assert first.bytes < second.bytes
