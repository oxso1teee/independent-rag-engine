"""Production implementations of the ``Clock`` port and identifier generation."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from rag_engine.domain import IdFactory, Uuid7Generator


class SystemClock:
    """Wall clock (UTC) and monotonic counter of the running process."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


def system_id_factory() -> IdFactory:
    """Identifiers stamped with the real time and OS entropy."""
    return IdFactory(Uuid7Generator(lambda: time.time_ns() // 1_000_000, os.urandom))
