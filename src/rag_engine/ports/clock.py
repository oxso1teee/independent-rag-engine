"""Time source port. Domain and application code never read the system clock directly."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Current wall-clock time as a UTC-aware datetime."""
        ...

    def monotonic_ns(self) -> int:
        """Monotonic counter in nanoseconds, for measuring durations."""
        ...
