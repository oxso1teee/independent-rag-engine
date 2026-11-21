"""Duration measurement on top of the :class:`Clock` port."""

from __future__ import annotations

from rag_engine.ports import Clock


class Timer:
    """Measures elapsed time with the injected monotonic clock."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._start_ns = clock.monotonic_ns()

    def elapsed_ms(self) -> int:
        return (self._clock.monotonic_ns() - self._start_ns) // 1_000_000
