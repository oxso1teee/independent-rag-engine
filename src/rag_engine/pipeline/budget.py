"""One time budget for a whole request, shared out between its stages.

A request has a total allowance. Retrieval and context assembly may each use at most their share of
it, so a slow search cannot leave generation no time at all, and generation gets whatever is left.
Time is read from the injected monotonic clock, so tests move it by hand. The budget is
cooperative: it bounds every await, and is checked between stages, but cannot interrupt a
synchronous call that never yields.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_engine.domain import ErrorCode, RagError, require
from rag_engine.ports import Clock

_NS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class StageShares:
    retrieval: float = 0.25
    context: float = 0.05

    def __post_init__(self) -> None:
        require(
            0.0 < self.retrieval <= 1.0 and 0.0 < self.context <= 1.0, "shares must be in (0, 1]"
        )
        require(self.retrieval + self.context < 1.0, "generation needs a share of its own")


class TimeBudget:
    def __init__(self, total_s: float, clock: Clock, shares: StageShares | None = None) -> None:
        require(total_s > 0, "the time budget must be positive")
        self._total_ns = int(total_s * _NS)
        self._clock = clock
        self._shares = shares or StageShares()
        self._started = clock.monotonic_ns()
        self._stage_ms: dict[str, int] = {}

    @property
    def total_s(self) -> float:
        return self._total_ns / _NS

    def elapsed_ms(self) -> int:
        return (self._clock.monotonic_ns() - self._started) // 1_000_000

    def remaining_s(self) -> float:
        return max(0.0, (self._total_ns - (self._clock.monotonic_ns() - self._started)) / _NS)

    def allowance_s(self, stage: str) -> float:
        """Seconds the stage may still use: its share of the total, never more than what is left."""
        share = {"retrieval": self._shares.retrieval, "context": self._shares.context}.get(
            stage, 1.0
        )
        return min(self.remaining_s(), self.total_s * share)

    def timeout(self, stage: str) -> RagError:
        return RagError(
            ErrorCode.E_TIMEOUT,
            f"the {stage} stage ran out of time",
            context={"stage": stage, "total_s": self.total_s, "elapsed_ms": self.elapsed_ms()},
        )

    def require_time(self, stage: str) -> float:
        """The stage's allowance in seconds; ``E_TIMEOUT`` when none is left."""
        allowance = self.allowance_s(stage)
        if allowance <= 0:
            raise self.timeout(stage)
        return allowance

    def record(self, stage: str, started_ns: int) -> None:
        self._stage_ms[stage] = (self._clock.monotonic_ns() - started_ns) // 1_000_000

    def now_ns(self) -> int:
        return self._clock.monotonic_ns()

    @property
    def stage_ms(self) -> dict[str, int]:
        return dict(self._stage_ms)
