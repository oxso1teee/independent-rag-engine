"""The shared time budget: shares per stage, never more than what is left."""

from __future__ import annotations

from datetime import timedelta

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.pipeline import StageShares, TimeBudget
from rag_engine.testing import FakeClock

pytestmark = pytest.mark.unit


def advance(clock: FakeClock, seconds: float) -> None:
    clock._monotonic_ns += int(seconds * 1_000_000_000)


def test_each_early_stage_gets_its_share_and_generation_gets_what_is_left() -> None:
    clock = FakeClock()
    budget = TimeBudget(100.0, clock, StageShares(retrieval=0.25, context=0.05))
    assert budget.allowance_s("retrieval") == pytest.approx(25.0)
    assert budget.allowance_s("context") == pytest.approx(5.0)
    assert budget.allowance_s("generation") == pytest.approx(100.0)
    advance(clock, 40.0)
    assert budget.allowance_s("generation") == pytest.approx(60.0)
    assert budget.allowance_s("retrieval") == pytest.approx(25.0)  # still bounded by its share
    advance(clock, 70.0)
    assert budget.allowance_s("retrieval") == 0.0 and budget.remaining_s() == 0.0


def test_a_spent_budget_is_a_timeout_that_names_the_stage_and_the_time_used() -> None:
    clock = FakeClock()
    budget = TimeBudget(10.0, clock)
    advance(clock, 11.0)
    with pytest.raises(RagError) as info:
        budget.require_time("generation")
    assert info.value.code is ErrorCode.E_TIMEOUT and info.value.retryable
    assert info.value.context["stage"] == "generation" and info.value.context["elapsed_ms"] == 11000


def test_stage_durations_are_recorded_in_milliseconds() -> None:
    clock = FakeClock()
    budget = TimeBudget(10.0, clock)
    began = budget.now_ns()
    advance(clock, 1.5)
    budget.record("retrieval", began)
    assert budget.stage_ms == {"retrieval": 1500} and budget.elapsed_ms() == 1500


@pytest.mark.parametrize("shares", [(0.0, 0.1), (0.5, 0.5), (1.5, 0.1), (0.6, 0.5)])
def test_invalid_shares_are_refused(shares: tuple[float, float]) -> None:
    with pytest.raises(RagError):
        StageShares(*shares)


def test_a_non_positive_total_is_refused() -> None:
    with pytest.raises(RagError):
        TimeBudget(0.0, FakeClock())
    assert timedelta(0) == timedelta(0)  # the fake clock needs no wall time
