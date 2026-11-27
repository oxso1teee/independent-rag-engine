"""ResilientLLM: timeout, bounded retries, circuit breaker, serialisation and cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.generation.resilience import ResiliencePolicy, ResilientLLM
from rag_engine.ports import (
    Delta,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelLimits,
    TokenUsage,
)
from tests.factories import make_ids

pytestmark = pytest.mark.unit

REQUEST = GenerationRequest(make_ids().request_id(), "system", "prompt")
DOWN = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")
OK = GenerationResult("fine", FinishReason.STOP, TokenUsage(2, 1))


class Flaky:
    """An inner provider that fails a set number of times, then succeeds; can also be slow."""

    def __init__(
        self,
        failures: list[RagError] | None = None,
        *,
        delay: float = 0.0,
        stream_fail_after: int | None = None,
    ) -> None:
        self.failures = list(failures or [])
        self.delay = delay
        self.stream_fail_after = stream_fail_after
        self.calls = 0
        self.active = 0
        self.peak = 0

    def limits(self) -> ModelLimits:
        return ModelLimits(4096, 256)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.failures:
                raise self.failures.pop(0)
            return OK
        finally:
            self.active -= 1

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        for index, piece in enumerate(["a ", "b ", "c"]):
            if self.stream_fail_after == index:
                raise DOWN
            yield Delta(piece)


class Pause:
    """A recording replacement for ``asyncio.sleep``."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class Tick:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def wrap(inner: Flaky, **policy: float) -> tuple[ResilientLLM, Pause, Tick]:
    pause, tick = Pause(), Tick()
    config = ResiliencePolicy(**{"backoff_s": 0.5, **policy})  # type: ignore[arg-type]
    return ResilientLLM(inner, config, sleep=pause, monotonic=tick), pause, tick


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_limits_and_counting_pass_through() -> None:
    llm, _, _ = wrap(Flaky())
    assert llm.limits() == ModelLimits(4096, 256) and llm.count_tokens("a b c") == 3


def test_a_transient_failure_is_retried_with_a_growing_backoff() -> None:
    inner = Flaky([DOWN, DOWN])
    llm, pause, _ = wrap(inner)
    assert run(llm.generate(REQUEST)) == OK
    assert inner.calls == 3 and pause.waits == [0.5, 1.0]


def test_retries_are_bounded_and_the_last_error_is_raised() -> None:
    inner = Flaky([DOWN, DOWN, DOWN, DOWN])
    llm, _, _ = wrap(inner, max_retries=2)
    with pytest.raises(RagError) as info:
        run(llm.generate(REQUEST))
    assert info.value is DOWN and inner.calls == 3


def test_a_rate_limit_is_retried_but_a_permanent_error_is_not() -> None:
    limited = RagError(ErrorCode.E_RATE_LIMITED, "slow down")
    inner = Flaky([limited])
    llm, _, _ = wrap(inner)
    assert run(llm.generate(REQUEST)) == OK
    bad = Flaky([RagError(ErrorCode.E_CONFIG_INVALID, "no such model")])
    llm, _, _ = wrap(bad)
    with pytest.raises(RagError):
        run(llm.generate(REQUEST))
    assert bad.calls == 1


def test_a_slow_provider_times_out_with_a_retryable_error_and_is_not_retried() -> None:
    inner = Flaky(delay=5.0)
    llm, _, _ = wrap(inner, timeout_s=0.05)
    with pytest.raises(RagError) as info:
        run(llm.generate(REQUEST))
    assert info.value.code is ErrorCode.E_PROVIDER_TIMEOUT and info.value.retryable
    assert inner.calls == 1


def test_the_breaker_opens_after_consecutive_failures_and_fails_fast() -> None:
    inner = Flaky([DOWN] * 20)
    llm, _, _ = wrap(inner, max_retries=0, breaker_failures=3)
    for _ in range(3):
        with pytest.raises(RagError):
            run(llm.generate(REQUEST))
    before = inner.calls
    with pytest.raises(RagError) as info:
        run(llm.generate(REQUEST))
    assert info.value.code is ErrorCode.E_PROVIDER_UNAVAILABLE and "circuit" in info.value.message
    assert inner.calls == before and info.value.retryable


def test_the_breaker_allows_a_trial_after_the_reset_time_and_closes_on_success() -> None:
    inner = Flaky([DOWN] * 3)
    llm, _, tick = wrap(inner, max_retries=0, breaker_failures=3, breaker_reset_s=60.0)
    for _ in range(3):
        with pytest.raises(RagError):
            run(llm.generate(REQUEST))
    tick.now += 61
    assert run(llm.generate(REQUEST)) == OK  # the trial succeeds
    assert run(llm.generate(REQUEST)) == OK  # and the breaker stays closed


def test_a_failed_trial_reopens_the_breaker() -> None:
    inner = Flaky([DOWN] * 4)
    llm, _, tick = wrap(inner, max_retries=0, breaker_failures=3, breaker_reset_s=60.0)
    for _ in range(3):
        with pytest.raises(RagError):
            run(llm.generate(REQUEST))
    tick.now += 61
    with pytest.raises(RagError) as trial:
        run(llm.generate(REQUEST))
    assert trial.value is DOWN
    calls = inner.calls
    with pytest.raises(RagError) as fast:
        run(llm.generate(REQUEST))
    assert "circuit" in fast.value.message and inner.calls == calls


def test_a_success_resets_the_failure_count() -> None:
    inner = Flaky([DOWN, DOWN])
    llm, _, _ = wrap(inner, max_retries=0, breaker_failures=3)
    for _ in range(2):
        with pytest.raises(RagError):
            run(llm.generate(REQUEST))
    assert run(llm.generate(REQUEST)) == OK
    inner.failures = [DOWN, DOWN]
    for _ in range(2):
        with pytest.raises(RagError):
            run(llm.generate(REQUEST))
    assert run(llm.generate(REQUEST)) == OK  # never three in a row


def test_calls_are_serialised_by_default() -> None:
    inner = Flaky(delay=0.01)
    llm, _, _ = wrap(inner)

    async def both() -> None:
        await asyncio.gather(llm.generate(REQUEST), llm.generate(REQUEST), llm.generate(REQUEST))

    run(both())
    assert inner.peak == 1 and inner.calls == 3


def test_a_cancelled_call_releases_the_slot_and_does_not_trip_the_breaker() -> None:
    inner = Flaky(delay=5.0)
    llm, _, _ = wrap(inner, breaker_failures=1, timeout_s=30.0)

    async def scenario() -> GenerationResult:
        task = asyncio.create_task(llm.generate(REQUEST))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        inner.delay = 0.0
        return await llm.generate(REQUEST)

    assert run(scenario()) == OK


async def collect(llm: ResilientLLM) -> str:
    return "".join([d.text async for d in llm.stream(REQUEST)])


def test_a_stream_is_retried_before_the_first_delta() -> None:
    inner = Flaky([DOWN])
    llm, _, _ = wrap(inner)
    assert run(collect(llm)) == "a b c" and inner.calls == 2


def test_a_stream_that_fails_midway_is_not_restarted() -> None:
    inner = Flaky(stream_fail_after=2)
    llm, _, _ = wrap(inner)
    with pytest.raises(RagError):
        run(collect(llm))
    assert inner.calls == 1


def test_a_stalled_stream_times_out_between_deltas() -> None:
    class Stalls(Flaky):
        async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
            yield Delta("first ")
            await asyncio.sleep(5.0)
            yield Delta("never")

    llm, _, _ = wrap(Stalls(), timeout_s=0.05)
    with pytest.raises(RagError) as info:
        run(collect(llm))
    assert info.value.code is ErrorCode.E_PROVIDER_TIMEOUT


@pytest.mark.parametrize(
    "change",
    [{"timeout_s": 0.0}, {"max_retries": -1}, {"breaker_failures": 0}, {"concurrency": 0}],
)
def test_an_invalid_policy_is_refused(change: dict[str, float]) -> None:
    with pytest.raises(RagError):
        ResiliencePolicy(**change)  # type: ignore[arg-type]


def test_a_rejection_that_is_not_retryable_is_neither_retried_nor_counted() -> None:
    rejected = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "no such model", retryable=False)
    inner = Flaky([rejected] * 10)
    llm, pause, _ = wrap(inner, breaker_failures=2)
    for _ in range(4):
        with pytest.raises(RagError) as info:
            run(llm.generate(REQUEST))
        assert info.value is rejected
    assert inner.calls == 4 and pause.waits == []  # no retries, and the breaker never opened
