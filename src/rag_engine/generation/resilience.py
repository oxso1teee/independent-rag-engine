"""Timeouts, bounded retries, a circuit breaker and serialised access around any ``LLMProvider``.

Only failures that say "try again" are retried: an error that is flagged ``retryable`` and is
either "the server is unreachable or answers 5xx" (``E_PROVIDER_UNAVAILABLE``) or "slow down"
(``E_RATE_LIMITED``). A 4xx rejection (unknown model, bad request) is neither retried nor counted
against the breaker. A timeout is not retried,
because the deadline already spent the caller's patience and a second try would double it. After
``breaker_failures`` consecutive failed calls the breaker opens and calls fail at once for
``breaker_reset_s`` seconds; the first call after that is a trial that closes or reopens it. A
cancelled call propagates its cancellation, frees its slot and counts as neither success nor
failure. A local model serves one request at a time on this hardware, so calls are serialised
unless the policy allows more.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from rag_engine.domain import ErrorCode, RagError, require
from rag_engine.observability import get_logger
from rag_engine.ports import (
    Delta,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ModelLimits,
)

_log = get_logger("rag_engine.generation.resilience")
_RETRYABLE = frozenset({ErrorCode.E_PROVIDER_UNAVAILABLE, ErrorCode.E_RATE_LIMITED})
_BREAKING = frozenset({ErrorCode.E_PROVIDER_UNAVAILABLE, ErrorCode.E_PROVIDER_TIMEOUT})


def _transient(error: RagError) -> bool:
    return error.retryable and error.code in _RETRYABLE


@dataclass(frozen=True, slots=True)
class ResiliencePolicy:
    timeout_s: float = 120.0
    max_retries: int = 2
    backoff_s: float = 0.5
    backoff_factor: float = 2.0
    breaker_failures: int = 3
    breaker_reset_s: float = 60.0
    concurrency: int = 1

    def __post_init__(self) -> None:
        require(
            self.timeout_s > 0
            and self.max_retries >= 0
            and self.backoff_s >= 0
            and self.backoff_factor >= 1.0
            and self.breaker_failures >= 1
            and self.breaker_reset_s > 0
            and self.concurrency >= 1,
            "invalid resilience policy",
        )


class ResilientLLM:
    def __init__(
        self,
        inner: LLMProvider,
        policy: ResiliencePolicy | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._policy = policy or ResiliencePolicy()
        self._sleep = sleep
        self._monotonic = monotonic
        self._slots = asyncio.Semaphore(self._policy.concurrency)
        self._failures = 0
        self._opened_at: float | None = None

    def limits(self) -> ModelLimits:
        return self._inner.limits()

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    # ------------------------------------------------------------------ breaker

    def _admit(self) -> None:
        if self._opened_at is None:
            return
        remaining = self._opened_at + self._policy.breaker_reset_s - self._monotonic()
        if remaining > 0:
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "the model provider is failing: circuit open, not calling it",
                context={"retry_after_s": round(remaining, 1)},
            )
        # Half-open: this call is the trial. A failure reopens at once, a success closes.
        self._failures = self._policy.breaker_failures - 1

    def _succeeded(self) -> None:
        self._failures = 0
        self._opened_at = None

    def _failed(self, error: RagError) -> None:
        if error.code not in _BREAKING or not error.retryable:
            return
        self._failures += 1
        if self._failures >= self._policy.breaker_failures:
            self._opened_at = self._monotonic()
            _log.warning("llm.circuit_open", failures=self._failures)

    # --------------------------------------------------------------------- calls

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        async with self._slots:
            self._admit()
            delay = self._policy.backoff_s
            for attempt in range(self._policy.max_retries + 1):
                try:
                    async with asyncio.timeout(self._policy.timeout_s):
                        result = await self._inner.generate(request)
                except TimeoutError:
                    error = self._timeout_error()
                    self._failed(error)
                    raise error from None
                except RagError as error:
                    if _transient(error) and attempt < self._policy.max_retries:
                        _log.warning("llm.retry", attempt=attempt + 1, code=error.code.value)
                        await self._sleep(delay)
                        delay *= self._policy.backoff_factor
                        continue
                    self._failed(error)
                    raise
                self._succeeded()
                return result
        raise AssertionError("unreachable")  # pragma: no cover

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        async with self._slots:
            self._admit()
            delay = self._policy.backoff_s
            attempt = 0
            while True:
                started = False
                iterator = self._inner.stream(request).__aiter__()
                try:
                    while True:
                        try:
                            async with asyncio.timeout(self._policy.timeout_s):
                                delta = await iterator.__anext__()
                        except StopAsyncIteration:
                            self._succeeded()
                            return
                        except TimeoutError:
                            error = self._timeout_error()
                            self._failed(error)
                            raise error from None
                        started = True
                        yield delta
                except RagError as error:
                    retry = _transient(error) and attempt < self._policy.max_retries
                    if started or not retry:
                        self._failed(error)
                        raise
                    _log.warning("llm.retry", attempt=attempt + 1, code=error.code.value)
                    await self._sleep(delay)
                    delay *= self._policy.backoff_factor
                    attempt += 1
                finally:
                    aclose = getattr(iterator, "aclose", None)
                    if aclose is not None:
                        await aclose()

    def _timeout_error(self) -> RagError:
        return RagError(
            ErrorCode.E_PROVIDER_TIMEOUT,
            "the model provider did not answer in time",
            context={"timeout_s": self._policy.timeout_s},
        )
