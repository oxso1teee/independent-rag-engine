"""Scripted LLM: replays a fixed list of replies and failures, and records every call."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from rag_engine.domain import RagError
from rag_engine.ports import (
    Delta,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelLimits,
    TokenUsage,
)

_PIECES = re.compile(r"\S+\s*|\s+")


@dataclass(frozen=True, slots=True)
class Reply:
    """A successful generation. ``prompt_tokens`` overrides the computed value
    (useful to simulate a prompt silently truncated to the context window)."""

    text: str
    finish_reason: FinishReason = FinishReason.STOP
    prompt_tokens: int | None = None
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class Fail:
    """A provider failure, raised when the step is consumed."""

    error: RagError


type Step = Reply | Fail


class ScriptExhaustedError(RuntimeError):
    """The test made more LLM calls than it scripted."""


class ScriptedLLM:
    """``LLMProvider`` whose behaviour is fully determined by its script."""

    def __init__(
        self,
        script: Sequence[Step],
        *,
        limits: ModelLimits = ModelLimits(context_window=8192, max_new_tokens=1024),  # noqa: B008
    ) -> None:
        self._script = list(script)
        self._limits = limits
        self._calls: list[GenerationRequest] = []

    @property
    def calls(self) -> tuple[GenerationRequest, ...]:
        return tuple(self._calls)

    def limits(self) -> ModelLimits:
        return self._limits

    def count_tokens(self, text: str) -> int:
        """Whitespace-separated word count: a documented approximation."""
        return len(text.split())

    def _next(self, request: GenerationRequest) -> Reply:
        self._calls.append(request)
        if not self._script:
            raise ScriptExhaustedError(f"no scripted step left for call #{len(self._calls)}")
        step = self._script.pop(0)
        if isinstance(step, Fail):
            raise step.error
        return step

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        step = self._next(request)
        prompt_tokens = (
            step.prompt_tokens
            if step.prompt_tokens is not None
            else self.count_tokens(request.system_prompt) + self.count_tokens(request.prompt)
        )
        return GenerationResult(
            text=step.text,
            finish_reason=step.finish_reason,
            usage=TokenUsage(prompt_tokens, self.count_tokens(step.text)),
            latency_ms=step.latency_ms,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        step = self._next(request)
        for piece in _PIECES.findall(step.text):
            yield Delta(piece)
