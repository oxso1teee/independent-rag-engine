"""LLM provider port."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from rag_engine.domain import RequestId, require


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_window: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        require(self.context_window > 0, "context_window must be positive")
        require(0 < self.max_new_tokens < self.context_window, "invalid max_new_tokens")


@dataclass(frozen=True, slots=True)
class GenerationParams:
    temperature: float = 0.0
    max_new_tokens: int = 512
    num_ctx: int | None = None

    def __post_init__(self) -> None:
        require(self.temperature >= 0.0, "temperature must be non-negative")
        require(self.max_new_tokens > 0, "max_new_tokens must be positive")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: RequestId
    system_prompt: str
    prompt: str
    params: GenerationParams = GenerationParams()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    finish_reason: FinishReason
    usage: TokenUsage
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class Delta:
    text: str


@runtime_checkable
class LLMProvider(Protocol):
    def limits(self) -> ModelLimits: ...

    def count_tokens(self, text: str) -> int: ...

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        """Concatenated deltas equal ``generate(request).text`` for the same input."""
        ...
