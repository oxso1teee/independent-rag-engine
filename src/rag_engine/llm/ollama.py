"""Ollama's native ``/api/chat`` as an ``LLMProvider``.

The native API is used rather than the OpenAI-compatible one because it takes ``num_ctx`` and
``think`` explicitly and reports ``prompt_eval_count`` and ``done_reason``, which truncation and
token accounting depend on. Reasoning is switched off by default (``think: false``): a reasoning
model would otherwise spend the answer budget on text that is never shown, and any ``thinking``
field that still arrives is dropped, never returned as part of the answer.

Ollama has no tokenizer endpoint, so ``count_tokens`` uses the injected counter, a deliberately
high estimate; the true prompt size is whatever the server reports.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_engine.domain import ErrorCode, RagError
from rag_engine.llm._http import invalid_response, status_error, stream_error, transport_error
from rag_engine.ports import (
    Delta,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelLimits,
    TokenCounter,
    TokenUsage,
)

_PROVIDER = "Ollama"


class OllamaChat:
    def __init__(
        self,
        model: str,
        *,
        context_window: int,
        max_new_tokens: int,
        counter: TokenCounter,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 180.0,
        keep_alive: str = "0",
        think: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip() or timeout_s <= 0:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "Ollama chat needs a model and a timeout")
        self._model = model
        self._limits = ModelLimits(context_window, max_new_tokens)  # validates the pair
        self._counter = counter
        self._keep_alive = keep_alive
        self._think = think
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 5.0)),
            trust_env=False,
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def limits(self) -> ModelLimits:
        return self._limits

    def count_tokens(self, text: str) -> int:
        return self._counter.count(text)

    # ------------------------------------------------------------------ requests

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        params = request.params
        window = params.num_ctx or self._limits.context_window
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "stream": stream,
            "think": self._think,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": params.temperature,
                "num_predict": min(params.max_new_tokens, self._limits.max_new_tokens),
                "num_ctx": window,
                "seed": 0,
            },
        }

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = await self._client.post(
                "/api/chat", json=self._payload(request, stream=False)
            )
        except httpx.HTTPError as exc:
            raise transport_error(_PROVIDER, exc) from exc
        if response.status_code >= 400:
            raise status_error(_PROVIDER, response.status_code, response.text)
        data = _object(response.text)
        if isinstance(data.get("error"), str):
            raise stream_error(_PROVIDER, data["error"])
        text = _content(data)
        return GenerationResult(
            text=text,
            finish_reason=_finish(data),
            usage=self._usage(data, request, text),
            latency_ms=_nanoseconds_to_ms(data.get("total_duration")),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=self._payload(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise status_error(_PROVIDER, response.status_code, body)
                finished = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = _object(line)
                    if isinstance(data.get("error"), str):
                        raise stream_error(_PROVIDER, data["error"])
                    piece = _content(data, allow_missing=True)
                    if piece:
                        yield Delta(piece)
                    if data.get("done") is True:
                        self._refuse_if_prompt_was_cut(data, request)
                        finished = True
                        break
                if not finished:
                    raise stream_error(_PROVIDER, "the stream ended before it was complete")
        except httpx.HTTPError as exc:
            raise transport_error(_PROVIDER, exc) from exc

    def _refuse_if_prompt_was_cut(self, data: dict[str, Any], request: GenerationRequest) -> None:
        """Ollama silently drops the start of a prompt longer than ``num_ctx``; a stream reports no
        usage to the caller, so the check on the closing event happens here."""
        window = request.params.num_ctx or self._limits.context_window
        if _count(data.get("prompt_eval_count")) >= window - 1:
            raise RagError(
                ErrorCode.E_CONTEXT_TRUNCATED,
                "the provider reports a prompt as long as its window: it was probably cut",
                context={"prompt_tokens": _count(data.get("prompt_eval_count")), "window": window},
            )

    # ---------------------------------------------------------------- accounting

    def _usage(self, data: dict[str, Any], request: GenerationRequest, text: str) -> TokenUsage:
        """Reported counts; a missing or zero prompt count (a cached prompt) is estimated."""
        prompt = _count(data.get("prompt_eval_count"))
        completion = _count(data.get("eval_count"))
        if not prompt:
            prompt = self.count_tokens(request.system_prompt) + self.count_tokens(request.prompt)
        if not completion and text:
            completion = self.count_tokens(text)
        return TokenUsage(prompt, completion)


def _object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise invalid_response(_PROVIDER, "not JSON") from exc
    if not isinstance(data, dict):
        raise invalid_response(_PROVIDER, "not an object")
    return data


def _content(data: dict[str, Any], *, allow_missing: bool = False) -> str:
    message = data.get("message")
    if not isinstance(message, dict):
        if allow_missing and data.get("done") is True:
            return ""
        raise invalid_response(_PROVIDER, "no message")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise invalid_response(_PROVIDER, "message content is not text")
    return content  # a ``thinking`` field, if any, is intentionally never read


def _finish(data: dict[str, Any]) -> FinishReason:
    return FinishReason.LENGTH if data.get("done_reason") == "length" else FinishReason.STOP


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _nanoseconds_to_ms(value: object) -> int:
    return value // 1_000_000 if isinstance(value, int) and not isinstance(value, bool) else 0
