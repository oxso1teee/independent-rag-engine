"""An OpenAI-compatible ``/chat/completions`` server as an ``LLMProvider``.

For servers that speak this protocol (llama.cpp, vLLM, LM Studio, hosted gateways). The server does
not state its context window, so ``context_window`` is mandatory configuration. The API key, when
there is one, goes only into the ``Authorization`` header: it is never part of an error, a log
field or the request payload.
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

_PROVIDER = "The chat server"


class OpenAICompatChat:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        context_window: int,
        max_new_tokens: int,
        counter: TokenCounter,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip() or timeout_s <= 0:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID, "the chat server needs a model and a timeout"
            )
        self._model = model
        self._limits = ModelLimits(context_window, max_new_tokens)
        self._counter = counter
        self._secret = api_key or None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
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

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.params.temperature,
            "max_tokens": min(request.params.max_new_tokens, self._limits.max_new_tokens),
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = await self._client.post(
                "/chat/completions", json=self._payload(request, stream=False)
            )
        except httpx.HTTPError as exc:
            raise transport_error(_PROVIDER, exc) from exc
        if response.status_code >= 400:
            raise status_error(_PROVIDER, response.status_code, response.text, self._secret)
        try:
            data = response.json()
        except ValueError as exc:
            raise invalid_response(_PROVIDER, "not JSON") from exc
        choice = _first_choice(data)
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise invalid_response(_PROVIDER, "no message content")
        return GenerationResult(
            text=text,
            finish_reason=_finish(choice),
            usage=self._usage(data, request, text),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=self._payload(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise status_error(_PROVIDER, response.status_code, body, self._secret)
                finished = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        finished = True
                        break
                    try:
                        data = json.loads(payload)
                    except ValueError as exc:
                        raise invalid_response(_PROVIDER, "a stream event is not JSON") from exc
                    if isinstance(data, dict) and "error" in data:
                        raise stream_error(_PROVIDER, str(data["error"]), self._secret)
                    for choice in data.get("choices", []) if isinstance(data, dict) else []:
                        delta = choice.get("delta") if isinstance(choice, dict) else None
                        piece = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(piece, str) and piece:
                            yield Delta(piece)
                if not finished:
                    raise stream_error(_PROVIDER, "the stream ended before it was complete")
        except httpx.HTTPError as exc:
            raise transport_error(_PROVIDER, exc) from exc

    def _usage(self, data: object, request: GenerationRequest, text: str) -> TokenUsage:
        usage = data.get("usage") if isinstance(data, dict) else None
        prompt = _count(usage.get("prompt_tokens")) if isinstance(usage, dict) else 0
        completion = _count(usage.get("completion_tokens")) if isinstance(usage, dict) else 0
        if not prompt:
            prompt = self.count_tokens(request.system_prompt) + self.count_tokens(request.prompt)
        if not completion and text:
            completion = self.count_tokens(text)
        return TokenUsage(prompt, completion)


def _first_choice(data: object) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise invalid_response(_PROVIDER, "no choices")
    return choices[0]


def _finish(choice: dict[str, Any]) -> FinishReason:
    return FinishReason.LENGTH if choice.get("finish_reason") == "length" else FinishReason.STOP


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
