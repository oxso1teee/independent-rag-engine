"""Ollama and OpenAI-compatible chat adapters against a mocked server (no sockets)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ErrorCode, RagError
from rag_engine.llm import OllamaChat, OpenAICompatChat
from rag_engine.ports import (
    FinishReason,
    GenerationParams,
    GenerationRequest,
    LLMProvider,
    ModelLimits,
)
from tests.factories import make_ids

pytestmark = pytest.mark.contract

REQUEST = GenerationRequest(
    make_ids().request_id(),
    "Answer from evidence.",
    "Evidence:\n[E1] a.md\nBackups run nightly.\n\nQuestion: When?",
    GenerationParams(temperature=0.0, max_new_tokens=64, num_ctx=2048),
)
COUNTER = HeuristicTokenCounter()
Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)


def ndjson(*objects: dict[str, object]) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objects)


def sse(*events: object) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events if e != "[DONE]"]
    if "[DONE]" in events:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def ollama(handler: Handler) -> OllamaChat:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.invalid"
    )
    return OllamaChat(
        "qwen-test", context_window=4096, max_new_tokens=256, counter=COUNTER, client=client
    )


def compat(handler: Handler, *, api_key: str | None = None) -> OpenAICompatChat:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://chat.invalid/v1"
    )
    return OpenAICompatChat(
        "m",
        base_url="http://chat.invalid/v1",
        context_window=4096,
        max_new_tokens=256,
        counter=COUNTER,
        api_key=api_key,
        client=client,
    )


def stream_text(llm: LLMProvider) -> str:
    async def run() -> str:
        return "".join([d.text async for d in llm.stream(REQUEST)])

    return asyncio.run(run())


# ----------------------------------------------------------------------------- Ollama

OLLAMA_DONE = {
    "model": "qwen-test",
    "message": {"role": "assistant", "content": "Backups run nightly [E1]."},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 51,
    "eval_count": 9,
    "total_duration": 2_500_000_000,
}


def test_ollama_satisfies_the_protocol_and_reports_its_limits() -> None:
    llm = ollama(lambda r: httpx.Response(200, json=OLLAMA_DONE))
    assert isinstance(llm, LLMProvider)
    assert llm.limits() == ModelLimits(4096, 256) and llm.count_tokens("") == 0


def test_ollama_request_is_native_chat_with_explicit_window_and_no_thinking() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OLLAMA_DONE)

    asyncio.run(ollama(handler).generate(REQUEST))
    body = seen[0]
    assert body["stream"] is False and body["think"] is False and body["model"] == "qwen-test"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    options = body["options"]
    assert options["num_ctx"] == 2048 and options["num_predict"] == 64
    assert options["temperature"] == 0.0 and options["seed"] == 0


def test_ollama_generate_maps_text_usage_finish_reason_and_latency() -> None:
    result = asyncio.run(ollama(lambda r: httpx.Response(200, json=OLLAMA_DONE)).generate(REQUEST))
    assert result.text == "Backups run nightly [E1]." and result.finish_reason is FinishReason.STOP
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (51, 9)
    assert result.latency_ms == 2500


def test_ollama_length_stop_is_reported() -> None:
    cut = {**OLLAMA_DONE, "done_reason": "length"}
    result = asyncio.run(ollama(lambda r: httpx.Response(200, json=cut)).generate(REQUEST))
    assert result.finish_reason is FinishReason.LENGTH


def test_ollama_a_missing_prompt_count_falls_back_to_the_estimate() -> None:
    cached = {k: v for k, v in OLLAMA_DONE.items() if k != "prompt_eval_count"}
    result = asyncio.run(ollama(lambda r: httpx.Response(200, json=cached)).generate(REQUEST))
    assert result.usage.prompt_tokens >= COUNTER.count(REQUEST.prompt)


def test_ollama_never_returns_the_thinking_field() -> None:
    body = {**OLLAMA_DONE, "message": {"role": "assistant", "content": "ok", "thinking": "secret"}}
    result = asyncio.run(ollama(lambda r: httpx.Response(200, json=body)).generate(REQUEST))
    assert result.text == "ok"


def test_ollama_stream_equals_generate() -> None:
    pieces = ["Backups ", "run ", "nightly ", "[E1]."]
    lines = [{"message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
    lines.append({"message": {"role": "assistant", "content": ""}, "done": True})
    llm = ollama(lambda r: httpx.Response(200, content=ndjson(*lines)))
    assert stream_text(llm) == "".join(pieces)


def test_ollama_a_stream_whose_prompt_was_cut_fails_after_the_closing_event() -> None:
    lines = [
        {"message": {"content": "Backups "}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 2048},  # == num_ctx
    ]
    llm = ollama(lambda r: httpx.Response(200, content=ndjson(*lines)))
    with pytest.raises(RagError) as info:
        stream_text(llm)
    assert info.value.code is ErrorCode.E_CONTEXT_TRUNCATED


def test_ollama_a_stream_cut_short_is_an_error_not_a_complete_answer() -> None:
    lines = [{"message": {"role": "assistant", "content": "Backups "}, "done": False}]
    llm = ollama(lambda r: httpx.Response(200, content=ndjson(*lines)))
    with pytest.raises(RagError) as info:
        stream_text(llm)
    assert info.value.code is ErrorCode.E_PROVIDER_UNAVAILABLE and not info.value.retryable


def test_ollama_an_error_inside_the_stream_is_reported_not_retryable() -> None:
    content = ndjson({"message": {"content": "x"}, "done": False}, {"error": "boom"})
    with pytest.raises(RagError) as info:
        stream_text(ollama(lambda r: httpx.Response(200, content=content)))
    assert not info.value.retryable


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (500, ErrorCode.E_PROVIDER_UNAVAILABLE, True),
        (503, ErrorCode.E_PROVIDER_UNAVAILABLE, True),
        (429, ErrorCode.E_RATE_LIMITED, True),
        (404, ErrorCode.E_PROVIDER_UNAVAILABLE, False),
        (400, ErrorCode.E_PROVIDER_UNAVAILABLE, False),
        (401, ErrorCode.E_CONFIG_INVALID, False),
    ],
)
def test_ollama_http_failures_are_classified(status: int, code: ErrorCode, retryable: bool) -> None:
    llm = ollama(lambda r: httpx.Response(status, text="model 'x' not found"))
    with pytest.raises(RagError) as info:
        asyncio.run(llm.generate(REQUEST))
    assert info.value.code is code and info.value.retryable is retryable
    assert info.value.context["status"] == status


def test_ollama_an_out_of_memory_failure_is_a_resource_error_and_never_retried() -> None:
    llm = ollama(lambda r: httpx.Response(500, text="model requires more system memory"))
    with pytest.raises(RagError) as info:
        asyncio.run(llm.generate(REQUEST))
    assert info.value.code is ErrorCode.E_RESOURCE_GATE and not info.value.retryable


def test_ollama_connection_and_timeout_errors_are_mapped() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(RagError) as down:
        asyncio.run(ollama(refuse).generate(REQUEST))
    with pytest.raises(RagError) as late:
        asyncio.run(ollama(slow).generate(REQUEST))
    assert down.value.code is ErrorCode.E_PROVIDER_UNAVAILABLE and down.value.retryable
    assert late.value.code is ErrorCode.E_PROVIDER_TIMEOUT


@pytest.mark.parametrize("body", ["not json", "[1, 2]", '{"done": true}'])
def test_ollama_malformed_responses_are_refused(body: str) -> None:
    with pytest.raises(RagError) as info:
        asyncio.run(ollama(lambda r: httpx.Response(200, text=body)).generate(REQUEST))
    assert info.value.code is ErrorCode.E_PROVIDER_UNAVAILABLE and not info.value.retryable


def test_ollama_a_cancelled_generation_stays_cancelled() -> None:
    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json=OLLAMA_DONE)

    async def scenario() -> None:
        task = asyncio.create_task(ollama(hang).generate(REQUEST))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_ollama_construction_is_validated() -> None:
    with pytest.raises(RagError):
        OllamaChat("", context_window=4096, max_new_tokens=256, counter=COUNTER)
    with pytest.raises(RagError):
        OllamaChat("m", context_window=4096, max_new_tokens=256, counter=COUNTER, timeout_s=0)
    with pytest.raises(RagError):  # a window no larger than the answer allowance
        OllamaChat("m", context_window=256, max_new_tokens=256, counter=COUNTER)


# ----------------------------------------------------------------------- OpenAI-compatible

COMPAT_OK = {
    "choices": [
        {"message": {"role": "assistant", "content": "Nightly [E1]."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 40, "completion_tokens": 5},
}


def test_compat_satisfies_the_protocol() -> None:
    llm = compat(lambda r: httpx.Response(200, json=COMPAT_OK))
    assert isinstance(llm, LLMProvider) and llm.limits() == ModelLimits(4096, 256)


def test_compat_generate_maps_text_usage_and_finish_reason() -> None:
    result = asyncio.run(compat(lambda r: httpx.Response(200, json=COMPAT_OK)).generate(REQUEST))
    assert result.text == "Nightly [E1]." and result.finish_reason is FinishReason.STOP
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (40, 5)


def test_compat_request_shape_and_bearer_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPAT_OK)

    asyncio.run(compat(handler, api_key="sk-test").generate(REQUEST))
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert body["max_tokens"] == 64 and body["temperature"] == 0.0 and body["stream"] is False
    assert "sk-test" not in seen[0].content.decode()  # the key is only a header, never payload


def test_compat_length_finish_is_reported() -> None:
    cut = {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}
    result = asyncio.run(compat(lambda r: httpx.Response(200, json=cut)).generate(REQUEST))
    assert result.finish_reason is FinishReason.LENGTH and result.usage.prompt_tokens > 0


def test_compat_stream_equals_generate() -> None:
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Nightly "}}]},
        {"choices": [{"delta": {"content": "[E1]."}}]},
        {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 5}},
        "[DONE]",
    ]
    llm = compat(lambda r: httpx.Response(200, content=sse(*chunks)))
    assert stream_text(llm) == "Nightly [E1]."


def test_compat_a_stream_without_done_is_an_error() -> None:
    chunks = [{"choices": [{"delta": {"content": "Nightly "}}]}]
    with pytest.raises(RagError):
        stream_text(compat(lambda r: httpx.Response(200, content=sse(*chunks))))


def test_compat_an_error_event_in_the_stream_is_reported() -> None:
    with pytest.raises(RagError) as info:
        stream_text(compat(lambda r: httpx.Response(200, content=sse({"error": "overloaded"}))))
    assert not info.value.retryable


def test_compat_failures_are_classified_and_the_key_never_leaks() -> None:
    llm = compat(lambda r: httpx.Response(401, text="bad key sk-secret"), api_key="sk-secret")
    with pytest.raises(RagError) as info:
        asyncio.run(llm.generate(REQUEST))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    assert "sk-secret" not in str(info.value.to_dict())  # a server that echoes the key
    rate = compat(lambda r: httpx.Response(429, text="slow"))
    with pytest.raises(RagError) as limited:
        asyncio.run(rate.generate(REQUEST))
    assert limited.value.code is ErrorCode.E_RATE_LIMITED and limited.value.retryable


@pytest.mark.parametrize("body", [{"choices": []}, {"choices": [{"message": {}}]}, {}])
def test_compat_malformed_responses_are_refused(body: dict[str, object]) -> None:
    with pytest.raises(RagError):
        asyncio.run(compat(lambda r: httpx.Response(200, json=body)).generate(REQUEST))


def test_compat_requires_a_window_and_a_model() -> None:
    with pytest.raises(TypeError):
        OpenAICompatChat("m", base_url="http://x", max_new_tokens=8, counter=COUNTER)  # type: ignore[call-arg]
    with pytest.raises(RagError):
        OpenAICompatChat(
            "", base_url="http://x", context_window=4096, max_new_tokens=8, counter=COUNTER
        )
