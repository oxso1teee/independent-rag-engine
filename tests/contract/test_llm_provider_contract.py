"""Contract every ``LLMProvider`` implementation must satisfy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.generation.resilience import ResilientLLM
from rag_engine.ports import (
    FinishReason,
    GenerationParams,
    GenerationRequest,
    LLMProvider,
    ModelLimits,
)
from rag_engine.testing import Fail, Reply, ScriptedLLM, ScriptExhaustedError
from tests.factories import make_ids

REQUEST = GenerationRequest(
    request_id=make_ids().request_id(),
    system_prompt="Answer from evidence only.",
    prompt="Question: what is retrieval?\nEvidence: [E1] retrieval finds passages.",
    params=GenerationParams(max_new_tokens=64),
)
ANSWER = "Retrieval finds relevant passages [E1].\n\nSecond   paragraph with  spacing."


def scripted(*steps: Reply | Fail) -> ScriptedLLM:
    return ScriptedLLM(list(steps))


FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "scripted": lambda: scripted(Reply(ANSWER), Reply(ANSWER)),
    "resilient-scripted": lambda: ResilientLLM(scripted(Reply(ANSWER), Reply(ANSWER))),
}


@pytest.fixture(params=sorted(FACTORIES))
def provider(request: pytest.FixtureRequest) -> LLMProvider:
    return FACTORIES[request.param]()


async def collect(provider: LLMProvider) -> str:
    return "".join([delta.text async for delta in provider.stream(REQUEST)])


def test_satisfies_the_protocol(provider: LLMProvider) -> None:
    assert isinstance(provider, LLMProvider)
    limits = provider.limits()
    assert 0 < limits.max_new_tokens < limits.context_window
    assert provider.count_tokens("") == 0
    assert provider.count_tokens("one two three") == 3


def test_stream_equals_generate(provider: LLMProvider) -> None:
    generated = asyncio.run(provider.generate(REQUEST))
    streamed = asyncio.run(collect(provider))
    assert streamed == generated.text == ANSWER


def test_generate_reports_usage_and_finish_reason() -> None:
    llm = scripted(Reply(ANSWER, latency_ms=7))
    result = asyncio.run(llm.generate(REQUEST))
    assert result.finish_reason is FinishReason.STOP
    assert result.usage.completion_tokens == llm.count_tokens(ANSWER)
    assert result.usage.prompt_tokens == llm.count_tokens(REQUEST.system_prompt) + llm.count_tokens(
        REQUEST.prompt
    )
    assert result.latency_ms == 7


def test_scripted_edge_cases_are_reproducible() -> None:
    llm = scripted(
        Reply(""),
        Reply("x"),
        Reply("cut off", finish_reason=FinishReason.LENGTH),
        Reply("answer", prompt_tokens=8192),
    )
    empty, one_char, truncated, saturated = (asyncio.run(llm.generate(REQUEST)) for _ in range(4))
    assert empty.text == ""
    assert empty.usage.completion_tokens == 0
    assert one_char.text == "x"
    assert truncated.finish_reason is FinishReason.LENGTH
    assert saturated.usage.prompt_tokens == 8192


def test_failures_are_raised_as_scripted_errors() -> None:
    error = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")
    llm = scripted(Fail(error), Fail(error))
    with pytest.raises(RagError) as info:
        asyncio.run(llm.generate(REQUEST))
    assert info.value is error
    with pytest.raises(RagError):
        asyncio.run(collect(llm))


def test_calls_are_recorded_in_order() -> None:
    llm = scripted(Reply("a"), Reply("b"))
    asyncio.run(llm.generate(REQUEST))
    asyncio.run(collect(llm))
    assert llm.calls == (REQUEST, REQUEST)


def test_exhausted_script_is_a_loud_failure() -> None:
    llm = scripted()
    with pytest.raises(ScriptExhaustedError):
        asyncio.run(llm.generate(REQUEST))


def test_custom_limits_are_reported() -> None:
    limits = ModelLimits(context_window=4096, max_new_tokens=256)
    assert ScriptedLLM([], limits=limits).limits() == limits


@pytest.mark.parametrize("text", ["", "  ", "word", "two words", " lead and trail ", "a\n\nb\n"])
def test_stream_reassembles_any_text(text: str) -> None:
    llm = scripted(Reply(text))
    assert asyncio.run(collect(llm)) == text
