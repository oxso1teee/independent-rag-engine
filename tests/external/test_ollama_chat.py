"""A real local model through ``OllamaChat`` (opt-in, behind the memory gate).

Set ``RAG_TEST_OLLAMA_CHAT_MODEL`` (for example ``qwen3.5:4b``) to run. The test is skipped, not
failed, when the machine cannot hold the model: the gate needs the model's size plus 1 GiB of
available memory and some free swap, so a small shared machine never gets pushed into an
out-of-memory kill. Only the loopback address is allowed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.llm import OllamaChat
from rag_engine.ports import FinishReason, GenerationParams, GenerationRequest
from tests.factories import make_ids

pytestmark = [pytest.mark.external, pytest.mark.allow_hosts(["127.0.0.1"])]

GIB = 1024**3
BASE_URL = "http://127.0.0.1:11434"


def available(field: str) -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    return 0


@pytest.fixture(scope="module")
def chat() -> OllamaChat:
    model = os.environ.get("RAG_TEST_OLLAMA_CHAT_MODEL")
    if not model:
        pytest.skip("RAG_TEST_OLLAMA_CHAT_MODEL is not set")
    try:
        tags = httpx.get(f"{BASE_URL}/api/tags", timeout=5.0, trust_env=False).json()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running")
    sizes = {m["name"]: int(m["size"]) for m in tags.get("models", [])}
    if model not in sizes:
        pytest.skip(f"{model} is not installed")
    if available("MemAvailable") < sizes[model] + GIB or available("SwapFree") < GIB // 2:
        pytest.skip("not enough free memory or swap for this model")
    return OllamaChat(
        model,
        context_window=2048,
        max_new_tokens=64,
        counter=HeuristicTokenCounter(),
        base_url=BASE_URL,
        timeout_s=300.0,
        keep_alive="0",
    )


REQUEST = GenerationRequest(
    make_ids().request_id(),
    "Reply with one short sentence.",
    "Say that the sky is blue.",
    GenerationParams(temperature=0.0, max_new_tokens=32, num_ctx=2048),
)


def test_a_real_model_answers_and_reports_usage(chat: OllamaChat) -> None:
    result = asyncio.run(chat.generate(REQUEST))
    assert result.text.strip() and result.finish_reason in set(FinishReason)
    assert result.usage.prompt_tokens > 0 and result.usage.completion_tokens > 0


def test_a_real_stream_ends_complete(chat: OllamaChat) -> None:
    async def collect() -> str:
        return "".join([d.text async for d in chat.stream(REQUEST)])

    assert asyncio.run(collect()).strip()
