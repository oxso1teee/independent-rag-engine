"""The container builds the chat provider; the API key comes only from the environment."""

from __future__ import annotations

import asyncio

import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.config.settings import LlmSettings, Settings
from rag_engine.container import build_chat
from rag_engine.domain import ErrorCode, RagError
from rag_engine.llm import OllamaChat, OpenAICompatChat
from rag_engine.ports import LLMProvider, ModelLimits

pytestmark = pytest.mark.unit

COUNTER = HeuristicTokenCounter()


def settings(**changes: object) -> LlmSettings:
    base: dict[str, object] = {
        "enabled": True,
        "provider": "ollama",
        "model": "qwen3.5:4b",
        "context_window": 4096,
        "max_new_tokens": 256,
    }
    base.update(changes)
    return LlmSettings(**base)


def test_ollama_is_built_with_the_configured_limits_behind_the_wrapper() -> None:
    built = build_chat(settings(), COUNTER)
    assert isinstance(built.inner, OllamaChat) and isinstance(built.provider, LLMProvider)
    assert built.provider.limits() == ModelLimits(4096, 256)
    asyncio.run(built.aclose())


def test_an_openai_compatible_server_needs_an_address() -> None:
    with pytest.raises(ValueError, match="base_url"):
        settings(provider="openai_compat")


def test_the_api_key_is_read_from_the_named_environment_variable() -> None:
    config = settings(
        provider="openai_compat", base_url="http://chat.invalid/v1", api_key_env="CHAT_KEY"
    )
    built = build_chat(config, COUNTER, environ={"CHAT_KEY": "sk-test"})
    assert isinstance(built.inner, OpenAICompatChat)
    asyncio.run(built.aclose())
    with pytest.raises(RagError) as info:
        build_chat(config, COUNTER, environ={})
    assert info.value.code is ErrorCode.E_CONFIG_INVALID and "CHAT_KEY" in info.value.message


def test_a_disabled_or_unknown_provider_is_refused() -> None:
    with pytest.raises(RagError):
        build_chat(LlmSettings(), COUNTER)
    with pytest.raises(RagError):
        build_chat(settings(provider="mystery"), COUNTER)


@pytest.mark.parametrize("value", ["0", "-1", "30s", "5m", "2h"])
def test_keep_alive_accepts_ollama_durations(value: str) -> None:
    assert settings(keep_alive=value).keep_alive == value


def test_a_bad_keep_alive_is_refused() -> None:
    with pytest.raises(ValueError, match="keep_alive"):
        settings(keep_alive="forever")


def test_a_model_stays_loaded_only_when_the_budget_allows_it() -> None:
    assert settings().keep_alive == "0"  # the safe default on the target machine
    chat = {"enabled": True, "provider": "ollama", "model": "m", "context_window": 4096}
    ranker = {
        "enabled": True,
        "model": "r",
        "model_dir": "/models/r",
        "revision": "a" * 64,
    }
    Settings(llm={**chat, "keep_alive": "5m"})  # no reranker: a resident model is allowed
    with pytest.raises(ValueError, match="keep_alive"):
        Settings(rerank=ranker, llm={**chat, "keep_alive": "5m"})
    Settings(rerank=ranker, llm=chat)  # unloads at once: allowed
