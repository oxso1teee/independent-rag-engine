"""Ollama native API contract on a deterministic in-process HTTP transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, ErrorCode, RagError
from rag_engine.embedding.ollama import OllamaEmbedder
from rag_engine.ports import EmbeddingProvider
from tests.factories import T0

DIGEST = "a" * 64


def model() -> EmbeddingModel:
    model_id = EmbeddingModelId.build(
        provider="ollama",
        name="test:latest",
        revision=DIGEST,
        dimension=4,
        normalized=True,
        query_prefix="",
        document_prefix="",
    )
    return EmbeddingModel(
        id=model_id,
        provider="ollama",
        name="test:latest",
        revision=DIGEST,
        dimension=4,
        normalized=True,
        max_input_tokens=128,
        created_at=T0,
    )


def test_native_embed_request_and_digest_check() -> None:
    seen: list[tuple[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            seen.append(("tags", None))
            return httpx.Response(200, json={"models": [{"name": "test:latest", "digest": DIGEST}]})
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        seen.append(("embed", payload))
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0, 0.0]]})

    async def run() -> np.ndarray:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            provider = OllamaEmbedder(model(), client=client)
            assert isinstance(provider, EmbeddingProvider)
            return await provider.embed_query("hello")

    vector = asyncio.run(run())
    assert vector.shape == (1, 4) and vector.dtype == np.float32
    assert seen == [
        ("tags", None),
        (
            "embed",
            {"model": "test:latest", "input": ["hello"], "truncate": False, "keep_alive": "0"},
        ),
    ]


@pytest.mark.parametrize(
    ("digest", "expected"),
    [("changed", ErrorCode.E_EMBED_MODEL_MISMATCH), (None, ErrorCode.E_PROVIDER_UNAVAILABLE)],
)
def test_missing_or_changed_model_never_calls_embed(
    digest: str | None, expected: ErrorCode
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        models = [] if digest is None else [{"name": "test:latest", "digest": digest}]
        return httpx.Response(200, json={"models": models})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            await OllamaEmbedder(model(), client=client).embed_documents(["x"])

    with pytest.raises(RagError) as info:
        asyncio.run(run())
    assert info.value.code is expected
    assert calls == ["/api/tags"]


def test_bad_vector_shape_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test:latest", "digest": DIGEST}]})
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            await OllamaEmbedder(model(), client=client).embed_documents(["x"])

    with pytest.raises(RagError) as info:
        asyncio.run(run())
    assert info.value.code is ErrorCode.E_EMBED_DIM_MISMATCH


def test_timeout_has_stable_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            await OllamaEmbedder(model(), client=client).embed_documents(["x"])

    with pytest.raises(RagError) as info:
        asyncio.run(run())
    assert info.value.code is ErrorCode.E_PROVIDER_TIMEOUT and info.value.retryable


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (400, ErrorCode.E_PROVIDER_UNAVAILABLE, False),
        (429, ErrorCode.E_RATE_LIMITED, True),
        (503, ErrorCode.E_PROVIDER_UNAVAILABLE, True),
    ],
)
def test_http_failures_are_classified(status: int, code: ErrorCode, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test:latest", "digest": DIGEST}]})
        return httpx.Response(status, json={"error": "private document text"})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            await OllamaEmbedder(model(), client=client).embed_documents(["x"])

    with pytest.raises(RagError) as info:
        asyncio.run(run())
    assert info.value.code is code and info.value.retryable is retryable
    assert "private document text" not in str(info.value)
