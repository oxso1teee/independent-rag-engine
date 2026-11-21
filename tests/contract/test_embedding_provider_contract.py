"""Contract every ``EmbeddingProvider`` implementation must satisfy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
import pytest

from rag_engine.ports import EmbeddingProvider
from rag_engine.testing import HashingEmbedder

FACTORIES: dict[str, Callable[[], EmbeddingProvider]] = {
    "hashing-256": lambda: HashingEmbedder(256),
    "hashing-64": lambda: HashingEmbedder(64),
}


@pytest.fixture(params=sorted(FACTORIES))
def provider(request: pytest.FixtureRequest) -> EmbeddingProvider:
    return FACTORIES[request.param]()


def test_satisfies_the_protocol(provider: EmbeddingProvider) -> None:
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension > 0
    assert provider.max_input_tokens > 0
    assert provider.model_id.dimension == provider.dimension


def test_documents_shape_dtype_and_order(provider: EmbeddingProvider) -> None:
    texts = ["first text", "second text", "third"]
    vectors = asyncio.run(provider.embed_documents(texts))
    assert vectors.shape == (3, provider.dimension)
    assert vectors.dtype == np.float32
    one_by_one = [asyncio.run(provider.embed_documents([t]))[0] for t in texts]
    np.testing.assert_allclose(vectors, np.stack(one_by_one), atol=1e-6)


def test_empty_batch(provider: EmbeddingProvider) -> None:
    vectors = asyncio.run(provider.embed_documents([]))
    assert vectors.shape == (0, provider.dimension)


def test_query_shape(provider: EmbeddingProvider) -> None:
    assert asyncio.run(provider.embed_query("hello")).shape == (1, provider.dimension)


@pytest.mark.parametrize("text", ["", "   ", "обычный текст", "emoji 🎉", "a" * 10_000])
def test_vectors_are_finite_and_normalised(provider: EmbeddingProvider, text: str) -> None:
    vector = asyncio.run(provider.embed_documents([text]))[0]
    assert np.all(np.isfinite(vector))
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3


def test_deterministic_across_calls_and_instances(provider: EmbeddingProvider) -> None:
    first = asyncio.run(provider.embed_documents(["same text"]))
    again = asyncio.run(provider.embed_documents(["same text"]))
    other = asyncio.run(FACTORIES["hashing-256"]().embed_documents(["same text"]))
    np.testing.assert_array_equal(first, again)
    if provider.dimension == 256:
        np.testing.assert_array_equal(first, other)


def test_model_id_is_stable_and_dimension_specific() -> None:
    assert HashingEmbedder(256).model_id == HashingEmbedder(256).model_id
    assert HashingEmbedder(256).model_id != HashingEmbedder(64).model_id
    assert HashingEmbedder(256).model_id != HashingEmbedder(256, revision="2").model_id


def test_lexical_similarity_is_meaningful() -> None:
    embedder = HashingEmbedder(512)
    query, related, unrelated = asyncio.run(
        embedder.embed_documents(
            [
                "how does hybrid retrieval combine rankings",
                "hybrid retrieval combines dense and lexical rankings",
                "the weather in the mountains is cold in winter",
            ]
        )
    )
    assert float(query @ related) > float(query @ unrelated) + 0.2
    assert float(query @ query) == pytest.approx(1.0, abs=1e-4)


def test_works_for_russian_text() -> None:
    embedder = HashingEmbedder(512)
    query, related, unrelated = asyncio.run(
        embedder.embed_documents(
            ["как работает гибридный поиск", "гибридный поиск объединяет ранги", "погода в горах"]
        )
    )
    assert float(query @ related) > float(query @ unrelated)


def test_rejects_degenerate_configuration() -> None:
    with pytest.raises(Exception, match="at least 8"):
        HashingEmbedder(4)
