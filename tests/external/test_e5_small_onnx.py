"""Real multilingual-e5-small inference through the production ONNX adapter (opt-in).

Set ``RAG_TEST_E5_DIR`` to a provisioned model directory (the upstream snapshot with
``onnx/model.onnx``, ``tokenizer.json`` and the tokenizer files). Without it every test here is
skipped, so the standard CI needs neither the weights nor a network. The suite runs with sockets
disabled, so a successful load also demonstrates that nothing is fetched.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, ErrorCode, RagError
from rag_engine.embedding.onnx import OnnxEmbedder, exact_token_counter, model_directory_digest
from tests.factories import T0

pytestmark = pytest.mark.external

NAME = "intfloat/multilingual-e5-small"
QUERY = "query: "
PASSAGE = "passage: "


@pytest.fixture(scope="module")
def directory() -> Path:
    raw = os.environ.get("RAG_TEST_E5_DIR")
    if not raw:
        pytest.skip("RAG_TEST_E5_DIR is not set")
    return Path(raw)


@pytest.fixture(scope="module")
def provider(directory: Path) -> Iterator[OnnxEmbedder]:
    revision = model_directory_digest(directory)
    model = EmbeddingModel(
        id=EmbeddingModelId.build(
            provider="onnx",
            name=NAME,
            revision=revision,
            dimension=384,
            normalized=True,
            query_prefix=QUERY,
            document_prefix=PASSAGE,
        ),
        provider="onnx",
        name=NAME,
        revision=revision,
        dimension=384,
        normalized=True,
        query_prefix=QUERY,
        document_prefix=PASSAGE,
        max_input_tokens=512,
        created_at=T0,
    )
    embedder = OnnxEmbedder(model, directory, threads=2)
    yield embedder
    embedder.close()


def embed(provider: OnnxEmbedder, texts: list[str]) -> np.ndarray:
    return asyncio.run(provider.embed_documents(texts))


def test_vectors_are_384_dimensional_unit_length_float32(provider: OnnxEmbedder) -> None:
    vectors = embed(provider, [PASSAGE + "Paris is the capital of France.", QUERY + "Париж"])
    assert provider.dimension == 384 and vectors.shape == (2, 384)
    assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_ranking_is_semantic_within_each_language(provider: OnnxEmbedder) -> None:
    # E5-small prefers a passage in the query's language over a better one in another language
    # (measured: a Russian question about France scored a Russian passage about Japan higher
    # than the English passage about Paris), so each language is ranked among its own passages.
    cases = {
        "en": (
            [
                PASSAGE + "Paris is the capital and largest city of France.",
                PASSAGE + "Tokyo is the capital of Japan and its largest city.",
                PASSAGE + "Photosynthesis converts light energy into chemical energy in plants.",
            ],
            {
                QUERY + "What is the capital of Japan?": 1,
                QUERY + "How do plants turn light into energy?": 2,
            },
        ),
        "ru": (
            [
                PASSAGE + "Париж является столицей и крупнейшим городом Франции.",
                PASSAGE + "Токио является столицей Японии и её крупнейшим городом.",
                PASSAGE + "Фотосинтез преобразует энергию света в химическую энергию растений.",
            ],
            {
                QUERY + "Какой город является столицей Франции?": 0,
                QUERY + "Как растения превращают свет в энергию?": 2,
            },
        ),
    }
    for passages, queries in cases.values():
        document_vectors = embed(provider, passages)
        for query, expected in queries.items():
            scores = document_vectors @ embed(provider, [query])[0]
            assert int(np.argmax(scores)) == expected, query


def test_inference_is_deterministic_and_batch_invariant(provider: OnnxEmbedder) -> None:
    texts = [
        PASSAGE + "Short.",
        PASSAGE + "A considerably longer passage on retrieval systems and evaluation. " * 8,
        PASSAGE + "Третий текст на русском языке для проверки пакетной обработки.",
    ]
    together = embed(provider, texts)
    np.testing.assert_allclose(together, embed(provider, texts), atol=1e-6)
    one_by_one = np.vstack([embed(provider, [text]) for text in texts])
    np.testing.assert_allclose(together, one_by_one, atol=1e-3)


def test_prefix_changes_the_vector(provider: OnnxEmbedder) -> None:
    text = "Quarterly revenue grew by twelve percent."
    prefixed, bare = embed(provider, [PASSAGE + text, text])
    assert float(prefixed @ bare) < 0.9999


def test_the_model_window_is_enforced_with_special_tokens_not_truncated(
    provider: OnnxEmbedder, directory: Path
) -> None:
    count = exact_token_counter(directory)

    def text_of(words: int) -> str:
        return PASSAGE + " ".join(["hello"] * words)

    words = 1
    while count(text_of(words)) < 512:
        words += 1
    assert count(text_of(words)) == 512  # the adapter counts <s> and </s> too
    assert embed(provider, [text_of(words)]).shape == (1, 384)
    with pytest.raises(RagError) as info:
        embed(provider, [text_of(words + 1)])
    assert info.value.code is ErrorCode.E_RESOURCE_GATE
    context = dict(info.value.context or {})
    tokens = context.get("tokens")
    assert context.get("limit") == 512
    assert isinstance(tokens, int) and tokens > 512


def test_a_second_concurrent_call_is_refused_instead_of_queued(
    provider: OnnxEmbedder, directory: Path
) -> None:
    count = exact_token_counter(directory)
    words = 1
    while count(PASSAGE + " ".join(["hello"] * (words + 1))) <= 500:
        words += 1
    long_text = PASSAGE + " ".join(["hello"] * words)  # about 500 tokens, within the window

    async def run() -> None:
        first = asyncio.create_task(provider.embed_documents([long_text] * 8))
        await asyncio.sleep(0.05)
        with pytest.raises(RagError) as info:
            await provider.embed_documents(["second"])
        assert info.value.code is ErrorCode.E_RESOURCE_GATE
        assert (await first).shape == (8, 384)

    asyncio.run(run())
