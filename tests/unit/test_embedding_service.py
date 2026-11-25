"""Embedding preparation is deterministic, bounded and never writes partial results."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, ErrorCode, RagError, TextHash
from rag_engine.embedding import EmbeddingService
from rag_engine.ports import StoredEmbedding, Vectors
from rag_engine.testing import FakeClock
from tests.factories import T0, make_chunk, make_ids


class Counter:
    counter_id = "words/1"

    def count(self, text: str) -> int:
        return len(text.split())


class Cache:
    def __init__(self) -> None:
        self.values: dict[TextHash, StoredEmbedding] = {}

    def get_many(
        self, model_id: EmbeddingModelId, hashes: Sequence[TextHash]
    ) -> Mapping[TextHash, StoredEmbedding]:
        return {key: self.values[key] for key in hashes if key in self.values}


class Provider:
    dimension = 8
    max_input_tokens = 100

    def __init__(self, model_id: EmbeddingModelId) -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []
        self.failures: list[Exception | None] = []
        self.bad_shape = False

    async def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.calls.append(list(texts))
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        if self.bad_shape:
            return np.zeros((len(texts), 7), dtype=np.float32)
        vectors = np.zeros((len(texts), 8), dtype=np.float32)
        vectors[:, 0] = 2.0
        return vectors

    async def embed_query(self, text: str) -> Vectors:
        return await self.embed_documents([text])


def model() -> EmbeddingModel:
    model_id = EmbeddingModelId.build(
        provider="test",
        name="record",
        revision="1",
        dimension=8,
        normalized=True,
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    return EmbeddingModel(
        id=model_id,
        provider="test",
        name="record",
        revision="1",
        dimension=8,
        normalized=True,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_input_tokens=100,
        created_at=T0,
    )


async def no_delay(_seconds: float) -> None:
    return None


def test_exact_input_hash_includes_heading_and_prefix_and_deduplicates() -> None:
    selected = model()
    provider = Provider(selected.id)
    service = EmbeddingService(provider, selected, Counter(), FakeClock())
    version = make_ids().document_version_id()
    first = replace(make_chunk("body", version_id=version), heading_path=("A",))
    same = replace(make_chunk("body", version_id=version, ordinal=1), heading_path=("A",))
    other = replace(make_chunk("body", version_id=version, ordinal=2), heading_path=("B",))
    result = asyncio.run(service.prepare([first, same, other], Cache()))
    assert provider.calls == [["passage: A\nbody", "passage: B\nbody"]]
    assert result.by_chunk[first.id] == result.by_chunk[same.id]
    assert result.by_chunk[first.id] != result.by_chunk[other.id]
    assert result.by_chunk[first.id] != first.text_hash
    assert len(result.new) == 2
    assert all(abs(float(np.linalg.norm(item.vector)) - 1) < 1e-6 for item in result.new)


def test_batch_count_and_token_budget() -> None:
    selected = model()
    provider = Provider(selected.id)
    service = EmbeddingService(
        provider, selected, Counter(), FakeClock(), batch_size=2, batch_tokens=4
    )
    version = make_ids().document_version_id()
    chunks = [make_chunk(f"word{i}", version_id=version, ordinal=i) for i in range(3)]
    assert len(asyncio.run(service.prepare(chunks, Cache())).new) == 3
    assert [len(call) for call in provider.calls] == [2, 1]


def test_retry_only_for_retryable_and_no_partial_result() -> None:
    selected = model()
    provider = Provider(selected.id)
    provider.failures = [RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "temporary")]
    service = EmbeddingService(provider, selected, Counter(), FakeClock(), sleep=no_delay)
    chunk = make_chunk("hello", version_id=make_ids().document_version_id())
    assert len(asyncio.run(service.prepare([chunk], Cache())).new) == 1
    assert len(provider.calls) == 2

    provider.calls.clear()
    provider.failures = [RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "permanent")]
    with pytest.raises(RagError) as info:
        asyncio.run(service.prepare([chunk], Cache()))
    assert info.value.code is ErrorCode.E_EMBED_DIM_MISMATCH
    assert len(provider.calls) == 1


def test_bad_provider_shape_never_produces_a_plan() -> None:
    selected = model()
    provider = Provider(selected.id)
    provider.bad_shape = True
    service = EmbeddingService(provider, selected, Counter(), FakeClock())
    chunk = make_chunk("hello", version_id=make_ids().document_version_id())
    with pytest.raises(RagError) as info:
        asyncio.run(service.prepare([chunk], Cache()))
    assert info.value.code is ErrorCode.E_EMBED_DIM_MISMATCH


def test_cache_hit_skips_provider_and_rejects_corrupt_vector() -> None:
    selected = model()
    provider = Provider(selected.id)
    service = EmbeddingService(provider, selected, Counter(), FakeClock())
    chunk = make_chunk("hello", version_id=make_ids().document_version_id())
    key = TextHash.of_text("passage: hello")
    cache = Cache()
    good = np.zeros(8, dtype=np.float32)
    good[0] = 1
    cache.values[key] = StoredEmbedding(key, good, 1.0, T0)
    assert asyncio.run(service.prepare([chunk], cache)).new == ()
    assert provider.calls == []
    cache.values[key] = StoredEmbedding(key, np.ones(8, dtype=np.float32), 1.0, T0)
    with pytest.raises(RagError) as info:
        asyncio.run(service.prepare([chunk], cache))
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_timeout_retries_and_token_limit_fails_without_provider_call() -> None:
    selected = model()
    provider = Provider(selected.id)
    provider.failures = [TimeoutError()]
    service = EmbeddingService(provider, selected, Counter(), FakeClock(), sleep=no_delay)
    chunk = make_chunk("hello", version_id=make_ids().document_version_id())
    assert len(asyncio.run(service.prepare([chunk], Cache())).new) == 1
    assert len(provider.calls) == 2

    provider.calls.clear()
    too_small = EmbeddingService(
        provider, selected, Counter(), FakeClock(), batch_tokens=1, sleep=no_delay
    )
    with pytest.raises(RagError) as info:
        asyncio.run(too_small.prepare([chunk], Cache()))
    assert info.value.code is ErrorCode.E_RESOURCE_GATE
    assert provider.calls == []


def test_later_batch_failure_returns_no_partial_plan() -> None:
    selected = model()
    provider = Provider(selected.id)
    provider.failures = [None, TimeoutError()]
    service = EmbeddingService(
        provider,
        selected,
        Counter(),
        FakeClock(),
        batch_size=1,
        max_attempts=1,
        sleep=no_delay,
    )
    version = make_ids().document_version_id()
    chunks = [make_chunk(f"word{i}", version_id=version, ordinal=i) for i in range(2)]
    with pytest.raises(RagError) as info:
        asyncio.run(service.prepare(chunks, Cache()))
    assert info.value.code is ErrorCode.E_PROVIDER_TIMEOUT
    assert len(provider.calls) == 2
