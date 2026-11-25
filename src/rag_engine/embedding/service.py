"""Bounded, content-addressed embedding preparation outside the writer transaction."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from rag_engine.domain import (
    Chunk,
    ChunkId,
    EmbeddingModel,
    ErrorCode,
    RagError,
    TextHash,
)
from rag_engine.ports import Clock, EmbeddingCache, EmbeddingProvider, StoredEmbedding, TokenCounter


@dataclass(frozen=True, slots=True)
class EmbeddingPlan:
    """Publisher input; nothing has been written to the store yet."""

    by_chunk: Mapping[ChunkId, TextHash]
    new: tuple[StoredEmbedding, ...]


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        model: EmbeddingModel,
        counter: TokenCounter,
        clock: Clock,
        *,
        batch_size: int = 16,
        batch_tokens: int = 4096,
        concurrency: int = 1,
        timeout_s: float = 180.0,
        max_attempts: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if provider.model_id != model.id or provider.dimension != model.dimension:
            raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "provider and model identity differ")
        if not model.normalized:
            raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "stored embeddings require L2 policy")
        if min(batch_size, batch_tokens, concurrency, max_attempts) <= 0 or timeout_s <= 0:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "embedding limits must be positive")
        self._provider = provider
        self._model = model
        self._counter = counter
        self._clock = clock
        self._batch_size = batch_size
        self._batch_tokens = batch_tokens
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter
        self._semaphore = asyncio.Semaphore(concurrency)

    def document_input(self, chunk: Chunk) -> str:
        """Exactly the text submitted to a document provider, including its model prefix."""
        body = "\n".join((*chunk.heading_path, chunk.text))
        return self._model.document_prefix + body

    async def prepare(self, chunks: Sequence[Chunk], cache: EmbeddingCache) -> EmbeddingPlan:
        by_chunk: dict[ChunkId, TextHash] = {}
        unique: dict[TextHash, tuple[str, int]] = {}
        limit = min(self._provider.max_input_tokens, self._model.max_input_tokens)
        for chunk in chunks:
            if not chunk.indexed:
                continue
            value = self.document_input(chunk)
            key = TextHash.of_text(value)
            by_chunk[chunk.id] = key
            if key in unique:
                continue
            tokens = self._counter.count(value)
            if tokens > limit or tokens > self._batch_tokens:
                raise RagError(
                    ErrorCode.E_RESOURCE_GATE,
                    "embedding input exceeds its token limit",
                    context={"tokens": tokens, "limit": min(limit, self._batch_tokens)},
                )
            unique[key] = (value, tokens)

        cached = cache.get_many(self._model.id, tuple(unique))
        for cached_item in cached.values():
            if (
                cached_item.vector.shape != (self._model.dimension,)
                or not bool(np.isfinite(cached_item.vector).all())
                or abs(float(np.linalg.norm(cached_item.vector)) - 1.0) >= 1e-3
            ):
                raise RagError(ErrorCode.E_INTEGRITY, "cached embedding is invalid")
        pending = [(key, *value) for key, value in unique.items() if key not in cached]
        batches: list[list[tuple[TextHash, str, int]]] = []
        current: list[tuple[TextHash, str, int]] = []
        current_tokens = 0
        for item in pending:
            if current and (
                len(current) >= self._batch_size or current_tokens + item[2] > self._batch_tokens
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item[2]
        if current:
            batches.append(current)

        created: list[StoredEmbedding] = []
        for batch in batches:
            vectors = await self._call([text for _, text, _ in batch])
            if (
                not isinstance(vectors, np.ndarray)
                or vectors.shape != (len(batch), self._model.dimension)
                or vectors.dtype != np.float32
            ):
                raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "provider returned invalid vectors")
            for (key, _, _), vector in zip(batch, vectors, strict=True):
                if not np.isfinite(vector).all():
                    raise RagError(ErrorCode.E_INTEGRITY, "provider returned non-finite vector")
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0:
                    raise RagError(ErrorCode.E_INTEGRITY, "provider returned a zero vector")
                normalized = np.asarray(vector / norm, dtype=np.float32)
                created.append(StoredEmbedding(key, normalized, 1.0, self._clock.now()))
        return EmbeddingPlan(MappingProxyType(by_chunk), tuple(created))

    async def _call(self, texts: Sequence[str]) -> np.ndarray:
        async with self._semaphore:
            for attempt in range(self._max_attempts):
                try:
                    return await asyncio.wait_for(
                        self._provider.embed_documents(texts), timeout=self._timeout_s
                    )
                except TimeoutError:
                    error = RagError(ErrorCode.E_PROVIDER_TIMEOUT, "embedding provider timed out")
                except RagError as exc:
                    error = exc
                except Exception as exc:
                    raise RagError(
                        ErrorCode.E_PROVIDER_UNAVAILABLE,
                        "embedding provider failed unexpectedly",
                        retryable=False,
                        context={"error_type": type(exc).__name__},
                    ) from exc
                if not error.retryable or attempt + 1 == self._max_attempts:
                    raise error
                delay = min(30.0, 0.5 * 2**attempt) * (0.8 + 0.4 * self._jitter())
                await self._sleep(delay)
        raise AssertionError("unreachable")
