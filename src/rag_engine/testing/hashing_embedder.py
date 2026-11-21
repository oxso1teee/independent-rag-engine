"""Deterministic hashing embedder: lexical similarity without a model or any RAM."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from rag_engine.domain import EmbeddingModelId, require
from rag_engine.ports import Vectors

_WORD = re.compile(r"\w+")
_NGRAM = 3
_NGRAM_WEIGHT = 0.5
_EMPTY_FEATURE = "\x00empty"


class HashingEmbedder:
    """``EmbeddingProvider`` built from signed feature hashing.

    Features are lowercase words and the character trigrams of each word.
    Texts sharing vocabulary get a high cosine similarity; unrelated texts are
    close to orthogonal. Output rows are L2-normalised float32. Uses BLAKE2b,
    never Python's randomised ``hash()``, so results are identical across runs.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        max_input_tokens: int = 8192,
        revision: str = "1",
    ) -> None:
        require(dimension >= 8, "HashingEmbedder needs at least 8 dimensions")
        require(max_input_tokens > 0, "max_input_tokens must be positive")
        self._dimension = dimension
        self._max_input_tokens = max_input_tokens
        self._model_id = EmbeddingModelId.build(
            provider="fake",
            name="hashing",
            revision=revision,
            dimension=dimension,
            normalized=True,
        )

    @property
    def model_id(self) -> EmbeddingModelId:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        features: list[tuple[str, float]] = []
        for word in _WORD.findall(text.lower()):
            features.append((word, 1.0))
            padded = f"^{word}$"
            features.extend(
                (padded[i : i + _NGRAM], _NGRAM_WEIGHT) for i in range(len(padded) - _NGRAM + 1)
            )
        return features or [(_EMPTY_FEATURE, 1.0)]

    def _slot(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self._dimension
        return index, 1.0 if digest[4] & 1 else -1.0

    def _vectorize(self, text: str) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        vector = np.zeros(self._dimension, dtype=np.float64)
        for feature, weight in self._features(text):
            index, sign = self._slot(feature)
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # exact cancellation: fall back to a fixed unit vector per text
            vector[self._slot(text)[0]] = 1.0
            norm = 1.0
        return vector / norm

    async def embed_documents(self, texts: Sequence[str]) -> Vectors:
        rows = [self._vectorize(text) for text in texts]
        if not rows:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    async def embed_query(self, text: str) -> Vectors:
        return await self.embed_documents([text])
