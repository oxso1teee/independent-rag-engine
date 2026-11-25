"""Ollama's native embedding API, pinned to a locally installed model digest."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, ErrorCode, RagError
from rag_engine.ports.embedding import Vectors


class OllamaEmbedder:
    """Async provider that refuses missing or changed models and never truncates input."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 180.0,
        keep_alive: str = "0",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if model.provider != "ollama" or not model.revision or not model.normalized:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID, "Ollama model needs a pinned digest and L2 policy"
            )
        if timeout_s <= 0:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "Ollama timeout must be positive")
        self._model = model
        self._keep_alive = keep_alive
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout_s, trust_env=False
        )
        self._owns_client = client is None

    @property
    def model_id(self) -> EmbeddingModelId:
        return self._model.id

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def max_input_tokens(self) -> int:
        return self._model.max_input_tokens

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        await self._verify_digest()
        payload = {
            "model": self._model.name,
            "input": list(texts),
            "truncate": False,
            "keep_alive": self._keep_alive,
        }
        response = await self._request("POST", "/api/embed", json=payload)
        data = self._json_object(response)
        raw = data.get("embeddings")
        try:
            vectors = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "Ollama returned invalid embedding data",
                retryable=False,
            ) from exc
        if vectors.shape != (len(texts), self.dimension) or not bool(np.isfinite(vectors).all()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "Ollama returned invalid vectors")
        norms = np.linalg.norm(vectors, axis=1)
        if not bool(np.isfinite(norms).all()) or bool((norms == 0).any()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "Ollama returned zero vectors")
        return np.asarray(vectors / norms[:, None], dtype=np.float32)

    async def embed_query(self, text: str) -> Vectors:
        return await self.embed_documents([text])

    async def _verify_digest(self) -> None:
        response = await self._request("GET", "/api/tags")
        data = self._json_object(response)
        models = data.get("models")
        if not isinstance(models, list):
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "Ollama returned invalid model metadata",
                retryable=False,
            )
        for entry in models:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") == self._model.name:
                if entry.get("digest") != self._model.revision:
                    raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "Ollama model digest changed")
                return
        raise RagError(
            ErrorCode.E_PROVIDER_UNAVAILABLE,
            "Ollama embedding model is not installed locally",
            retryable=False,
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise RagError(ErrorCode.E_PROVIDER_TIMEOUT, "Ollama request timed out") from exc
        except httpx.RequestError as exc:
            raise RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "Ollama is unavailable") from exc
        if response.status_code == 429:
            raise RagError(ErrorCode.E_RATE_LIMITED, "Ollama rate limit")
        if response.status_code >= 400:
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "Ollama rejected the request",
                retryable=response.status_code >= 500,
                context={"status": response.status_code},
            )
        return response

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "Ollama returned invalid JSON",
                retryable=False,
            ) from exc
        if not isinstance(data, dict):
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "Ollama returned invalid JSON shape",
                retryable=False,
            )
        return data
