"""Offline ONNX cross-encoder reranking through the optional FastEmbed extra.

Like the ONNX embedder it loads only a provisioned local directory whose fingerprint was pinned,
never downloads anything, and refuses an input instead of cutting it: a (query, passage) pair that
does not fit the model window is an error, which the rerank stage turns into a fallback to the
fused order. Scores are raw logits, comparable only within one model.
"""

from __future__ import annotations

import gc
import importlib
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, cast

from rag_engine.domain import ErrorCode, RagError
from rag_engine.embedding.onnx import model_directory_digest
from rag_engine.ports import Candidate, RerankBudget, RerankedCandidate


class _Backend(Protocol):
    def rerank(self, query: str, documents: Iterable[str], batch_size: int) -> Iterable[float]: ...


def exact_pair_counter(model_dir: Path) -> Callable[[str, str], int]:
    """Tokens of a (query, passage) pair as the model sees them, special tokens included."""
    try:
        tokenizers = importlib.import_module("tokenizers")
    except ImportError as exc:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, "the rerank extra is required for this provider"
        ) from exc
    try:
        tokenizer = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    except Exception as exc:  # the library raises a bare Exception for unreadable files
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, "the reranker model has no readable tokenizer.json"
        ) from exc
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count(query: str, text: str) -> int:
        return len(tokenizer.encode(query, text).ids)

    return count


class OnnxCrossEncoder:
    def __init__(
        self,
        name: str,
        model_dir: Path,
        revision: str,
        *,
        max_pair_tokens: int = 512,
        batch_size: int = 8,
        threads: int = 1,
        backend: _Backend | None = None,
        pair_counter: Callable[[str, str], int] | None = None,
    ) -> None:
        if min(max_pair_tokens, batch_size, threads) < 1:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "invalid cross-encoder limits")
        if model_directory_digest(model_dir) != revision:
            raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "reranker model files changed")
        self._name = name
        self._max_pair_tokens = max_pair_tokens
        self._batch_size = batch_size
        self._count = pair_counter or exact_pair_counter(model_dir)
        if backend is None:
            try:
                fastembed = importlib.import_module("fastembed.rerank.cross_encoder")
            except ImportError as exc:
                raise RagError(
                    ErrorCode.E_CONFIG_INVALID, "the rerank extra is required for this provider"
                ) from exc
            try:
                backend = cast(
                    _Backend,
                    fastembed.TextCrossEncoder(
                        model_name=name,
                        specific_model_path=str(model_dir.resolve()),
                        local_files_only=True,
                        threads=threads,
                        providers=["CPUExecutionProvider"],
                    ),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise RagError(
                    ErrorCode.E_PROVIDER_UNAVAILABLE,
                    "the local reranker model could not be loaded",
                    retryable=False,
                    context={"error_type": type(exc).__name__},
                ) from exc
        self._backend: _Backend | None = backend

    def release(self) -> None:
        """Drop the loaded model; a later call is refused (see ``OnnxEmbedder.release``)."""
        self._backend = None
        gc.collect()

    def rerank(
        self, query: str, candidates: Sequence[Candidate], budget: RerankBudget
    ) -> Sequence[RerankedCandidate]:
        pool = list(candidates[: budget.pool_size])
        if not pool:
            return []
        backend = self._backend
        if backend is None:
            raise RagError(ErrorCode.E_RESOURCE_GATE, "the reranker model was released from memory")
        for candidate in pool:
            tokens = self._count(query, candidate.text)
            if tokens > self._max_pair_tokens:
                raise RagError(
                    ErrorCode.E_RESOURCE_GATE,
                    "a rerank input exceeds the model window",
                    context={"tokens": tokens, "limit": self._max_pair_tokens},
                )
        texts = [c.text for c in pool]
        scores = [float(s) for s in backend.rerank(query, texts, self._batch_size)]
        if len(scores) != len(pool) or not all(math.isfinite(s) for s in scores):
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "the reranker returned unusable scores",
                retryable=False,
            )
        ranked = [RerankedCandidate(c, s) for c, s in zip(pool, scores, strict=True)]
        ranked.sort(key=lambda item: (-item.score, str(item.candidate.chunk_ref.chunk_id)))
        return ranked
