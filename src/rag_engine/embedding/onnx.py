"""Offline, CPU-only ONNX embeddings through the optional FastEmbed extra."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, ErrorCode, RagError
from rag_engine.ports.embedding import Vectors


class _Backend(Protocol):
    def embed(
        self, documents: Sequence[str], *, batch_size: int, parallel: int | None
    ) -> Iterable[np.ndarray]: ...


def model_directory_digest(directory: Path) -> str:
    """Fingerprint every local model file, including tokenizer and ONNX weights."""
    if not directory.is_dir() or directory.is_symlink():
        raise RagError(ErrorCode.E_CONFIG_INVALID, "ONNX model directory is unavailable")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in directory.rglob("*")):
        raise RagError(ErrorCode.E_CONFIG_INVALID, "ONNX model files are missing or linked")
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def exact_token_counter(model_dir: Path) -> Callable[[str], int]:
    """Count tokens as the model sees them: special tokens included, nothing truncated.

    The generic ``HfTokenCounter`` leaves special tokens out; an input that fits by that count
    can still exceed the model window once ``<s>`` and ``</s>`` are added, and the inference
    library would then cut it silently.
    """
    try:
        tokenizers = importlib.import_module("tokenizers")
    except ImportError as exc:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, "the onnx extra is required for this provider"
        ) from exc
    try:
        tokenizer = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    except Exception as exc:  # the library raises a bare Exception for unreadable files
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, "the ONNX model has no readable tokenizer.json"
        ) from exc
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return count


class OnnxEmbedder:
    """Load only explicitly provisioned local files; never download a model on use."""

    def __init__(
        self,
        model: EmbeddingModel,
        model_dir: Path,
        *,
        threads: int = 1,
        backend: _Backend | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if model.provider != "onnx" or not model.normalized or threads < 1:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "invalid ONNX embedding configuration")
        if model_directory_digest(model_dir) != model.revision:
            raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "ONNX model files changed")
        self._model = model
        if backend is None:
            token_counter = token_counter or exact_token_counter(model_dir)
            try:
                fastembed = importlib.import_module("fastembed")
            except ImportError as exc:
                raise RagError(
                    ErrorCode.E_CONFIG_INVALID, "the onnx extra is required for this provider"
                ) from exc
            try:
                if model.name == "intfloat/multilingual-e5-small":
                    if (
                        model.dimension != 384
                        or model.max_input_tokens > 512
                        or model.query_prefix != "query: "
                        or model.document_prefix != "passage: "
                    ):
                        raise RagError(
                            ErrorCode.E_CONFIG_INVALID,
                            "E5-small requires 384 dimensions, 512 tokens, and E5 prefixes",
                        )
                    supported = fastembed.TextEmbedding.list_supported_models()
                    if not any(item["model"] == model.name for item in supported):
                        metadata = importlib.import_module("fastembed.common.model_description")
                        fastembed.TextEmbedding.add_custom_model(
                            model=model.name,
                            pooling=metadata.PoolingType.MEAN,
                            normalization=True,
                            sources=metadata.ModelSource(hf=model.name),
                            dim=384,
                            model_file="onnx/model.onnx",
                            license="mit",
                        )
                backend = cast(
                    _Backend,
                    fastembed.TextEmbedding(
                        model_name=model.name,
                        specific_model_path=str(model_dir.resolve()),
                        local_files_only=True,
                        threads=threads,
                        providers=["CPUExecutionProvider"],
                    ),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise RagError(
                    ErrorCode.E_PROVIDER_UNAVAILABLE,
                    "the local ONNX model could not be loaded",
                    retryable=False,
                    context={"error_type": type(exc).__name__},
                ) from exc
        self._backend: _Backend | None = backend
        self._token_counter = token_counter
        if model_directory_digest(model_dir) != model.revision:
            raise RagError(ErrorCode.E_EMBED_MODEL_MISMATCH, "ONNX model files changed during load")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-onnx")
        self._active: Future[Vectors] | None = None

    @property
    def model_id(self) -> EmbeddingModelId:
        return self._model.id

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def max_input_tokens(self) -> int:
        return self._model.max_input_tokens

    async def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._backend is None:
            raise RagError(ErrorCode.E_RESOURCE_GATE, "the ONNX model was released from memory")
        if self._active is not None and not self._active.done():
            raise RagError(ErrorCode.E_RESOURCE_GATE, "previous ONNX inference is still running")
        self._active = self._executor.submit(self._infer, list(texts))
        try:
            vectors = await asyncio.wrap_future(self._active)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RagError(
                ErrorCode.E_PROVIDER_UNAVAILABLE,
                "ONNX inference failed",
                retryable=False,
                context={"error_type": type(exc).__name__},
            ) from exc
        if vectors.shape != (len(texts), self.dimension) or not bool(np.isfinite(vectors).all()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "ONNX returned invalid vectors")
        norms = np.linalg.norm(vectors, axis=1)
        if not bool(np.isfinite(norms).all()) or bool((norms == 0).any()):
            raise RagError(ErrorCode.E_EMBED_DIM_MISMATCH, "ONNX returned zero vectors")
        return np.asarray(vectors / norms[:, None], dtype=np.float32)

    def _infer(self, texts: list[str]) -> Vectors:
        if self._token_counter is not None:
            for text in texts:
                tokens = self._token_counter(text)
                if tokens > self._model.max_input_tokens:
                    raise RagError(
                        ErrorCode.E_RESOURCE_GATE,
                        "embedding input exceeds the model token limit",
                        context={"tokens": tokens, "limit": self._model.max_input_tokens},
                    )
        backend = self._backend
        if backend is None:
            raise RagError(ErrorCode.E_RESOURCE_GATE, "the ONNX model was released from memory")
        rows = list(backend.embed(texts, batch_size=len(texts), parallel=None))
        return np.asarray(rows, dtype=np.float32)

    async def embed_query(self, text: str) -> Vectors:
        return await self.embed_documents([text])

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def release(self) -> None:
        """Drop the loaded model so its memory can be returned; any later use is refused.

        For a short-lived process that is about to hand the machine to a local generator. A long
        running service keeps its models loaded and never calls this.
        """
        self._backend = None
        gc.collect()
