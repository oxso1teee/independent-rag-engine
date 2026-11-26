"""Offline ONNX adapter contract with a deterministic inference backend."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, ErrorCode, RagError
from rag_engine.embedding.onnx import OnnxEmbedder, model_directory_digest
from rag_engine.ports import EmbeddingProvider
from tests.factories import T0


class Backend:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(
        self, documents: Sequence[str], *, batch_size: int, parallel: int | None
    ) -> Iterable[np.ndarray]:
        assert batch_size == len(documents) and parallel is None
        self.calls.append(list(documents))
        for _ in documents:
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[0] = 2.0
            yield vector


def local_model(tmp_path: Path) -> tuple[EmbeddingModel, Path]:
    directory = tmp_path / "local-model"
    directory.mkdir()
    (directory / "model.onnx").write_bytes(b"test model bytes")
    (directory / "tokenizer.json").write_text("{}")
    revision = model_directory_digest(directory)
    model_id = EmbeddingModelId.build(
        provider="onnx",
        name="test/local",
        revision=revision,
        dimension=4,
        normalized=True,
        query_prefix="",
        document_prefix="",
    )
    return (
        EmbeddingModel(
            id=model_id,
            provider="onnx",
            name="test/local",
            revision=revision,
            dimension=4,
            normalized=True,
            max_input_tokens=64,
            created_at=T0,
        ),
        directory,
    )


def test_offline_adapter_shape_order_and_normalization(tmp_path: Path) -> None:
    model, directory = local_model(tmp_path)
    backend = Backend()
    provider = OnnxEmbedder(model, directory, backend=backend)
    assert isinstance(provider, EmbeddingProvider)
    vectors = asyncio.run(provider.embed_documents(["a", "b"]))
    assert vectors.shape == (2, 4) and vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[:, 0], np.ones(2, dtype=np.float32))
    assert backend.calls == [["a", "b"]]
    assert asyncio.run(provider.embed_documents([])).shape == (0, 4)
    assert backend.calls == [["a", "b"]]


def test_model_files_and_tokenizer_are_part_of_identity(tmp_path: Path) -> None:
    model, directory = local_model(tmp_path)
    (directory / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(RagError) as info:
        OnnxEmbedder(model, directory, backend=Backend())
    assert info.value.code is ErrorCode.E_EMBED_MODEL_MISMATCH


def test_wrong_dimension_fails_without_persisting(tmp_path: Path) -> None:
    model, directory = local_model(tmp_path)
    provider = OnnxEmbedder(model, directory, backend=Backend(dimension=3))
    with pytest.raises(RagError) as info:
        asyncio.run(provider.embed_documents(["x"]))
    assert info.value.code is ErrorCode.E_EMBED_DIM_MISMATCH


def test_missing_extra_does_not_attempt_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, directory = local_model(tmp_path)

    def missing(_name: str) -> None:
        raise ImportError("not installed")

    monkeypatch.setattr("rag_engine.embedding.onnx.importlib.import_module", missing)
    with pytest.raises(RagError) as info:
        OnnxEmbedder(model, directory)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_e5_registers_offline_fastembed_model_with_required_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, directory = local_model(tmp_path)
    name = "intfloat/multilingual-e5-small"
    model = replace(
        original,
        id=EmbeddingModelId.build(
            provider="onnx",
            name=name,
            revision=original.revision,
            dimension=384,
            normalized=True,
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        name=name,
        dimension=384,
        max_input_tokens=512,
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    registered: list[dict[str, object]] = []
    loaded: list[dict[str, object]] = []

    class TextEmbedding(Backend):
        @staticmethod
        def list_supported_models() -> list[dict[str, str]]:
            return []

        @staticmethod
        def add_custom_model(**kwargs: object) -> None:
            registered.append(kwargs)

        def __init__(self, **kwargs: object) -> None:
            super().__init__(384)
            loaded.append(kwargs)

    def module(module_name: str) -> SimpleNamespace:
        if module_name == "fastembed":
            return SimpleNamespace(TextEmbedding=TextEmbedding)
        if module_name == "fastembed.common.model_description":
            return SimpleNamespace(
                PoolingType=SimpleNamespace(MEAN="mean"),
                ModelSource=lambda **kwargs: SimpleNamespace(**kwargs),
            )
        raise AssertionError(module_name)

    monkeypatch.setattr("rag_engine.embedding.onnx.importlib.import_module", module)
    provider = OnnxEmbedder(model, directory)
    assert registered[0]["model"] == name
    assert registered[0]["pooling"] == "mean"
    assert registered[0]["dim"] == 384
    assert loaded[0]["local_files_only"] is True
    assert asyncio.run(provider.embed_documents(["passage: example"])).shape == (1, 384)
    provider.close()
    with pytest.raises(RagError) as info:
        OnnxEmbedder(replace(model, max_input_tokens=513), directory)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_timed_out_inference_cannot_start_a_second_cpu_job(tmp_path: Path) -> None:
    model, directory = local_model(tmp_path)
    release = threading.Event()

    class BlockingBackend(Backend):
        def embed(
            self, documents: Sequence[str], *, batch_size: int, parallel: int | None
        ) -> Iterable[np.ndarray]:
            release.wait(timeout=1)
            return super().embed(documents, batch_size=batch_size, parallel=parallel)

    provider = OnnxEmbedder(model, directory, backend=BlockingBackend())

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(provider.embed_documents(["first"]), timeout=0.01)
        with pytest.raises(RagError) as info:
            await provider.embed_documents(["second"])
        assert info.value.code is ErrorCode.E_RESOURCE_GATE

    try:
        asyncio.run(run())
    finally:
        release.set()
        provider.close()
