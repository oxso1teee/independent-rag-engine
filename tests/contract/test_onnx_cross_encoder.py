"""The offline ONNX cross-encoder: pinned files, bounded pairs, finite descending scores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_engine.domain import ChunkId, ChunkRef, ErrorCode, RagError
from rag_engine.embedding.cross_encoder import OnnxCrossEncoder
from rag_engine.embedding.onnx import model_directory_digest
from rag_engine.ports import Candidate, RerankBudget, Reranker
from rag_engine.testing.fake_reranker import word_overlap
from tests.factories import make_ids

pytestmark = pytest.mark.contract

IDS = make_ids()
VERSION = IDS.document_version_id()
DOCUMENT = IDS.document_id()


class Backend:
    """Stands in for FastEmbed's TextCrossEncoder."""

    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self.calls: list[tuple[str, list[str], int]] = []
        self._scores = scores

    def rerank(self, query: str, documents: Iterable[str], batch_size: int) -> Iterable[float]:
        texts = list(documents)
        self.calls.append((query, texts, batch_size))
        return (
            list(self._scores)
            if self._scores is not None
            else [word_overlap(query, t) for t in texts]
        )


def model_dir(tmp_path: Path) -> tuple[Path, str]:
    directory = tmp_path / "cross-encoder"
    directory.mkdir()
    (directory / "model.onnx").write_bytes(b"weights")
    (directory / "tokenizer.json").write_text("{}")
    return directory, model_directory_digest(directory)


def pool(*texts: str) -> list[Candidate]:
    return [
        Candidate(ChunkRef(ChunkId.derive(VERSION, "ce", i), DOCUMENT, VERSION), text, 1.0)
        for i, text in enumerate(texts)
    ]


def encoder(tmp_path: Path, backend: Backend | None = None, **options: object) -> OnnxCrossEncoder:
    directory, revision = model_dir(tmp_path)
    return OnnxCrossEncoder(
        "BAAI/bge-reranker-v2-m3-int8",
        directory,
        revision,
        backend=backend or Backend(),
        pair_counter=lambda query, text: len(query.split()) + len(text.split()),
        **options,  # type: ignore[arg-type]
    )


def test_it_satisfies_the_reranker_port_and_orders_by_descending_logit(tmp_path: Path) -> None:
    ranker = encoder(tmp_path)
    assert isinstance(ranker, Reranker)
    ranked = ranker.rerank(
        "nightly backups", pool("menu of the day", "nightly backups run", "backups"), RerankBudget()
    )
    assert [r.candidate.text for r in ranked] == [
        "nightly backups run",
        "backups",
        "menu of the day",
    ]
    assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)
    assert all(math.isfinite(r.score) for r in ranked)


def test_the_pool_is_cut_to_the_budget_and_scored_in_batches(tmp_path: Path) -> None:
    backend = Backend()
    ranker = encoder(tmp_path, backend, batch_size=4)
    ranker.rerank("a", pool("a one", "a two", "a three"), RerankBudget(pool_size=2))
    assert backend.calls == [("a", ["a one", "a two"], 4)]
    assert ranker.rerank("a", [], RerankBudget()) == []


def test_files_that_changed_since_pinning_are_refused(tmp_path: Path) -> None:
    directory, revision = model_dir(tmp_path)
    (directory / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(RagError) as info:
        OnnxCrossEncoder("m", directory, revision, backend=Backend(), pair_counter=lambda q, t: 1)
    assert info.value.code is ErrorCode.E_EMBED_MODEL_MISMATCH


def test_a_pair_longer_than_the_window_is_refused_not_truncated(tmp_path: Path) -> None:
    backend = Backend()
    ranker = encoder(tmp_path, backend, max_pair_tokens=5)
    with pytest.raises(RagError) as info:
        ranker.rerank("one two", pool("short", "this passage has too many words"), RerankBudget())
    assert info.value.code is ErrorCode.E_RESOURCE_GATE
    assert backend.calls == []  # nothing reached the model


@pytest.mark.parametrize("scores", [[1.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_unusable_scores_are_an_error(tmp_path: Path, scores: list[float]) -> None:
    ranker = encoder(tmp_path, Backend(scores))
    with pytest.raises(RagError) as info:
        ranker.rerank("q", pool("a", "b"), RerankBudget())
    assert info.value.code is ErrorCode.E_PROVIDER_UNAVAILABLE


def test_the_real_backend_is_loaded_offline_from_the_pinned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, revision = model_dir(tmp_path)
    loaded: list[dict[str, object]] = []

    class TextCrossEncoder(Backend):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            loaded.append(kwargs)

    monkeypatch.setattr(
        "rag_engine.embedding.cross_encoder.importlib.import_module",
        lambda name: SimpleNamespace(TextCrossEncoder=TextCrossEncoder),
    )
    OnnxCrossEncoder(
        "BAAI/bge-reranker-v2-m3-int8", directory, revision, pair_counter=lambda q, t: 1
    )
    assert loaded[0]["local_files_only"] is True
    assert loaded[0]["specific_model_path"] == str(directory.resolve())
    assert loaded[0]["model_name"] == "BAAI/bge-reranker-v2-m3-int8"


def test_a_missing_extra_is_a_configuration_error_and_never_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, revision = model_dir(tmp_path)

    def refuse(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("rag_engine.embedding.cross_encoder.importlib.import_module", refuse)
    with pytest.raises(RagError) as info:
        OnnxCrossEncoder("m", directory, revision, pair_counter=lambda q, t: 1)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_a_released_reranker_refuses_further_use_and_drops_the_backend(tmp_path: Path) -> None:
    ranker = encoder(tmp_path)
    assert ranker.rerank("backups", pool("backups run"), RerankBudget())
    ranker.release()
    assert ranker._backend is None
    with pytest.raises(RagError) as info:
        ranker.rerank("backups", pool("backups run"), RerankBudget())
    assert info.value.code is ErrorCode.E_RESOURCE_GATE
    assert ranker.rerank("backups", [], RerankBudget()) == []
