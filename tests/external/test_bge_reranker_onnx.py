"""The real multilingual cross-encoder through the production adapter (opt-in).

Set ``RAG_TEST_RERANK_DIR`` to the provisioned ``bge-reranker-v2-m3-int8`` directory (the ONNX
file under ``onnx/`` plus the tokenizer files). Without it every test here is skipped, so the
standard CI needs neither the weights nor a network; sockets stay disabled, so a successful run
also shows that nothing is fetched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_engine.domain import ChunkId, ChunkRef, ErrorCode, RagError
from rag_engine.embedding.cross_encoder import OnnxCrossEncoder
from rag_engine.embedding.onnx import model_directory_digest
from rag_engine.ports import Candidate, RerankBudget
from tests.factories import make_ids

pytestmark = pytest.mark.external

NAME = "BAAI/bge-reranker-v2-m3-int8"
IDS = make_ids()
VERSION = IDS.document_version_id()
DOCUMENT = IDS.document_id()


@pytest.fixture(scope="module")
def encoder() -> OnnxCrossEncoder:
    raw = os.environ.get("RAG_TEST_RERANK_DIR")
    if not raw:
        pytest.skip("RAG_TEST_RERANK_DIR is not set")
    directory = Path(raw)
    return OnnxCrossEncoder(NAME, directory, model_directory_digest(directory), threads=2)


def pool(*texts: str) -> list[Candidate]:
    return [
        Candidate(ChunkRef(ChunkId.derive(VERSION, "bge", i), DOCUMENT, VERSION), text, 1.0)
        for i, text in enumerate(texts)
    ]


def test_the_relevant_passage_is_first_in_english_and_in_russian(encoder: OnnxCrossEncoder) -> None:
    english = pool(
        "The weekly menu has soup, bread and a salad with olive oil.",
        "Backups run every night at 02:30 UTC and are kept for 14 days.",
        "Printers are on the third floor next to the kitchen.",
    )
    ranked = encoder.rerank("When do the backups run?", english, RerankBudget())
    assert ranked[0].candidate.text.startswith("Backups run")
    russian = pool(
        "Суточные при поездках по России составляют 1000 рублей в день.",
        "Резервные копии создаются ежедневно в 03:10 по московскому времени.",
        "Дежурная смена длится одну неделю.",
    )
    ranked = encoder.rerank("Во сколько создаются резервные копии?", russian, RerankBudget())
    assert ranked[0].candidate.text.startswith("Резервные копии")
    assert ranked[0].score > ranked[-1].score


def test_scoring_is_deterministic_and_a_cross_language_pair_still_works(
    encoder: OnnxCrossEncoder,
) -> None:
    candidates = pool("Копии хранятся 10 дней.", "The office has a kitchen.")
    first = encoder.rerank("How long are backups kept?", candidates, RerankBudget())
    second = encoder.rerank("How long are backups kept?", candidates, RerankBudget())
    assert [(r.candidate.text, r.score) for r in first] == [
        (r.candidate.text, r.score) for r in second
    ]
    assert first[0].candidate.text.startswith("Копии")


def test_a_pair_longer_than_the_window_is_refused_before_inference(
    encoder: OnnxCrossEncoder,
) -> None:
    with pytest.raises(RagError) as info:
        encoder.rerank("short question", pool("word " * 2000), RerankBudget())
    assert info.value.code is ErrorCode.E_RESOURCE_GATE
