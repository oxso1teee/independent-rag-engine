"""Contract of the lexical index port: documents carry the language of their chunk."""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import FrozenInstanceError

import pytest

from rag_engine.domain import ChunkId, Language, ScoredChunk
from rag_engine.ports import LexicalDocument, LexicalIndex
from tests.factories import make_ids

IDS = make_ids()
CHUNK_ID = ChunkId.derive(IDS.document_version_id(), "fp", 0)


def test_language_defaults_to_undetermined() -> None:
    assert LexicalDocument(CHUNK_ID, "body").language is Language.UND
    assert LexicalDocument(chunk_id=CHUNK_ID, body="body", headings="h").language is Language.UND


def test_positional_construction_stays_compatible() -> None:
    document = LexicalDocument(CHUNK_ID, "body", "Heading > Sub")
    assert (document.body, document.headings, document.language) == (
        "body",
        "Heading > Sub",
        Language.UND,
    )


@pytest.mark.parametrize("language", list(Language))
def test_every_domain_language_is_accepted(language: Language) -> None:
    document = LexicalDocument(CHUNK_ID, "text", language=language)
    assert document.language is language
    assert document == LexicalDocument(CHUNK_ID, "text", language=language)
    assert document != LexicalDocument(CHUNK_ID, "text", language=Language.UND) or (
        language is Language.UND
    )


def test_documents_are_immutable_and_hashable() -> None:
    document = LexicalDocument(CHUNK_ID, "text", language=Language.RU)
    with pytest.raises(FrozenInstanceError):
        document.language = Language.EN  # type: ignore[misc]
    assert len({document, LexicalDocument(CHUNK_ID, "text", language=Language.RU)}) == 1


class _RecordingIndex:
    """Minimal structural implementation, used only to check the port's typing."""

    def __init__(self) -> None:
        self.received: list[LexicalDocument] = []

    def __len__(self) -> int:
        return len(self.received)

    def upsert(self, documents: Sequence[LexicalDocument]) -> None:
        self.received.extend(documents)

    def remove(self, ids: Sequence[ChunkId]) -> None:
        self.received = [d for d in self.received if d.chunk_id not in set(ids)]

    def search(
        self, query: str, k: int, allowed: Set[ChunkId] | None = None
    ) -> Sequence[ScoredChunk]:
        return []


def test_port_carries_language_through_upsert() -> None:
    index: LexicalIndex = _RecordingIndex()
    assert isinstance(index, LexicalIndex)
    index.upsert([LexicalDocument(CHUNK_ID, "текст", language=Language.RU)])
    assert len(index) == 1
    assert index.search("текст", 5) == []
    index.remove([CHUNK_ID])
    assert len(index) == 0
