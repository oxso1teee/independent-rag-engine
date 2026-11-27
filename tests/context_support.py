"""In-memory helpers for context tests: a word counter and a content reader over fake documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rag_engine.domain import ChunkId, ChunkRef, DocumentId, DocumentVersionId
from rag_engine.ports import ChunkContent
from tests.factories import make_ids

IDS = make_ids()


class Words:
    """One token per whitespace-separated word."""

    counter_id = "words/1"

    def count(self, text: str) -> int:
        return len(text.split())


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


WHOLE = {"a": words("a", 60), "b": words("b", 60), "c": words("c", 60)}


@dataclass
class Library:
    """An in-memory content reader; chunks of one document are slices of one normalised text."""

    contents: dict[ChunkId, ChunkContent] = field(default_factory=dict)
    documents: dict[str, DocumentId] = field(default_factory=dict)
    versions: dict[str, DocumentVersionId] = field(default_factory=dict)
    next_ordinal: int = 0
    texts: dict[str, str] = field(default_factory=lambda: dict(WHOLE))

    def read(self, ids: Sequence[ChunkId]) -> Mapping[ChunkId, ChunkContent]:
        return {item: self.contents[item] for item in ids if item in self.contents}

    def add(
        self,
        document: str,
        start: int,
        end: int,
        *,
        text: str | None = None,
        level: int = 0,
        parent: ChunkId | None = None,
        headings: tuple[str, ...] = ("Section",),
    ) -> ChunkContent:
        doc_id = self.documents.setdefault(document, IDS.document_id())
        version = self.versions.setdefault(document, IDS.document_version_id())
        chunk_text = text if text is not None else self.texts[document][start:end]
        self.next_ordinal += 1
        chunk_id = ChunkId.derive(version, "ctx", self.next_ordinal)
        content = ChunkContent(
            ref=ChunkRef(chunk_id, doc_id, version),
            text=chunk_text,
            token_count=len(chunk_text.split()),
            level=level,
            parent_id=parent,
            ordinal=self.next_ordinal,
            char_start=start,
            char_end=end,
            heading_path=headings,
            page_start=None,
            page_end=None,
            source_name="notes",
            source_key=f"{document}.md",
            title=None,
        )
        self.contents[chunk_id] = content
        return content
