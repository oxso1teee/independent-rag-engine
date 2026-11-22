"""Turns chunk spans into ``Chunk`` objects with their provenance."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace

from rag_engine.chunking.pack import Packer
from rag_engine.domain import (
    Block,
    BlockKind,
    Chunk,
    ChunkId,
    ChunkKind,
    ChunkPolicy,
    DocumentVersionId,
    ParsedDocument,
    TextHash,
    chunker_fingerprint,
    detect_language,
)
from rag_engine.ports import Chunker, TokenCounter

_KIND = {
    BlockKind.PARAGRAPH: ChunkKind.PARAGRAPH,
    BlockKind.HEADING: ChunkKind.PARAGRAPH,
    BlockKind.LIST_ITEM: ChunkKind.LIST,
    BlockKind.TABLE: ChunkKind.TABLE,
    BlockKind.CODE: ChunkKind.CODE,
    BlockKind.QUOTE: ChunkKind.QUOTE,
}


def policy_fingerprint(
    chunker: Chunker, counter: TokenCounter, policy: ChunkPolicy, normalizer_version: str
) -> str:
    """The fingerprint of a strategy: its identity, every size parameter and the counter."""
    return chunker_fingerprint(
        chunker.chunker_id,
        chunker.version,
        {
            "target": policy.target_tokens,
            "max": policy.max_tokens,
            "min": policy.min_tokens,
            "overlap": policy.overlap_ratio,
            "counter": counter.counter_id,
        },
        normalizer_version,
    )


@dataclass(frozen=True, slots=True)
class Group:
    """A section parent (optional) and the leaf spans inside it."""

    leaves: tuple[tuple[int, int], ...]
    parent: tuple[int, int] | None = None


class ChunkAssembler:
    def __init__(
        self,
        document: ParsedDocument,
        counter: TokenCounter,
        policy: ChunkPolicy,
        version_id: DocumentVersionId,
        fingerprint: str,
    ) -> None:
        self._document = document
        self._counter = counter
        self._policy = policy
        self._version_id = version_id
        self._fingerprint = fingerprint
        self._starts = [block.char_start for block in document.blocks]

    def blocks_in(self, start: int, end: int) -> list[Block]:
        blocks = self._document.blocks
        index = max(bisect_right(self._starts, start) - 1, 0)
        found: list[Block] = []
        while index < len(blocks) and blocks[index].char_start < end:
            if blocks[index].char_end > start:
                found.append(blocks[index])
            index += 1
        return found

    def _make(
        self, start: int, end: int, *, ordinal: int, level: int, parent: ChunkId | None
    ) -> Chunk | None:
        text = self._document.normalized_text[start:end]
        start += len(text) - len(text.lstrip())
        text = text.strip()
        if not text:
            return None
        end = start + len(text)
        blocks = self.blocks_in(start, end)
        kinds = {_KIND[block.kind] for block in blocks}
        kind = kinds.pop() if len(kinds) == 1 and level == 0 else ChunkKind.MIXED
        pages = [block.page for block in blocks if block.page is not None]
        tokens = self._counter.count(text)
        return Chunk(
            id=ChunkId.derive(self._version_id, self._fingerprint, ordinal),
            document_version_id=self._version_id,
            ordinal=ordinal,
            level=level,
            parent_id=parent,
            kind=kind,
            text=text,
            text_hash=TextHash.of_text(text),
            token_count=tokens,
            char_start=start,
            char_end=end,
            heading_path=blocks[0].heading_path if blocks else (),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            language=detect_language(text),
            indexed=level == 0,
            oversize=level == 0 and tokens > self._policy.max_tokens,
        )

    def assemble(self, groups: list[Group], packer: Packer) -> list[Chunk]:
        """Chunks in document order (a parent, then its leaves); repeated leaf texts after the
        first are marked as duplicates and left out of the index."""
        chunks: list[Chunk] = []
        seen: set[TextHash] = set()
        for group in groups:
            parent_id: ChunkId | None = None
            if group.parent is not None:
                parent = self._make(*group.parent, ordinal=len(chunks), level=1, parent=None)
                if parent is not None:
                    chunks.append(parent)
                    parent_id = parent.id
            for span in group.leaves:
                for start, end in packer.fit(*span):
                    leaf = self._make(start, end, ordinal=len(chunks), level=0, parent=parent_id)
                    if leaf is None:
                        continue
                    if leaf.text_hash in seen:
                        leaf = replace(leaf, is_duplicate=True, indexed=False)
                    seen.add(leaf.text_hash)
                    chunks.append(leaf)
        return chunks
