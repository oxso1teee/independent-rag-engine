"""The chunking strategies and the choice between them."""

from __future__ import annotations

from rag_engine.chunking.fixed import FixedChunker
from rag_engine.chunking.recursive import RecursiveChunker
from rag_engine.chunking.structure import StructureAwareChunker
from rag_engine.domain import BlockKind, ErrorCode, ParsedDocument, RagError
from rag_engine.ports import Chunker, TokenCounter

AUTO = "auto"
STRATEGIES = (AUTO, "structure_aware", "recursive", "fixed")


class ChunkerRegistry:
    """Holds every strategy; ``select`` returns the configured one or, for ``auto``, the one that
    suits the document: headings give ``structure_aware``, several blocks give ``recursive`` and
    a single block of text gives ``fixed``."""

    def __init__(self, counter: TokenCounter, strategy: str = AUTO) -> None:
        if strategy not in STRATEGIES:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID,
                "unknown chunking strategy",
                context={"strategy": strategy, "known": ", ".join(STRATEGIES)},
            )
        self._strategy = strategy
        chunkers: tuple[Chunker, ...] = (
            StructureAwareChunker(counter),
            RecursiveChunker(counter),
            FixedChunker(counter),
        )
        self._chunkers = {chunker.chunker_id: chunker for chunker in chunkers}

    def get(self, chunker_id: str) -> Chunker:
        return self._chunkers[chunker_id]

    def select(self, document: ParsedDocument) -> Chunker:
        if self._strategy != AUTO:
            return self._chunkers[self._strategy]
        if any(block.kind is BlockKind.HEADING for block in document.blocks):
            return self._chunkers["structure_aware"]
        if len(document.blocks) > 1:
            return self._chunkers["recursive"]
        return self._chunkers["fixed"]
