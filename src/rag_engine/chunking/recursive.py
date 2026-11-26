"""Recursive strategy: paragraph, then sentence, then word, then character boundaries.

Consecutive pieces are packed up to the target size. Paragraph boundaries anchor groups so
an edit does not shift chunk boundaries through the rest of a document. There are no parents.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.chunking.assemble import ChunkAssembler, Group, policy_fingerprint
from rag_engine.chunking.pack import Packer, Piece
from rag_engine.domain import (
    Chunk,
    ChunkPolicy,
    DocumentVersionId,
    ParsedDocument,
)
from rag_engine.ports import TokenCounter


class RecursiveChunker:
    chunker_id = "recursive"
    version = "2"

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def fingerprint(self, policy: ChunkPolicy, normalizer_version: str) -> str:
        return policy_fingerprint(self, self._counter, policy, normalizer_version)

    def chunk(
        self,
        document: ParsedDocument,
        policy: ChunkPolicy,
        *,
        version_id: DocumentVersionId,
        normalizer_version: str,
    ) -> Sequence[Chunk]:
        if not document.blocks:
            return []
        packer = Packer(document.normalized_text, self._counter, policy)
        pieces: list[Piece] = []
        groups: list[Group] = []
        small_blocks = 0
        for block in document.blocks:
            pieces.extend(packer.pieces_of_block(block))
            if packer.count(block.char_start, block.char_end) >= packer.min or small_blocks == 3:
                groups.append(Group(leaves=tuple(packer.pack(pieces))))
                pieces.clear()
                small_blocks = 0
            else:
                small_blocks += 1
        if pieces:
            groups.append(Group(leaves=tuple(packer.pack(pieces))))
        assembler = ChunkAssembler(
            document,
            self._counter,
            policy,
            version_id,
            self.fingerprint(policy, normalizer_version),
        )
        return assembler.assemble(groups, packer)
