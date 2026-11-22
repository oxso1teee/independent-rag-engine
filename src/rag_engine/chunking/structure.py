"""Structure-aware strategy: sections by headings, parent chunks and leaf chunks.

A *section* starts at a heading (of any level) and ends at the next heading. Neighbouring tiny
sections are merged into one group while the result stays within the target size, and a heading
without text of its own is joined to the section it introduces. Each group is
packed into leaf chunks; a group that starts with a heading and has two or more leaves also gets
a *parent* chunk (level 1) holding the whole group, so that a retrieved leaf can be shown in its
context. The heading of a group with a parent belongs to the parent only; a group with one leaf
keeps its heading in that leaf. Tables and code are never cut inside a line unless they exceed
``max_tokens``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.chunking.assemble import ChunkAssembler, Group, policy_fingerprint
from rag_engine.chunking.pack import Packer
from rag_engine.domain import (
    Block,
    BlockKind,
    Chunk,
    ChunkPolicy,
    DocumentVersionId,
    ParsedDocument,
)
from rag_engine.ports import TokenCounter

Span = tuple[int, int]


def split_sections(blocks: Sequence[Block]) -> list[list[Block]]:
    """Blocks cut before every heading; text before the first heading is its own section."""
    sections: list[list[Block]] = []
    for block in blocks:
        if block.kind is BlockKind.HEADING or not sections:
            sections.append([])
        sections[-1].append(block)
    return sections


def _level(section: Sequence[Block]) -> int:
    first = section[0]
    return (first.level or 0) if first.kind is BlockKind.HEADING else 0


class StructureAwareChunker:
    chunker_id = "structure_aware"
    version = "1"

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
        groups = [
            group
            for unit in self._merge_tiny(packer, split_sections(document.blocks))
            if (group := self._group(packer, unit)) is not None
        ]
        assembler = ChunkAssembler(
            document,
            self._counter,
            policy,
            version_id,
            self.fingerprint(policy, normalizer_version),
        )
        return assembler.assemble(groups, packer)

    # -- merging -----------------------------------------------------------------------

    @staticmethod
    def _merge_tiny(packer: Packer, sections: list[list[Block]]) -> list[list[Block]]:
        """Join a section to the group before it unless it starts a higher-level heading, when the
        group has no text of its own (only headings), or when one of the two is smaller than
        ``min_tokens`` and together they fit the target size."""
        units: list[list[Block]] = []
        group_tokens = 0
        for section in sections:
            tokens = packer.count(section[0].char_start, section[-1].char_end)
            if units and _level(section) >= _level(units[-1]):
                unit = units[-1]
                combined = packer.count(unit[0].char_start, section[-1].char_end)
                bare = all(block.kind is BlockKind.HEADING for block in unit)
                small = group_tokens < packer.min or tokens < packer.min
                if bare or (small and combined <= packer.target):
                    unit.extend(section)
                    group_tokens = combined
                    continue
            units.append(list(section))
            group_tokens = tokens
        return units

    # -- grouping ----------------------------------------------------------------------

    @staticmethod
    def _leaves(packer: Packer, blocks: Sequence[Block]) -> list[Span]:
        pieces = [piece for block in blocks for piece in packer.pieces_of_block(block)]
        return [fitted for span in packer.pack(pieces) for fitted in packer.fit(*span)]

    def _group(self, packer: Packer, unit: list[Block]) -> Group | None:
        first = unit[0]
        start, end = first.char_start, unit[-1].char_end
        if first.kind is not BlockKind.HEADING:
            leaves = self._leaves(packer, unit)
        else:
            leaves = self._leaves(packer, unit[1:])
            if len(leaves) <= 1:  # the heading stays with its text
                whole = (start, leaves[0][1] if leaves else first.char_end)
                leaves = [whole] if packer.count(*whole) <= packer.max else []
                if not leaves:
                    leaves = self._leaves(packer, unit)
        if not leaves:
            return None
        has_heading = first.kind is BlockKind.HEADING
        parent = (start, end) if has_heading and len(leaves) >= 2 else None
        return Group(leaves=tuple(leaves), parent=parent)
