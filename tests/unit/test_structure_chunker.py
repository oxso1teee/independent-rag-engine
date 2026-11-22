"""Structure-aware chunking: sections, parent chunks, heading placement and tiny-section merging."""

from __future__ import annotations

from rag_engine.chunking.structure import StructureAwareChunker, split_sections
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import BlockKind, Chunk, ChunkKind, ChunkPolicy, ParsedDocument
from tests.chunking_checks import SMALL_POLICY, check_chunks, document_of, prose
from tests.factories import make_ids

COUNTER = HeuristicTokenCounter()
CHUNKER = StructureAwareChunker(COUNTER)
VERSION = make_ids().document_version_id()
P, H, T, C = BlockKind.PARAGRAPH, BlockKind.HEADING, BlockKind.TABLE, BlockKind.CODE
SHORT = "A short note."


def run(document: ParsedDocument, policy: ChunkPolicy = SMALL_POLICY) -> list[Chunk]:
    chunks = list(CHUNKER.chunk(document, policy, version_id=VERSION, normalizer_version="norm/1"))
    check_chunks(document, chunks, policy, COUNTER, CHUNKER.fingerprint(policy, "norm/1"))
    return chunks


def leaves(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if c.level == 0]


def test_sections_start_at_every_heading_and_text_before_the_first_is_its_own() -> None:
    document = document_of((P, "intro"), (H, "A", 1), (P, "a"), (H, "B", 2), (H, "C", 2), (P, "c"))
    sections = split_sections(document.blocks)
    assert [[b.text for b in section] for section in sections] == [
        ["intro"],
        ["A", "a"],
        ["B"],
        ["C", "c"],
    ]
    assert split_sections([]) == []


def test_a_section_with_several_leaves_gets_a_parent_that_holds_the_heading() -> None:
    document = document_of((H, "Guide", 1), (P, prose(3)), (P, prose(3, start=10)))
    chunks = run(document)
    (parent,) = [c for c in chunks if c.level == 1]
    children = leaves(chunks)
    assert len(children) >= 2 and chunks[0] is parent
    assert parent.text.startswith("Guide") and parent.kind is ChunkKind.MIXED
    assert not parent.indexed and parent.parent_id is None
    assert all(c.parent_id == parent.id for c in children)
    assert not any(
        "Guide" in c.text.split("\n")[0] for c in children
    )  # the heading is in the parent
    assert all(c.heading_path == ("Guide",) for c in children)
    assert parent.char_start == 0 and parent.char_end == len(document.normalized_text)


def test_a_section_with_one_leaf_keeps_its_heading_and_has_no_parent() -> None:
    chunks = run(document_of((H, "Notes", 1), (P, SHORT)))
    (only,) = chunks
    assert only.level == 0 and only.parent_id is None
    assert only.text == "Notes\n\nA short note." and only.heading_path == ("Notes",)


def test_a_heading_never_pushes_a_leaf_above_the_maximum() -> None:
    for words in range(1, 60):
        sentence = " ".join(f"w{i % 10}" for i in range(words)) + "."
        chunks = run(document_of((H, "Title", 1), (P, sentence)))
        assert all(c.token_count <= SMALL_POLICY.max_tokens for c in leaves(chunks))
        assert any(c.text.startswith("Title") for c in chunks)


def test_text_before_the_first_heading_has_no_parent() -> None:
    chunks = run(document_of((P, prose(6)), (H, "After", 1), (P, SHORT)))
    assert len(leaves(chunks)) >= 3 and not [c for c in chunks if c.level == 1]
    assert chunks[-1].text.startswith("After") and chunks[-1].heading_path == ("After",)


def test_each_section_is_chunked_on_its_own_without_overlap_between_sections() -> None:
    document = document_of((H, "One", 1), (P, prose(4)), (H, "Two", 1), (P, prose(4, start=50)))
    chunks = run(document)
    parents = [c for c in chunks if c.level == 1]
    assert len(parents) == 2 and parents[0].char_end <= parents[1].char_start
    first = [c for c in leaves(chunks) if c.parent_id == parents[0].id]
    second = [c for c in leaves(chunks) if c.parent_id == parents[1].id]
    assert first and second and first[-1].char_end <= second[0].char_start


def test_tiny_neighbouring_sections_are_merged_when_they_fit_together() -> None:
    document = document_of(
        (H, "A", 1), (P, SHORT), (H, "B", 2), (P, SHORT), (H, "C", 2), (P, SHORT)
    )
    (chunk,) = run(document)
    assert chunk.text.startswith("A\n\n") and "B" in chunk.text and "C" in chunk.text
    assert chunk.heading_path == ("A",) and chunk.level == 0 and chunk.parent_id is None


def test_a_higher_level_heading_is_never_merged_into_the_section_before_it() -> None:
    document = document_of((H, "Deep", 3), (P, SHORT), (H, "Top", 1), (P, SHORT))
    chunks = run(document)
    assert [c.text.split("\n")[0] for c in chunks] == ["Deep", "Top"]


def test_sections_that_do_not_fit_together_stay_apart() -> None:
    policy = ChunkPolicy(target_tokens=10, max_tokens=20, min_tokens=8, overlap_ratio=0.0)
    document = document_of(
        (H, "A", 1), (P, SHORT), (H, "B", 1), (P, SHORT), (H, "C", 1), (P, SHORT)
    )
    chunks = run(document, policy)
    assert [c.text.split("\n")[0] for c in chunks] == ["A", "B", "C"]


def test_a_heading_without_text_joins_the_section_it_introduces() -> None:
    document = document_of(
        (H, "Manual", 1), (H, "Install", 2), (P, prose(4)), (H, "Run", 2), (P, SHORT)
    )
    chunks = run(document)
    (parent, *rest) = chunks
    assert parent.level == 1 and parent.text.startswith("Manual\n\nInstall")
    assert rest[0].text.startswith("Install") and rest[0].heading_path == ("Manual", "Install")
    assert chunks[-1].text == "Run\n\nA short note." and chunks[-1].heading_path == (
        "Manual",
        "Run",
    )
    assert not any(c.text == "Manual" for c in chunks)


def test_big_sections_are_not_merged_even_when_a_neighbour_is_tiny() -> None:
    document = document_of((H, "Big", 1), (P, prose(5)), (H, "Tiny", 2), (P, SHORT))
    chunks = run(document)
    assert chunks[-1].text.startswith("Tiny") and chunks[-1].level == 0


def test_a_table_that_fits_is_never_cut_inside_a_row() -> None:
    table = "\n".join(f"row{i} | value{i}" for i in range(4))
    chunks = run(document_of((H, "Data", 1), (P, prose(2)), (T, table)))
    assert any(table in c.text for c in leaves(chunks))


def test_a_huge_code_block_is_cut_between_lines() -> None:
    code = "\n".join(f"step_{i % 10} = run({i % 10})" for i in range(120))
    chunks = run(document_of((H, "Script", 1), (C, code)))
    pieces = leaves(chunks)
    assert len(pieces) > 2
    assert all(p.text.splitlines()[-1].startswith("step_") for p in pieces[1:])
    assert {p.kind for p in pieces[1:]} <= {ChunkKind.CODE, ChunkKind.MIXED}


def test_nested_headings_give_leaves_their_full_heading_path() -> None:
    document = document_of((H, "Manual", 1), (P, SHORT * 20), (H, "Install", 2), (P, prose(4)))
    paths = {c.heading_path for c in leaves(run(document))}
    assert paths == {("Manual",), ("Manual", "Install")}


def test_a_document_with_only_headings_is_chunked() -> None:
    (chunk,) = run(document_of((H, "One", 1), (H, "Two", 2)))
    assert chunk.text == "One\n\nTwo" and chunk.level == 0
