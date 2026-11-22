"""Domain additions for parsing: block attributes, document dates, language, provenance and
duplicate invariants."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import (
    Block,
    BlockKind,
    ContentHash,
    Document,
    DocumentStatus,
    ErrorCode,
    Language,
    ParsedDocument,
    RagError,
    detect_language,
)
from rag_engine.domain.invariants import check_block_provenance, check_duplicate_graph
from tests.factories import T0, make_document, make_ids

IDS = make_ids()
SOURCE = IDS.source_id()
SAME = ContentHash.of_bytes(b"same")
OTHER = ContentHash.of_bytes(b"other")


def raises_integrity(call: object) -> None:
    with pytest.raises(RagError) as info:
        call()  # type: ignore[operator]
    assert info.value.code is ErrorCode.E_INTEGRITY


# ------------------------------------------------------------------ language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", Language.UND),
        ("12345 67890 !!!", Language.UND),
        ("ok", Language.UND),  # too few letters
        ("Привет, это обычный русский текст.", Language.RU),
        ("This is an ordinary English sentence.", Language.EN),
        ("Ceci est une phrase française avec des accents é è à.", Language.EN),  # Latin script
        ("Größe und Straße", Language.EN),
        ("Это предложение с английскими словами: retrieval and chunking pipeline.", Language.MIXED),
        ("Привет hello мир world привет hello", Language.MIXED),
        ("日本語のテキストだけの文章です", Language.UND),  # another script
        ("Привет мир " + "漢字" * 200, Language.UND),  # other scripts dominate
        ("def f(x): return x  # a comment", Language.EN),
        ("def f(x): return x  # комментарий", Language.MIXED),
        ("Ёж и ёлка — зелёные", Language.RU),
        ("naïve café résumé", Language.EN),
    ],
)
def test_language_table(text: str, expected: Language) -> None:
    assert detect_language(text) is expected


def test_language_threshold_boundary() -> None:
    # 85 % of one script decides; one letter fewer makes the text mixed
    assert detect_language("а" * 17 + "abc") is Language.RU
    assert detect_language("а" * 17 + "abcd") is Language.MIXED
    assert detect_language("a" * 17 + "абв") is Language.EN
    assert detect_language("a" * 17 + "абвг") is Language.MIXED


def test_long_text_is_sampled_not_scanned_completely() -> None:
    text = ("Русский текст. " * 20_000) + ("english text. " * 5_000)
    assert len(text) > 200_000
    assert detect_language(text) in {Language.RU, Language.MIXED}
    assert detect_language(text) is detect_language(text)


@given(st.text(max_size=400))
def test_language_is_deterministic_and_total(text: str) -> None:
    result = detect_language(text)
    assert isinstance(result, Language)
    assert detect_language(text) is result


@given(st.text(alphabet=st.characters(min_codepoint=0x410, max_codepoint=0x44F), min_size=5))
def test_pure_cyrillic_is_russian(text: str) -> None:
    assert detect_language(text) is Language.RU


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=5))
def test_pure_latin_is_english(text: str) -> None:
    assert detect_language(text) is Language.EN


# ------------------------------------------------------------- block attrs


def heading(text: str, start: int, level: int, path: tuple[str, ...]) -> Block:
    return Block(
        kind=BlockKind.HEADING,
        text=text,
        char_start=start,
        char_end=start + len(text),
        level=level,
        heading_path=path,
    )


def paragraph(text: str, start: int, path: tuple[str, ...], page: int | None = None) -> Block:
    return Block(
        kind=BlockKind.PARAGRAPH,
        text=text,
        char_start=start,
        char_end=start + len(text),
        heading_path=path,
        page=page,
    )


def test_block_attributes_are_frozen_scalars() -> None:
    block = Block(
        kind=BlockKind.CODE,
        text="x = 1",
        char_start=0,
        char_end=5,
        attrs={"lang": "python", "lines": 1, "numbered": False, "ratio": 0.5},
    )
    assert block.attrs["lang"] == "python"
    with pytest.raises(TypeError):
        block.attrs["lang"] = "c"  # type: ignore[index]
    source = {"k": "v"}
    copy = Block(kind=BlockKind.PARAGRAPH, text="t", char_start=0, char_end=1, attrs=source)
    source["k"] = "changed"
    assert copy.attrs["k"] == "v"  # the block keeps its own copy


@pytest.mark.parametrize("bad", [{"k": [1]}, {"k": None}, {"k": {"a": 1}}, {1: "v"}])
def test_block_attributes_must_be_scalars(bad: dict[object, object]) -> None:
    raises_integrity(
        lambda: Block(
            kind=BlockKind.PARAGRAPH,
            text="t",
            char_start=0,
            char_end=1,
            attrs=bad,  # type: ignore[arg-type]
        )
    )


def test_document_creation_date_must_be_utc() -> None:
    ParsedDocument(normalized_text="a", created=T0)
    ParsedDocument(normalized_text="a", created=None)
    raises_integrity(lambda: ParsedDocument(normalized_text="a", created=datetime(2026, 1, 1)))  # noqa: DTZ001
    raises_integrity(
        lambda: ParsedDocument(
            normalized_text="a", created=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=3)))
        )
    )


# ---------------------------------------------------------------- provenance


def test_heading_paths_follow_the_headings_above() -> None:
    blocks = (
        heading("Guide", 0, 1, ("Guide",)),
        paragraph("intro", 6, ("Guide",)),
        heading("Setup", 12, 2, ("Guide", "Setup")),
        paragraph("steps", 18, ("Guide", "Setup")),
        heading("Usage", 24, 2, ("Guide", "Usage")),  # a sibling closes "Setup"
        paragraph("body", 30, ("Guide", "Usage")),
        heading("Appendix", 36, 1, ("Appendix",)),  # a top-level heading closes everything
        paragraph("tail", 45, ("Appendix",)),
    )
    check_block_provenance(blocks)
    check_block_provenance(())
    check_block_provenance((paragraph("no headings", 0, ()),))


@pytest.mark.parametrize(
    "broken",
    [
        (heading("A", 0, 1, ("A",)), paragraph("x", 2, ())),  # path lost
        (heading("A", 0, 1, ()), paragraph("x", 2, ("A",))),  # heading omits itself
        (heading("A", 0, 1, ("A",)), heading("B", 2, 2, ("B",))),  # ancestor dropped
        (
            heading("A", 0, 1, ("A",)),
            heading("B", 2, 2, ("A", "B")),
            heading("C", 4, 2, ("A", "B", "C")),  # B was not closed by its sibling
        ),
        (paragraph("x", 0, ("ghost",)),),
    ],
)
def test_wrong_heading_paths_are_rejected(broken: tuple[Block, ...]) -> None:
    raises_integrity(lambda: check_block_provenance(broken))


def test_pages_never_decrease() -> None:
    ok = (paragraph("a", 0, (), page=1), paragraph("b", 2, (), page=1), paragraph("c", 4, (), 3))
    check_block_provenance(ok)
    bad = (paragraph("a", 0, (), page=2), paragraph("b", 2, (), page=1))
    raises_integrity(lambda: check_block_provenance(bad))


# ---------------------------------------------------------------- duplicates


def make(
    source_key: str, *, status: DocumentStatus = DocumentStatus.ACTIVE, **extra: object
) -> Document:
    document = make_document(
        document_id=IDS.document_id(),
        source_id=SOURCE,
        version_id=None,
        key=source_key,
    )
    return replace(document, status=status, **extra)  # type: ignore[arg-type]


def test_valid_duplicate_graph() -> None:
    original = make("a.md")
    copy = make("b.md", duplicate_of=original.id)
    third = make("c.md", duplicate_of=original.id)
    lone = make("d.md")
    hashes = {original.id: SAME, copy.id: SAME, third.id: SAME, lone.id: OTHER}
    check_duplicate_graph([original, copy, third, lone], hashes)
    check_duplicate_graph([], {})


def test_duplicate_graph_rejects_broken_links() -> None:
    original = make("a.md")
    copy = make("b.md", duplicate_of=original.id)
    hashes = {original.id: SAME, copy.id: SAME}

    ghost = make("g.md", duplicate_of=IDS.document_id())
    raises_integrity(lambda: check_duplicate_graph([ghost], {ghost.id: SAME}))
    raises_integrity(lambda: check_duplicate_graph([copy], hashes))  # original missing

    chained = make("c.md", duplicate_of=copy.id)
    raises_integrity(
        lambda: check_duplicate_graph([original, copy, chained], {**hashes, chained.id: SAME})
    )
    raises_integrity(  # different content
        lambda: check_duplicate_graph([original, copy], {original.id: SAME, copy.id: OTHER})
    )
    raises_integrity(lambda: check_duplicate_graph([original, copy], {original.id: SAME}))
    gone = make("a.md", status=DocumentStatus.MISSING, missing_since=T0)
    gone_copy = make("z.md", duplicate_of=gone.id)
    raises_integrity(
        lambda: check_duplicate_graph([gone, gone_copy], {gone.id: SAME, gone_copy.id: SAME})
    )
    other_source = replace(make("o.md"), source_id=IDS.source_id())
    cross = make("x.md", duplicate_of=other_source.id)
    raises_integrity(
        lambda: check_duplicate_graph(
            [other_source, cross], {other_source.id: SAME, cross.id: SAME}
        )
    )
