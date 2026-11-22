"""HTML parser: structure, page furniture, metadata, encodings, nesting."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_engine.domain import BlockKind, Language, ParsedDocument, RagError
from rag_engine.domain.invariants import check_block_provenance
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.parsing.html import HtmlParser
from rag_engine.ports import ParseHints, ParseLimits


def parse(page: str | bytes, parser: HtmlParser | None = None, **hints: object) -> ParsedDocument:
    data = page.encode("utf-8") if isinstance(page, str) else page
    document = (parser or HtmlParser()).parse(data, ParseHints(**hints))  # type: ignore[arg-type]
    check_block_provenance(document.blocks)
    for block in document.blocks:
        assert document.normalized_text[block.char_start : block.char_end] == block.text
    return document


def shape(document: ParsedDocument) -> list[tuple[str, int | None, str]]:
    return [(b.kind.value, b.level, b.text) for b in document.blocks]


PAGE = """<!doctype html>
<html lang="en"><head>
<title>  Field   Guide </title>
<meta name="author" content="A. Writer">
<meta property="article:published_time" content="2024-05-01T10:00:00Z">
<style>body { display: none }</style><script>track()</script>
</head><body>
<nav><a href="/">Home</a> <a href="/docs">Docs</a></nav>
<main>
<h1>Field Guide</h1>
loose text with <b>bold</b> and <a href="x">a link</a>
<p>First paragraph<br>second line &amp; more.</p>
<h2>Setup</h2>
<ul><li>one<ul><li>nested</li></ul></li><li>two</li></ul>
<pre><code class="language-python">def f():
    return 1</code></pre>
<table><thead><tr><th>Name</th><th>Value</th></tr></thead>
<tbody><tr><td>a</td><td>1</td></tr><tr><td>b</td><td></td></tr></tbody></table>
<blockquote><p>quoted</p><p>text</p></blockquote>
<div hidden>secret</div><p aria-hidden="true">decoration</p>
<p style="display: none">invisible</p>
</main>
<aside>related links</aside>
<footer>copyright</footer>
</body></html>"""


def test_a_full_page() -> None:
    document = parse(PAGE)
    assert shape(document) == [
        ("heading", 1, "Field Guide"),
        ("paragraph", None, "loose text with bold and a link"),
        ("paragraph", None, "First paragraph\nsecond line & more."),
        ("heading", 2, "Setup"),
        ("list_item", 1, "one"),
        ("list_item", 2, "nested"),
        ("list_item", 1, "two"),
        ("code", None, "def f():\n    return 1"),
        ("table", None, "Name | Value\na | 1\nb"),
        ("quote", None, "quoted\ntext"),
    ]
    assert document.title == "Field Guide"
    assert document.author == "A. Writer"
    assert document.created == datetime(2024, 5, 1, 10, 0, tzinfo=UTC)
    assert document.language is Language.EN and document.warnings == ()
    assert document.blocks[-1].heading_path == ("Field Guide", "Setup")
    code = next(b for b in document.blocks if b.kind is BlockKind.CODE)
    assert code.attrs == {"lang": "python", "lines": 1}
    table = next(b for b in document.blocks if b.kind is BlockKind.TABLE)
    assert table.attrs == {"rows": 3, "columns": 2}
    for noise in ("Home", "track", "secret", "decoration", "invisible", "related", "copyright"):
        assert noise not in document.normalized_text


def test_main_content_selection() -> None:
    body = "<body><p>outside</p>{}</body>"
    assert parse(body.format("<main><p>inside main</p></main>")).normalized_text == "inside main"
    assert (
        parse(body.format("<div role=main><p>inside role</p></div>")).normalized_text
        == "inside role"
    )
    single = parse(body.format("<article><p>inside article</p></article>"))
    assert single.normalized_text == "inside article"
    many = parse(body.format("<article><p>one</p></article><article><p>two</p></article>"))
    assert many.normalized_text == "outside\n\none\n\ntwo"  # an index of articles: keep all
    assert parse("<p>just a fragment</p>").normalized_text == "just a fragment"


def test_entities_are_decoded_exactly_once() -> None:
    document = parse("<p>Fish &amp; chips &amp;lt;tag&amp;gt; &copy; &#1055;&#x440;</p>")
    assert document.normalized_text == "Fish & chips &lt;tag&gt; © Пр"
    title = parse("<title>A &amp;lt; B</title><p>x</p>")
    assert title.title == "A &lt; B"


def test_inline_flow_and_breaks_and_images() -> None:
    page = "<div>Start <span>in <em>span</em></span><br>next "
    page += '<img src="a.png" alt="alt text"> end</div>'
    document = parse(page)
    assert document.normalized_text == "Start in span\nnext alt text end"


def test_list_items_with_blocks_inside() -> None:
    document = parse("<ol><li><p>first</p><p>continued</p></li><li>second</li></ol>")
    assert shape(document) == [
        ("list_item", 1, "first\ncontinued"),
        ("list_item", 1, "second"),
    ]


def test_definition_lists_and_headings_levels() -> None:
    document = parse("<h2>Terms</h2><dl><dt>Term</dt><dd>Meaning</dd></dl><h4>Deep</h4>")
    assert shape(document) == [
        ("heading", 2, "Terms"),
        ("paragraph", None, "Term"),
        ("paragraph", None, "Meaning"),
        ("heading", 4, "Deep"),
    ]
    assert document.blocks[-1].heading_path == ("Terms", "Deep")


def test_code_blocks_keep_whitespace_and_ignore_cleaners() -> None:
    parser = HtmlParser(compile_cleaners([r"noise"]))
    document = parse("<pre>  a\n\tnoise\n</pre><p>noise</p><p>kept</p>", parser)
    assert shape(document) == [("code", None, "  a\n    noise"), ("paragraph", None, "kept")]


def test_nested_tables_are_flattened_into_their_cell() -> None:
    document = parse(
        "<table><tr><td>outer <table><tr><td>inner</td></tr></table></td></tr></table>"
    )
    assert document.blocks[0].kind is BlockKind.TABLE and "inner" in document.blocks[0].text


def test_metadata_fallbacks_and_problems() -> None:
    assert parse('<meta property="og:title" content="Social"><p>x</p>').title == "Social"
    assert parse("<title>Doc title</title><h1>Heading</h1>").title == "Doc title"
    assert parse("<h1>Only heading</h1>").title == "Only heading"
    assert parse("<p>nothing</p>").title is None
    bad = parse('<meta name="date" content="someday"><p>x</p>')
    assert bad.created is None and "date_ignored" in bad.warnings


def test_legacy_encodings_with_a_hint_and_without() -> None:
    sentence = "Привет, это достаточно длинный русский текст для проверки кодировки."
    page = f"<html><body><p>{sentence}</p></body></html>"
    data = page.encode("cp1251")
    assert parse(data, encoding="cp1251").language is Language.RU
    detected = parse(data)
    assert "Привет" in detected.normalized_text and "decoded_with_replacement" in detected.warnings


@pytest.mark.parametrize(
    "page", ["", "   ", "<html></html>", "<body><nav>only furniture</nav></body>", "<!-- c -->"]
)
def test_pages_without_content_are_empty_documents(page: str) -> None:
    document = parse(page)
    assert document.blocks == () and "empty_document" in document.warnings


def test_malformed_html_still_yields_the_text() -> None:
    document = parse("<div><p>unclosed paragraph<div>next <b>bold</div></i></p>stray </li>text")
    assert "unclosed paragraph" in document.normalized_text and "stray" in document.normalized_text


def test_excessive_nesting_is_an_error() -> None:
    page = "<div>" * 300 + "deep" + "</div>" * 300
    with pytest.raises(RagError) as info:
        HtmlParser().parse(page.encode(), ParseHints(limits=ParseLimits(max_nesting=50)))
    assert info.value.context["reason"] == "too_deep"
    assert parse("<div>" * 20 + "ok" + "</div>" * 20).normalized_text == "ok"


def test_a_large_page_is_parsed() -> None:
    page = (
        "<body>"
        + "".join(f"<h2>Section {i}</h2><p>Text of section {i}.</p>" for i in range(5_000))
        + "</body>"
    )
    assert len(parse(page).blocks) == 10_000


@pytest.mark.parametrize(
    ("mime", "extension", "expected"),
    [
        ("text/html", None, True),
        ("text/plain", "html", False),
        (None, "htm", True),
        (None, "XHTML", True),
        (None, "md", False),
        ("text/markdown", "html", False),
    ],
)
def test_supports(mime: str | None, extension: str | None, expected: bool) -> None:
    assert HtmlParser().supports(mime, extension) is expected
