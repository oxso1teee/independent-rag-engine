"""HTML parser: the readable content of a page as headings, paragraphs, lists, tables, code.

Scripts, styles, hidden elements and page furniture (navigation, footers, sidebars) are
dropped; the main content is ``<main>``, else a single ``<article>``, else ``<body>``. Entities
are decoded once, by the HTML parser.
"""

from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser, LexborNode

from rag_engine.domain import BlockKind, ParsedDocument, ParseReason, parse_failed
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.parsing.cleaners import CleanerSet, normalizer_version
from rag_engine.parsing.common import check_input_size, clean_metadata, decode_text
from rag_engine.parsing.frontmatter import parse_date
from rag_engine.ports import ParseHints

_EXTENSIONS = frozenset({"html", "htm", "xhtml"})
_REMOVED = (
    "script, style, noscript, template, svg, canvas, iframe, object, embed, input, button, select, "
    "textarea, nav, footer, aside, [hidden], [aria-hidden=true], "
    "[role=navigation], [role=contentinfo], [role=complementary], [role=banner], [role=search], "
    '[style*="display:none"], [style*="display: none"]'
)
_INLINE = frozenset(
    {
        *("a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "del", "dfn", "em", "font"),
        *("i", "ins", "kbd", "label", "mark", "q", "s", "samp", "small", "span", "strike"),
        *("strong", "sub", "sup", "time", "tt", "u", "var", "wbr", "img"),
    }
)
_BLOCK = frozenset(
    {
        *("address", "article", "blockquote", "body", "details", "dialog", "dd", "div", "dl"),
        *("dt", "fieldset", "figcaption", "figure", "form", "h1", "h2", "h3", "h4", "h5", "h6"),
        *("header", "hgroup", "hr", "li", "main", "ol", "p", "pre", "section", "summary"),
        *("table", "ul", "caption", "tbody", "thead", "tfoot", "tr", "td", "th"),
    }
)
_PARAGRAPH_TAGS = frozenset({"p", "dt", "dd", "figcaption", "summary", "caption", "address"})


def _tag(node: LexborNode) -> str:
    return node.tag or ""


def _inline_of(node: LexborNode) -> str:
    """Text a single inline element contributes (an image gives its alternative text)."""
    if _tag(node) == "img":
        return node.attributes.get("alt") or ""
    return _inline(node)


def _inline(node: LexborNode) -> str:
    """Text of an inline run: ``<br>`` becomes a line break, images contribute their alt text."""
    parts: list[str] = []
    for child in node.iter(include_text=True):
        tag = _tag(child)
        if tag == "-text":
            parts.append(child.text(deep=False))
        elif tag == "br":
            parts.append("\n")
        elif tag == "img":
            parts.append(child.attributes.get("alt") or "")
        elif not tag.startswith("-"):
            parts.append(_inline(child))
    return "".join(parts)


class HtmlParser:
    parser_id = "html"
    version = "1"

    def __init__(self, cleaners: CleanerSet | None = None) -> None:
        self._cleaners = cleaners or CleanerSet()

    @property
    def normalizer_version(self) -> str:
        return normalizer_version(self._cleaners)

    def supports(self, mime: str | None, extension: str | None) -> bool:
        if mime is not None:
            return mime == "text/html"
        return (extension or "").lower() in _EXTENSIONS

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        check_input_size(data, hints)
        text, warnings = decode_text(data, hints)
        tree = LexborHTMLParser(text)
        meta = self._metadata(tree)
        for node in tree.css(_REMOVED):
            node.decompose()
        assembler = TextAssembler(hints.limits, self._cleaners)
        root = self._root(tree)
        if root is not None:
            _Walker(assembler, hints.limits.max_nesting).walk(root, 0)
        built = assembler.build()
        first_h1 = next(
            (b.text for b in built.blocks if b.kind is BlockKind.HEADING and b.level == 1), None
        )
        created = parse_date(meta["created"]) if meta.get("created") else None
        if meta.get("created") and created is None:
            warnings.append("date_ignored")
        if not built.blocks:
            warnings.append("empty_document")
        return assembler.parsed(
            title=clean_metadata(meta.get("title")) or first_h1,
            author=clean_metadata(meta.get("author")),
            created=created,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _root(tree: LexborHTMLParser) -> LexborNode | None:
        main = tree.css_first("main, [role=main]")
        if main is not None:
            return main
        articles = tree.css("article")
        if len(articles) == 1:
            return articles[0]
        return tree.body

    @staticmethod
    def _metadata(tree: LexborHTMLParser) -> dict[str, str]:
        meta: dict[str, str] = {}
        title = tree.css_first("title")
        if title is not None and title.text().strip():
            meta["title"] = " ".join(title.text().split())
        for node in tree.css("meta"):
            attrs = node.attributes
            key = (attrs.get("name") or attrs.get("property") or "").lower()
            value = (attrs.get("content") or "").strip()
            if not value:
                continue
            if key == "og:title":
                meta.setdefault("title", value)
            elif key in ("author", "article:author"):
                meta.setdefault("author", value)
            elif key in ("article:published_time", "date", "dc.date"):
                meta.setdefault("created", value)
        return meta


class _Walker:
    def __init__(self, out: TextAssembler, max_nesting: int) -> None:
        self._out = out
        self._max = max_nesting

    def walk(self, node: LexborNode, depth: int) -> None:
        if depth > self._max:
            raise parse_failed(ParseReason.TOO_DEEP, "the document is nested too deeply")
        run: list[str] = []

        def flush() -> None:
            text = "".join(run)
            run.clear()
            if text.strip():
                self._out.add(BlockKind.PARAGRAPH, text)

        for child in node.iter(include_text=True):
            tag = _tag(child)
            if tag == "-text":
                run.append(child.text(deep=False))
            elif tag.startswith("-"):
                continue  # comments and the like
            elif tag == "br":
                run.append("\n")
            elif tag in _INLINE or tag not in _BLOCK:
                run.append(_inline_of(child))
            else:
                flush()
                self._block(child, depth)
        flush()

    def _block(self, node: LexborNode, depth: int) -> None:
        tag = _tag(node)
        if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self._out.add(BlockKind.HEADING, _inline(node), level=int(tag[1]))
        elif tag in _PARAGRAPH_TAGS:
            self._out.add(BlockKind.PARAGRAPH, _inline(node))
        elif tag == "pre":
            self._code(node)
        elif tag in ("ul", "ol"):
            self._list(node, 1, depth)
        elif tag == "blockquote":
            self._out.add(BlockKind.QUOTE, node.text(deep=True, separator="\n", strip=True))
        elif tag == "table":
            self._table(node)
        elif tag == "hr":
            return
        else:
            self.walk(node, depth + 1)

    def _code(self, node: LexborNode) -> None:
        code = node.css_first("code")
        language = ""
        if code is not None:
            for name in (code.attributes.get("class") or "").split():
                if name.startswith(("language-", "lang-")):
                    language = name.split("-", 1)[1]
        text = node.text(deep=True, separator="")
        attrs: dict[str, str | int | bool] = {"lines": text.count("\n")}
        if language:
            attrs["lang"] = language
        self._out.add(BlockKind.CODE, text, attrs=attrs)

    def _list(self, node: LexborNode, level: int, depth: int) -> None:
        if depth > self._max:
            raise parse_failed(ParseReason.TOO_DEEP, "the document is nested too deeply")
        for item in node.iter():
            if _tag(item) != "li":
                continue
            own: list[str] = []
            nested: list[LexborNode] = []
            for child in item.iter(include_text=True):
                if _tag(child) in ("ul", "ol"):
                    nested.append(child)
                elif _tag(child) == "-text":
                    own.append(child.text(deep=False))
                elif not _tag(child).startswith("-"):
                    prefix = "\n" if _tag(child) in _BLOCK else ""
                    own.append(prefix + _inline_of(child))
            self._out.add(BlockKind.LIST_ITEM, "".join(own), level=level)
            for sub in nested:
                self._list(sub, level + 1, depth + 1)

    def _table(self, node: LexborNode) -> None:
        rows: list[str] = []
        columns = 0
        for row in node.css("tr"):
            cells = [" ".join(_inline(c).split()) for c in row.css("th, td")]
            columns = max(columns, len(cells))
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                rows.append(" | ".join(cells))
        self._out.add(
            BlockKind.TABLE, "\n".join(rows), attrs={"rows": len(rows), "columns": columns}
        )
