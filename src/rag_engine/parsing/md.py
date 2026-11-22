"""Markdown parser (CommonMark with tables).

Headings, paragraphs, list items (with nesting level), block quotes, fenced and indented code,
tables (one row per line, cells joined by `` | ``) and raw HTML blocks (reduced to their text).
Inline markup is dropped and its text kept; entities and escapes are decoded once by the
Markdown parser itself.
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

from rag_engine.domain import BlockKind, ParsedDocument, ParseReason, parse_failed
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.parsing.cleaners import CleanerSet, normalizer_version
from rag_engine.parsing.common import check_input_size, clean_metadata, decode_text
from rag_engine.parsing.frontmatter import parse_date, split_front_matter
from rag_engine.parsing.htmltext import fragment_text
from rag_engine.ports import ParseHints

_EXTENSIONS = frozenset({"md", "markdown"})


def _inline_text(token: Token | None) -> str:
    if token is None or not token.children:
        return token.content if token is not None else ""
    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "text_special", "code_inline"):
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append(" ")
        elif child.type == "hardbreak":
            parts.append("\n")
        elif child.type == "image":
            parts.append(child.content)  # the alternative text
    return "".join(parts)


class MarkdownParser:
    parser_id = "md"
    version = "1"

    def __init__(self, cleaners: CleanerSet | None = None) -> None:
        self._cleaners = cleaners or CleanerSet()
        self._md = MarkdownIt("commonmark").enable("table")

    @property
    def normalizer_version(self) -> str:
        return normalizer_version(self._cleaners)

    def supports(self, mime: str | None, extension: str | None) -> bool:
        if mime is not None:
            return mime == "text/markdown"
        return (extension or "").lower() in _EXTENSIONS

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        check_input_size(data, hints)
        raw, warnings = decode_text(data, hints)
        meta, body, front_warnings = split_front_matter(
            raw.replace("\r\n", "\n").replace("\r", "\n")
        )
        warnings = [*warnings, *front_warnings]
        assembler = TextAssembler(hints.limits, self._cleaners)
        tokens = self._md.parse(body)
        if any(token.level > hints.limits.max_nesting for token in tokens):
            raise parse_failed(ParseReason.TOO_DEEP, "the document is nested too deeply")
        _Walker(assembler).walk(tokens)
        title = clean_metadata(meta.get("title"))
        first_h1 = next(
            (
                b.text
                for b in assembler.build().blocks
                if b.kind is BlockKind.HEADING and b.level == 1
            ),
            None,
        )
        created = parse_date(meta["date"]) if "date" in meta else None
        if "date" in meta and created is None:
            warnings.append("front_matter_date_ignored")
        document = assembler.parsed(
            title=title or first_h1,
            author=clean_metadata(meta.get("author")),
            created=created,
            warnings=tuple(warnings),
        )
        if not document.blocks:
            return assembler.parsed(
                title=title,
                author=clean_metadata(meta.get("author")),
                warnings=(*warnings, "empty_document"),
            )
        return document


class _Walker:
    """Turns the flat token stream into blocks."""

    def __init__(self, assembler: TextAssembler) -> None:
        self._out = assembler
        self._list_stack: list[bool] = []  # one flag per open list: ordered or not
        self._item: list[str] | None = None
        self._item_level = 0
        self._quote_depth = 0
        self._quote_lines: list[str] = []
        self._row: list[str] | None = None
        self._rows: list[str] = []
        self._columns = 0

    def walk(self, tokens: list[Token]) -> None:
        index = 0
        while index < len(tokens):
            token = tokens[index]
            nxt = tokens[index + 1] if index + 1 < len(tokens) else None
            handler = getattr(self, f"_on_{token.type}", None)
            if handler is not None:
                handler(token, nxt)
            index += 1

    # -- text containers ----------------------------------------------------------

    def _on_heading_open(self, token: Token, nxt: Token | None) -> None:
        level = int(token.tag[1])
        text = _inline_text(nxt)
        if self._quote_depth:
            self._quote_lines.append(text)
        else:
            self._out.add(BlockKind.HEADING, text, level=level)

    def _on_paragraph_open(self, token: Token, nxt: Token | None) -> None:
        del token
        text = _inline_text(nxt)
        if self._item is not None:
            self._item.append(text)
        elif self._quote_depth:
            self._quote_lines.append(text)
        else:
            self._out.add(BlockKind.PARAGRAPH, text)

    def _on_list_item_open(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._flush_item()
        self._item = []
        self._item_level = len(self._list_stack)

    def _on_list_item_close(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._flush_item()

    def _on_bullet_list_open(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._flush_item()
        self._list_stack.append(False)

    _on_ordered_list_open = _on_bullet_list_open

    def _on_bullet_list_close(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._list_stack.pop()

    _on_ordered_list_close = _on_bullet_list_close

    def _flush_item(self) -> None:
        if self._item is None:
            return
        text = "\n".join(part for part in self._item if part)
        level, self._item = self._item_level, None
        if self._quote_depth:
            self._quote_lines.append(text)
        elif text:
            self._out.add(BlockKind.LIST_ITEM, text, level=max(level, 1))

    def _on_blockquote_open(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._quote_depth += 1

    def _on_blockquote_close(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._quote_depth -= 1
        if self._quote_depth == 0:
            self._out.add(BlockKind.QUOTE, "\n".join(self._quote_lines))
            self._quote_lines = []

    # -- code, tables, html -------------------------------------------------------

    def _on_fence(self, token: Token, nxt: Token | None) -> None:
        del nxt
        language = token.info.strip().split(" ")[0] if token.info else ""
        attrs: dict[str, str | int | bool] = {"lines": token.content.count("\n")}
        if language:
            attrs["lang"] = language
        self._out.add(BlockKind.CODE, token.content, attrs=attrs)

    def _on_code_block(self, token: Token, nxt: Token | None) -> None:
        del nxt
        self._out.add(BlockKind.CODE, token.content, attrs={"lines": token.content.count("\n")})

    def _on_html_block(self, token: Token, nxt: Token | None) -> None:
        del nxt
        text = fragment_text(token.content)
        if text:
            self._out.add(BlockKind.PARAGRAPH, text)

    def _on_table_open(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._rows, self._columns = [], 0

    def _on_tr_open(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._row = []

    def _on_th_open(self, token: Token, nxt: Token | None) -> None:
        del token
        if self._row is not None:
            self._row.append(" ".join(_inline_text(nxt).split()))

    _on_td_open = _on_th_open

    def _on_tr_close(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        if self._row is not None:
            self._columns = max(self._columns, len(self._row))
            while self._row and not self._row[-1]:
                self._row.pop()  # cells the table syntax padded in
            self._rows.append(" | ".join(self._row))
        self._row = None

    def _on_table_close(self, token: Token, nxt: Token | None) -> None:
        del token, nxt
        self._out.add(
            BlockKind.TABLE,
            "\n".join(self._rows),
            attrs={"rows": len(self._rows), "columns": self._columns},
        )
