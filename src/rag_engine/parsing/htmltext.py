"""Plain text of an HTML fragment (used for raw HTML inside Markdown)."""

from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser

DROPPED_TAGS = ("script", "style", "noscript", "template", "svg", "canvas", "iframe")


def fragment_text(markup: str) -> str:
    """Visible text of ``markup`` on one line (inline and block elements alike), with entities
    decoded once by the HTML parser."""
    tree = LexborHTMLParser(markup)
    for tag in DROPPED_TAGS:
        for node in tree.css(tag):
            node.decompose()
    body = tree.body
    return body.text(separator=" ", strip=True) if body is not None else ""
