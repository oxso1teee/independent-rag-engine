"""Minimal reader of the metadata block at the top of a Markdown file.

Only flat ``key: value`` lines are understood; anything richer (lists, nesting) is skipped
and reported, so no YAML library is needed and nothing is executed.
"""

from __future__ import annotations

from datetime import UTC, datetime

MAX_FRONT_MATTER = 16 * 1024


def split_front_matter(text: str) -> tuple[dict[str, str], str, list[str]]:
    """``(metadata, body, warnings)``; the body is the text without the metadata block."""
    stripped = text.lstrip("﻿")
    if not stripped.startswith("---"):
        return {}, text, []
    first_end = stripped.find("\n")
    if first_end == -1 or stripped[:first_end].strip() != "---":
        return {}, text, []
    meta: dict[str, str] = {}
    warnings: list[str] = []
    offset = first_end + 1
    for line in stripped[offset : offset + MAX_FRONT_MATTER].splitlines(keepends=True):
        offset += len(line)
        content = line.strip()
        if content in ("---", "..."):
            return meta, stripped[offset:], warnings
        if not content or content.startswith("#"):
            continue
        key, separator, value = content.partition(":")
        if not separator or line[0] in " \t-" or not key.strip().replace("_", "").isalnum():
            if "front_matter_partial" not in warnings:
                warnings.append("front_matter_partial")
            continue
        meta[key.strip().lower()] = value.strip().strip("\"'")
    return {}, text, []  # never closed: it was not a metadata block


def parse_date(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
