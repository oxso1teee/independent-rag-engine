"""Rendering of a processed document for ``rag chunks``: a JSON-able dict and a text listing.

Chunk ids are left out on purpose: they belong to a stored version, and a preview must print
the same output every time. Parents are referred to by the ordinal of the parent chunk.
"""

from __future__ import annotations

from rag_engine.domain import ChunkPolicy
from rag_engine.ingestion.processing import ProcessedDocument

_PREVIEW_CHARS = 100


def chunks_to_dict(done: ProcessedDocument, policy: ChunkPolicy, name: str) -> dict[str, object]:
    parsed, chunks = done.parsed, done.chunks
    ordinals = {chunk.id: chunk.ordinal for chunk in chunks}
    leaves = [c for c in chunks if c.level == 0]
    return {
        "document": {
            "name": name,
            "parser": parsed.parser_id,
            "parser_version": parsed.parser_version,
            "normalizer_version": parsed.normalizer_version,
            "mime": parsed.mime.mime,
            "encoding": parsed.mime.encoding,
            "size_bytes": parsed.size_bytes,
            "content_hash": str(parsed.content_hash),
            "title": parsed.document.title,
            "warnings": list(parsed.document.warnings),
        },
        "chunking": {
            "chunker": done.chunker_id,
            "version": done.chunker_version,
            "fingerprint": done.chunker_fingerprint,
            "target_tokens": policy.target_tokens,
            "max_tokens": policy.max_tokens,
            "min_tokens": policy.min_tokens,
            "overlap_ratio": policy.overlap_ratio,
        },
        "summary": {
            "chunks": len(chunks),
            "leaves": len(leaves),
            "sections": len(chunks) - len(leaves),
            "indexed": sum(1 for c in chunks if c.indexed),
            "duplicates": sum(1 for c in chunks if c.is_duplicate),
            "leaf_tokens": sum(c.token_count for c in leaves),
        },
        "chunks": [
            {
                "ordinal": c.ordinal,
                "level": c.level,
                "parent": ordinals[c.parent_id] if c.parent_id is not None else None,
                "kind": c.kind.value,
                "tokens": c.token_count,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "heading_path": list(c.heading_path),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "language": c.language.value,
                "indexed": c.indexed,
                "is_duplicate": c.is_duplicate,
                "text": c.text,
            }
            for c in chunks
        ],
    }


def describe_chunks(done: ProcessedDocument, name: str, *, full: bool) -> list[str]:
    """Header and one entry per chunk; ``full`` prints whole texts instead of a preview."""
    chunks = done.chunks
    leaves = sum(1 for c in chunks if c.level == 0)
    lines = [
        f"{name}: {done.parsed.parser_id} -> {done.chunker_id} "
        f"({done.chunker_fingerprint[:12]}), {len(chunks)} chunk(s): "
        f"{leaves} leaf, {len(chunks) - leaves} section"
    ]
    if done.parsed.document.warnings:
        lines.append("warnings: " + ", ".join(done.parsed.document.warnings))
    for chunk in chunks:
        flags = "".join(
            [
                " duplicate" if chunk.is_duplicate else "",
                "" if chunk.indexed or chunk.level else " not-indexed",
            ]
        )
        path = " > ".join(chunk.heading_path)
        lines.append(
            f"#{chunk.ordinal} {'section' if chunk.level else 'leaf'} {chunk.kind.value} "
            f"{chunk.token_count} tok{flags}" + (f"  [{path}]" if path else "")
        )
        if full:
            lines.extend(f"    {line}" for line in chunk.text.splitlines() or [""])
        else:
            preview = " ".join(chunk.text.split())
            more = "…" if len(preview) > _PREVIEW_CHARS else ""
            lines.append(f"    {preview[:_PREVIEW_CHARS]}{more}")
    return lines
