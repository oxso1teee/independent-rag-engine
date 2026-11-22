"""A compact, exact description of how the ``corpus_mini`` documents are parsed and chunked.

The snapshot is compared with a stored copy (``tests/fixtures/golden/corpus_mini.json``): any
change to normalisation, parsing, chunk boundaries, token counts or identifiers shows up as a
reviewable difference. ``python tests/fixtures/update_golden.py`` rewrites the stored copy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_engine.chunking.registry import ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ChunkPolicy, SourceKey
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseLimits
from rag_engine.testing import FakeClock
from tests.factories import make_ids

CORPUS = Path(__file__).parent / "fixtures" / "corpus_mini"
GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "corpus_mini.json"
POLICY = ChunkPolicy()


def snapshot(name: str) -> dict[str, Any]:
    counter = HeuristicTokenCounter()
    processor = DocumentProcessor(default_parsers(), FakeClock(), ChunkerRegistry(counter))
    version = make_ids().document_version_id()
    done = processor.process(
        (CORPUS / name).read_bytes(), SourceKey(name), ParseLimits(), POLICY, version
    )
    parsed, document = done.parsed, done.parsed.document
    ordinals = {chunk.id: chunk.ordinal for chunk in done.chunks}
    return {
        "parser": [parsed.parser_id, parsed.parser_version, parsed.normalizer_version],
        "mime": parsed.mime.mime,
        "encoding": parsed.mime.encoding,
        "content_hash": str(parsed.content_hash)[:16],
        "title": document.title,
        "author": document.author,
        "language": document.language.value,
        "warnings": list(document.warnings),
        "text_chars": len(document.normalized_text),
        "text_hash": hashlib.sha256(document.normalized_text.encode()).hexdigest()[:16],
        "blocks": [b.kind.value for b in document.blocks],
        "chunker": [done.chunker_id, done.chunker_version, done.chunker_fingerprint[:16]],
        "chunks": [
            [
                c.ordinal,
                c.level,
                None if c.parent_id is None else ordinals[c.parent_id],
                c.kind.value,
                c.token_count,
                c.char_start,
                c.char_end,
                str(c.text_hash)[:12],
                list(c.heading_path),
                c.language.value,
                c.page_start,
                c.page_end,
                c.is_duplicate,
            ]
            for c in done.chunks
        ],
    }


def snapshot_all() -> dict[str, Any]:
    return {path.name: snapshot(path.name) for path in sorted(CORPUS.iterdir())}


def dump_snapshot(data: dict[str, Any]) -> str:
    """Stable text with one chunk per line, so a change reads as a small diff."""
    out = ["{"]
    names = sorted(data)
    for number, name in enumerate(names):
        entry = data[name]
        out.append(f" {json.dumps(name)}: {{")
        for key in sorted(k for k in entry if k != "chunks"):
            out.append(f"  {json.dumps(key)}: {json.dumps(entry[key], ensure_ascii=False)},")
        out.append('  "chunks": [')
        rows = [f"   {json.dumps(row, ensure_ascii=False)}" for row in entry["chunks"]]
        out.append(",\n".join(rows))
        out.append("  ]")
        out.append(" }" + ("," if number < len(names) - 1 else ""))
    out.append("}")
    return "\n".join(out) + "\n"
