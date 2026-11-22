"""Chunk-size grid report: how a corpus is cut at several target sizes.

Every document is parsed once and chunked at each grid point (default targets 192, 256 and 384
tokens; maximum 1.5 times the target, minimum 3/16 of it, 12 % overlap). The report describes
the *shape* of the result: counts, token distribution, share of small and near-limit chunks,
overlap, duplicates and time. It does not rank the sizes: choosing one needs retrieval quality
measurements, which come with the retrieval phases.

    uv run python eval/scripts/chunk_grid.py [--corpus DIR] [--format markdown|json] [--output FILE]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_engine.chunking.registry import STRATEGIES, ChunkerRegistry, build_counter
from rag_engine.domain import Chunk, ChunkPolicy, SourceKey
from rag_engine.ingestion.processing import DocumentProcessor, ParsedResult
from rag_engine.observability import configure_logging
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseLimits, TokenCounter
from rag_engine.system import SystemClock, system_id_factory

DEFAULT_CORPUS = Path(__file__).parents[2] / "tests" / "fixtures" / "corpus_mini"
DEFAULT_TARGETS = (192, 256, 384)
NOTE = (
    "This report describes the shape of the chunks only. It does not rank the sizes: choosing "
    "one needs retrieval quality measurements."
)


def grid_policy(target: int) -> ChunkPolicy:
    """The policy of one grid point: maximum 1.5 x target, minimum 3/16 x target."""
    return ChunkPolicy(
        target_tokens=target,
        max_tokens=target * 3 // 2,
        min_tokens=max(1, target * 3 // 16),
        overlap_ratio=0.12,
    )


@dataclass(frozen=True)
class GridRow:
    target: int
    max_tokens: int
    documents: int
    leaves: int
    sections: int
    duplicates: int
    leaf_tokens: int
    min: int
    median: float
    mean: float
    p95: int
    max: int
    small_share: float  # leaves below the minimum size
    near_limit_share: float  # leaves above 90 % of the maximum
    overlap_share: float  # characters of leaves that repeat the previous leaf
    seconds: float


def _percentile(values: Sequence[int], share: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))] if ordered else 0


def _overlap_chars(leaves: Sequence[Chunk]) -> int:
    total = 0
    previous: Chunk | None = None
    for leaf in leaves:
        if previous is not None and previous.parent_id == leaf.parent_id:
            total += max(0, previous.char_end - leaf.char_start)
        previous = leaf
    return total


def measure(parsed: Sequence[ParsedResult], registry: ChunkerRegistry, target: int) -> GridRow:
    policy = grid_policy(target)
    leaves: list[Chunk] = []
    sections = overlap = 0
    version = system_id_factory().document_version_id()
    started = time.perf_counter()
    for result in parsed:
        chunks = registry.select(result.document).chunk(
            result.document,
            policy,
            version_id=version,
            normalizer_version=result.normalizer_version,
        )
        document_leaves = [c for c in chunks if c.level == 0]
        leaves.extend(document_leaves)
        sections += len(chunks) - len(document_leaves)
        overlap += _overlap_chars(document_leaves)
    seconds = time.perf_counter() - started
    tokens = [c.token_count for c in leaves]
    chars = sum(len(c.text) for c in leaves) or 1
    count = len(leaves) or 1
    return GridRow(
        target=target,
        max_tokens=policy.max_tokens,
        documents=len(parsed),
        leaves=len(leaves),
        sections=sections,
        duplicates=sum(1 for c in leaves if c.is_duplicate),
        leaf_tokens=sum(tokens),
        min=min(tokens, default=0),
        median=statistics.median(tokens) if tokens else 0.0,
        mean=round(statistics.fmean(tokens), 1) if tokens else 0.0,
        p95=_percentile(tokens, 0.95),
        max=max(tokens, default=0),
        small_share=round(sum(1 for t in tokens if t < policy.min_tokens) / count, 3),
        near_limit_share=round(sum(1 for t in tokens if t > 0.9 * policy.max_tokens) / count, 3),
        overlap_share=round(overlap / chars, 3),
        seconds=round(seconds, 3),
    )


def parse_corpus(corpus: Path, processor: DocumentProcessor) -> list[ParsedResult]:
    results = []
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        key = SourceKey(path.relative_to(corpus).as_posix())
        results.append(processor.parse(path.read_bytes(), key, ParseLimits()))
    return results


def render_markdown(rows: Sequence[GridRow], corpus: str) -> str:
    head = (
        "| target | max | docs | leaves | sections | dup | tokens | min | median | mean | p95 "
        "| max | <min | >90% max | overlap | s |\n|" + "---|" * 16
    )
    lines = [f"# Chunk grid: {corpus}", "", NOTE, "", head]
    for r in rows:
        lines.append(
            f"| {r.target} | {r.max_tokens} | {r.documents} | {r.leaves} | {r.sections} "
            f"| {r.duplicates} | {r.leaf_tokens} | {r.min} | {r.median:g} | {r.mean} | {r.p95} "
            f"| {r.max} | {r.small_share:.1%} | {r.near_limit_share:.1%} "
            f"| {r.overlap_share:.1%} | {r.seconds} |"
        )
    return "\n".join(lines) + "\n"


def render_json(rows: Sequence[GridRow], corpus: str) -> str:
    payload = {"corpus": corpus, "note": NOTE, "rows": [asdict(r) for r in rows]}
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def run(
    corpus: Path,
    targets: Sequence[int] = DEFAULT_TARGETS,
    *,
    strategy: str = "auto",
    counter: TokenCounter | None = None,
) -> list[GridRow]:
    configure_logging("ERROR", stream=io.StringIO())
    registry = ChunkerRegistry(counter or build_counter("heuristic"), strategy)
    processor = DocumentProcessor(default_parsers(), SystemClock(), registry)
    parsed = parse_corpus(corpus, processor)
    return [measure(parsed, registry, target) for target in targets]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk-size grid report (shape only).")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=int, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--strategy", choices=STRATEGIES, default="auto")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = run(args.corpus, args.targets, strategy=args.strategy)
    name = args.corpus.name
    text = render_json(rows, name) if args.format == "json" else render_markdown(rows, name)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
