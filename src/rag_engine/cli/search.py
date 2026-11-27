"""Arguments, display data and rendering of ``rag search`` (the pipeline is in retrieval)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from rag_engine.domain import (
    ChunkId,
    ChunkKind,
    Context,
    DocumentId,
    DocumentVersionId,
    ErrorCode,
    Language,
    RagError,
    RetrievalFilters,
    RetrievalResult,
)
from rag_engine.ports import Store, StoredChunk
from rag_engine.retrieval.service import RetrievalDiagnostics, SearchOutcome

_PREVIEW_CHARS = 200


def _invalid(message: str) -> RagError:
    return RagError(ErrorCode.E_FILTER_INVALID, message)


def _moment(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _invalid(f"{name} must be an ISO 8601 date or time") from None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _enum[T: (Language, ChunkKind)](kind: type[T], value: str, name: str) -> T:
    try:
        return kind(value)
    except ValueError:
        allowed = ", ".join(item.value for item in kind)
        raise _invalid(f"{name} must be one of: {allowed}") from None


def _documents(values: Sequence[str], name: str) -> tuple[DocumentId, ...]:
    try:
        return tuple(DocumentId.parse(value) for value in values)
    except (ValueError, RagError):
        raise _invalid(f"{name} must be a document id") from None


def parse_filters(
    *,
    sources: Sequence[str],
    path_prefix: str | None,
    path_glob: str | None,
    tags_any: Sequence[str],
    tags_all: Sequence[str],
    language: str | None,
    kinds: Sequence[str],
    modified_after: str | None,
    modified_before: str | None,
    doc_ids: Sequence[str],
    exclude_doc_ids: Sequence[str],
) -> RetrievalFilters:
    try:
        return RetrievalFilters(
            sources=tuple(sources),
            path_prefix=path_prefix,
            path_glob=path_glob,
            tags_any=tuple(tags_any),
            tags_all=tuple(tags_all),
            language=None if language is None else _enum(Language, language, "--language"),
            modified_after=_moment(modified_after, "--modified-after"),
            modified_before=_moment(modified_before, "--modified-before"),
            doc_ids=_documents(doc_ids, "--doc-id"),
            exclude_doc_ids=_documents(exclude_doc_ids, "--exclude-doc-id"),
            kinds=tuple(_enum(ChunkKind, value, "--kind") for value in kinds),
        )
    except RagError as error:
        if error.code is ErrorCode.E_INTEGRITY:  # a domain invariant, e.g. an empty time window
            raise _invalid(error.message) from error
        raise


@dataclass(frozen=True, slots=True)
class HitDetail:
    rank: int
    chunk_id: str
    document_id: str
    source: str
    path: str
    heading_path: tuple[str, ...]
    text: str
    dense: float | None
    lexical: float | None
    fused: float | None


def load_details(store: Store, result: RetrievalResult) -> list[HitDetail]:
    """Names and text of each hit, read once per document and version."""
    reader = store.reader()
    versions: dict[DocumentVersionId, Mapping[ChunkId, StoredChunk]] = {}
    details: list[HitDetail] = []
    for hit in result.hits:
        ref = hit.chunk_ref
        document = reader.get_document(ref.document_id)
        source = None if document is None else reader.get_source(document.source_id)
        if ref.version_id not in versions:
            versions[ref.version_id] = {
                item.chunk.id: item for item in reader.list_chunks(ref.version_id)
            }
        stored = versions[ref.version_id].get(ref.chunk_id)
        details.append(
            HitDetail(
                rank=hit.rank,
                chunk_id=str(ref.chunk_id),
                document_id=str(ref.document_id),
                source="" if source is None else source.name,
                path="" if document is None else str(document.source_key),
                heading_path=() if stored is None else stored.chunk.heading_path,
                text="" if stored is None else stored.chunk.text,
                dense=hit.scores.dense,
                lexical=hit.scores.lexical,
                fused=hit.scores.fused,
            )
        )
    return details


def _diagnostics(item: RetrievalDiagnostics) -> dict[str, object]:
    return {
        "mode": item.mode.value,
        "query_language": item.language.value,
        "query_tokens": item.query_tokens,
        "filter_matched": item.filter_matched,
        "dense_candidates": item.dense_candidates,
        "lexical_candidates": item.lexical_candidates,
        "fused_candidates": item.fused_candidates,
        "collapsed_duplicates": item.collapsed_duplicates,
        "capped": item.capped,
        "unknown": item.unknown,
        "document_share": item.document_share,
        "channel_ranks": {
            str(chunk): {"dense": ranks[0], "lexical": ranks[1]}
            for chunk, ranks in item.channel_ranks.items()
        },
        "timings_ms": dict(item.timings_ms),
    }


def outcome_to_dict(
    outcome: SearchOutcome, details: Sequence[HitDetail], *, debug: bool
) -> dict[str, object]:
    result = outcome.result
    data: dict[str, object] = {
        "request_id": str(result.request_id),
        "mode": outcome.diagnostics.mode.value,
        "index_generation": result.index_generation,
        "embedding_model": None
        if result.embedding_model_id is None
        else str(result.embedding_model_id),
        "fallbacks": list(result.fallbacks),
        "hits": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "source": item.source,
                "path": item.path,
                "heading_path": list(item.heading_path),
                "text": item.text,
                "scores": {"dense": item.dense, "lexical": item.lexical, "fused": item.fused},
            }
            for item in details
        ],
    }
    if debug:
        data["diagnostics"] = _diagnostics(outcome.diagnostics)
    return data


def describe_hits(outcome: SearchOutcome, details: Sequence[HitDetail], *, full: bool) -> list[str]:
    lines: list[str] = []
    for item in details:
        score = "" if item.fused is None else f"{item.fused:.4f}"
        where = f"{item.source}/{item.path}" if item.source else item.path
        headings = f"  [{' > '.join(item.heading_path)}]" if item.heading_path else ""
        lines.append(f"{item.rank:>2}  {score:>7}  {where}{headings}")
        text = " ".join(item.text.split())
        if not full and len(text) > _PREVIEW_CHARS:
            text = text[: _PREVIEW_CHARS - 1] + "…"
        lines.append(f"    {text}")
    if not details:
        lines.append("no results")
    for name in outcome.result.fallbacks:
        lines.append(f"note: {name}")
    return lines


def context_to_dict(
    outcome: SearchOutcome, context: Context, *, system_tokens: int, question_tokens: int
) -> dict[str, object]:
    """The context, its budget arithmetic and the full accounting of what was left out."""
    budget, report = context.budget, context.report
    return {
        "request_id": str(outcome.result.request_id),
        "mode": outcome.diagnostics.mode.value,
        "index_generation": outcome.result.index_generation,
        "fallbacks": list(outcome.result.fallbacks),
        "budget": {
            "window": budget.window,
            "system": system_tokens,
            "question": question_tokens,
            "max_new": budget.max_new,
            "margin": budget.margin,
            "evidence_budget": budget.evidence_budget,
            "tokens_used": context.tokens_used,
        },
        "context_hash": context.context_hash,
        "evidence": [
            {
                "id": item.evidence_id,
                "chunk_id": str(item.chunk_id),
                "matched": [str(chunk) for chunk in item.matched],
                "document_id": str(item.document_id),
                "version_id": str(item.version_id),
                "source": item.locator.source_name,
                "path": item.locator.source_key,
                "title": item.locator.title,
                "heading_path": list(item.locator.heading_path),
                "pages": [item.locator.page_start, item.locator.page_end],
                "char_start": item.locator.char_start,
                "char_end": item.locator.char_end,
                "truncated": item.truncated,
                "score": item.score,
                "tokens": item.tokens,
                "text": item.text,
            }
            for item in context.evidence
        ],
        "report": {
            "included": [
                {"chunk_id": str(i.chunk_id), "evidence_id": i.evidence_id, "tokens": i.tokens}
                for i in report.included
            ],
            "dropped": [
                {"chunk_id": str(d.chunk_id), "reason": d.reason.value} for d in report.dropped
            ],
            "merged": [{"chunk_id": str(m.chunk_id), "into": str(m.into)} for m in report.merged],
            "truncated": [str(chunk) for chunk in report.truncated],
            "tokens_used": report.tokens_used,
            "budget": report.budget,
        },
    }


def describe_context(
    outcome: SearchOutcome,
    context: Context,
    *,
    system_tokens: int,
    question_tokens: int,
    full: bool,
) -> list[str]:
    budget = context.budget
    lines = [
        f"budget: window {budget.window} - system {system_tokens} - question {question_tokens}"
        f" - max_new {budget.max_new} - margin {budget.margin} = evidence {budget.evidence_budget};"
        f" used {context.tokens_used}",
    ]
    for item in context.evidence:
        where = f"{item.locator.source_name}/{item.locator.source_key}"
        headings = (
            f"  [{' > '.join(item.locator.heading_path)}]" if item.locator.heading_path else ""
        )
        cut = "  (truncated)" if item.truncated else ""
        lines.append(f"{item.evidence_id}  {item.tokens:>4} tok  {where}{headings}{cut}")
        body = " ".join(item.text.split())
        if not full and len(body) > _PREVIEW_CHARS:
            body = body[: _PREVIEW_CHARS - 1] + "…"
        lines.append(f"    {body}")
    for dropped in context.report.dropped:
        lines.append(f"dropped {dropped.reason.value}: {dropped.chunk_id}")
    for merged in context.report.merged:
        lines.append(f"merged {merged.chunk_id} into {merged.into}")
    for name in outcome.result.fallbacks:
        lines.append(f"note: {name}")
    return lines
