"""A self-contained kit for a person to review evaluation questions, and the way back.

``build_kit`` turns one or more datasets into **one HTML file** that works offline: for every
question the source documents, the expected answer, the evidence quoted with its context, the
closest passages for questions that have no answer, and controls to accept, edit or reject it. The
person's decisions are exported as a CSV sheet, and ``apply_kit_sheet`` records them, under the
reviewer's own name, into new versions of the datasets. Nothing here ever marks a question as
reviewed: building the kit changes no dataset, and a row without a decision stays a draft.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from rag_engine.domain import ErrorCode, Language, RagError
from rag_engine.evaluation.dataset import Dataset, Question, parse_dataset, validate_dataset
from rag_engine.evaluation.review import COLUMNS, apply_review
from rag_engine.lexical.analysis import analyze

SHEET_COLUMNS = ("dataset", *COLUMNS)
_CONTEXT_CHARS = 400
_NEIGHBOURS = 3


@dataclass(frozen=True, slots=True)
class SourceDataset:
    """A dataset as it is on disk, with its corpus."""

    name: str  # the file stem, used to route decisions back
    raw: Mapping[str, Any]
    dataset: Dataset
    corpus: Mapping[str, str]


def load_source(path: Path, root: Path) -> SourceDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = parse_dataset(raw)
    corpus_dir = root / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}
    report = validate_dataset(dataset, corpus)
    if not report.ok:
        raise RagError(
            ErrorCode.E_INTEGRITY,
            f"{path.name} does not validate",
            context={"errors": list(report.errors)[:5]},
        )
    return SourceDataset(path.stem, raw, dataset, corpus)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _terms(text: str) -> set[str]:
    return {t for t in analyze(text, Language.MIXED).split() if len(t) >= 3 or t.isdigit()}


def _around(text: str, anchor: str) -> dict[str, str]:
    start = text.casefold().find(anchor.casefold())
    if start < 0:
        return {"before": "", "anchor": anchor, "after": ""}
    low = max(0, start - _CONTEXT_CHARS)
    high = min(len(text), start + len(anchor) + _CONTEXT_CHARS)
    return {
        "before": ("…" if low else "") + text[low:start],
        "anchor": text[start : start + len(anchor)],
        "after": text[start + len(anchor) : high] + ("…" if high < len(text) else ""),
    }


def _closest(question: Question, corpus: Mapping[str, str]) -> list[dict[str, str]]:
    """The corpus sentences that share most terms with the question (no model: stems only)."""
    wanted = _terms(question.question)
    scored = []
    for doc, text in corpus.items():
        for sentence in _sentences(text):
            overlap = len(wanted & _terms(sentence))
            if overlap:
                scored.append((overlap, doc, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"doc": d, "text": s} for _, d, s in scored[:_NEIGHBOURS]]


def question_card(source: SourceDataset, question: Question) -> dict[str, Any]:
    evidence = [
        {"doc": ref.doc, **_around(source.corpus.get(ref.doc, ""), ref.anchor)}
        for ref in question.gold
    ]
    card: dict[str, Any] = {
        "key": f"{source.name}:{question.id}",
        "dataset": source.name,
        "id": question.id,
        "split": question.split,
        "type": question.type,
        "language": question.language,
        "question": question.question,
        "answerable": question.answerable,
        "reference_answer": question.reference_answer or "",
        "reference_facts": list(question.reference_facts),
        "evidence": evidence,
        "gold": "\n".join(f"{ref.doc} :: {ref.anchor}" for ref in question.gold),
        "absence_terms": list(question.absence_terms),
        "closest": [] if question.answerable else _closest(question, source.corpus),
        "status": question.review.status,
        "provenance": f"{question.provenance.created_by}: {question.provenance.method}",
    }
    return card


AUDIT_NOTE = (
    "The audit column is an automatic screen made from the corpus alone (stems, figures and, when "
    "it was run, an embedding model's ranking of every passage). 'correct' means it found no "
    "objection, not that anybody checked the question; 'doubtful' means look at it; 'incorrect' "
    "means something is wrong on its face. It never fills in a decision. A bulk accept applies "
    "only to rows you selected and whose evidence you opened."
)


def build_kit(
    sources: Sequence[SourceDataset],
    *,
    title: str = "Evaluation question review",
    audits: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """The review page. ``audits`` maps ``dataset:id`` to an audit (``QuestionAudit.as_dict()``)."""
    cards = [question_card(s, q) for s in sources for q in s.dataset.questions]
    if audits is not None:
        for card in cards:
            card["audit"] = audits.get(card["key"])
    documents = {f"{s.name}/{doc}": text for s in sources for doc, text in s.corpus.items()}
    payload = {
        "title": title,
        "columns": [*SHEET_COLUMNS, *(["audit"] if audits is not None else [])],
        "audit_note": AUDIT_NOTE,
        "cards": cards,
        "documents": documents,
        "datasets": {s.name: s.dataset.identifier for s in sources},
    }
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _page().replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


# --------------------------------------------------------------------------- applying a sheet


@dataclass(frozen=True, slots=True)
class AppliedDataset:
    name: str
    version: int
    path: Path
    accepted: int
    edited: int
    rejected: int
    undecided: int


def apply_kit_sheet(
    sources: Sequence[SourceDataset], sheet: str, *, reviewer: str, outdir: Path, on: date
) -> list[AppliedDataset]:
    """Record the sheet's decisions into new dataset versions in ``outdir``; never overwrite."""
    rows = list(csv.DictReader(io.StringIO(sheet)))
    if rows and "dataset" not in rows[0]:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "review sheet: missing the dataset column")
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    known = {s.name for s in sources}
    unknown = set(by_dataset) - known
    if unknown:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, f"review sheet: unknown dataset {sorted(unknown)}"
        )
    applied: list[AppliedDataset] = []
    outdir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        decided = by_dataset.get(source.name, [])
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(COLUMNS), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(decided)
        updated = apply_review(source.raw, buffer.getvalue(), reviewer=reviewer, on=on)
        version = int(source.raw["version"]) + 1
        updated["version"] = version
        if not validate_dataset(parse_dataset(updated), source.corpus).ok:
            raise RagError(ErrorCode.E_INTEGRITY, f"{source.name}: the reviewed dataset is invalid")
        target = outdir / f"{source.name.rsplit('_v', 1)[0]}_v{version}.json"
        if target.exists():
            raise RagError(ErrorCode.E_INTEGRITY, f"{target.name} already exists; not overwriting")
        target.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        statuses = [q["review"]["status"] for q in updated["questions"]]
        applied.append(
            AppliedDataset(
                source.name, version, target,
                statuses.count("accepted"), statuses.count("edited"), statuses.count("rejected"),
                statuses.count("unreviewed"),
            )
        )  # fmt: skip
    return applied


def _page() -> str:
    return resources.files("rag_engine.evaluation").joinpath("review_page.html").read_text("utf-8")
