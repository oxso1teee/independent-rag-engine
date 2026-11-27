"""Review sheets: put a dataset in front of a person, and take back only that person's decisions.

The sheet is a CSV with one row per question: the question, its gold quotes with the text around
them, the reference answer and an empty ``decision`` column (``accept``, ``edit`` or ``reject``).
Applying a filled sheet records the decision, the reviewer's name and the date in each record.
Nothing is ever marked reviewed by this module on its own: a reviewer whose name looks like a
program is refused, an empty decision leaves the record untouched, and an unclear decision is an
error rather than a guess.
"""

from __future__ import annotations

import copy
import csv
import io
from collections.abc import Mapping
from datetime import date
from typing import Any

from rag_engine.domain import ErrorCode, RagError
from rag_engine.evaluation.dataset import Dataset, looks_like_program

COLUMNS = (
    "id", "split", "type", "language", "question", "answerable", "gold", "context",
    "reference_answer", "absence_terms", "status", "decision", "edited_question",
    "edited_reference_answer", "notes",
)  # fmt: skip
_DECISIONS = {"accept": "accepted", "edit": "edited", "reject": "rejected"}
_CONTEXT = 160


def _bad(message: str) -> RagError:
    return RagError(ErrorCode.E_CONFIG_INVALID, f"review sheet: {message}")


def _around(text: str, anchor: str) -> str:
    start = text.casefold().find(anchor.casefold())
    if start < 0:
        return ""
    low, high = max(0, start - _CONTEXT), min(len(text), start + len(anchor) + _CONTEXT)
    return (
        ("…" if low else "") + " ".join(text[low:high].split()) + ("…" if high < len(text) else "")
    )


def export_review_sheet(dataset: Dataset, corpus: Mapping[str, str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    for item in dataset.questions:
        writer.writerow(
            {
                "id": item.id,
                "split": item.split,
                "type": item.type,
                "language": item.language,
                "question": item.question,
                "answerable": "yes" if item.answerable else "no",
                "gold": "\n".join(f"{ref.doc} :: {ref.anchor}" for ref in item.gold),
                "context": "\n".join(
                    f"[{ref.doc}] {_around(corpus.get(ref.doc, ''), ref.anchor)}"
                    for ref in item.gold
                ),
                "reference_answer": item.reference_answer or "",
                "absence_terms": ", ".join(item.absence_terms),
                "status": item.review.status,
                "decision": "",
                "edited_question": "",
                "edited_reference_answer": "",
                "notes": item.review.notes,
            }
        )
    return out.getvalue()


def apply_review(data: Mapping[str, Any], sheet: str, *, reviewer: str, on: date) -> dict[str, Any]:
    """A copy of the raw dataset with the sheet's decisions recorded under ``reviewer``."""
    name = reviewer.strip()
    if not name or looks_like_program(name):
        raise _bad("the reviewer must be a person's name, not a program")
    rows = list(csv.DictReader(io.StringIO(sheet)))
    missing = {"id", "decision", "edited_question", "edited_reference_answer", "notes"} - set(
        rows[0] if rows else {}
    )
    if missing:
        raise _bad(f"missing columns {sorted(missing)}")
    updated = copy.deepcopy(dict(data))
    records = {item["id"]: item for item in updated["questions"]}
    for row in rows:
        decision = (row["decision"] or "").strip().lower()
        if not decision:
            continue
        record = records.get(row["id"])
        if record is None:
            raise _bad(f"unknown question id {row['id']!r}")
        if decision not in _DECISIONS:
            raise _bad(f"{row['id']}: decision must be accept, edit or reject, not {decision!r}")
        notes = (row["notes"] or "").strip()
        if decision == "reject" and not notes:
            raise _bad(f"{row['id']}: a rejection needs a reason in notes")
        if decision == "edit":
            question = (row["edited_question"] or "").strip()
            answer = (row["edited_reference_answer"] or "").strip()
            if not question and not answer:
                raise _bad(f"{row['id']}: an edit needs an edited question or answer")
            if question:
                record["question"] = question
            if answer:
                record["reference_answer"] = answer
        record["review"] = {
            "status": _DECISIONS[decision],
            "reviewer": name,
            "date": on.isoformat(),
            "notes": notes,
        }
    return updated
