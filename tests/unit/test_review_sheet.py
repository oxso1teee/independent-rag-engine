"""Exporting a dataset for a person to review, and applying only a person's decisions."""

from __future__ import annotations

import csv
import io
from datetime import date
from typing import Any

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.evaluation.dataset import parse_dataset, validate_dataset
from rag_engine.evaluation.review import apply_review, export_review_sheet

pytestmark = pytest.mark.unit

CORPUS = {
    "a.md": (
        "Backups run every night at 02:30 UTC and are kept for 14 days. "
        "Restores are tested monthly."
    ),
    "b.md": "Employees receive 28 calendar days of paid vacation per year.",
}


def raw(*questions: dict[str, Any]) -> dict[str, Any]:
    return {"name": "demo", "version": 1, "corpus": "docs", "questions": list(questions)}


def question(identifier: str, **changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": identifier,
        "split": "dev",
        "type": "keyword",
        "language": "en",
        "question": "When do the backups run?",
        "answerable": True,
        "gold": [{"doc": "a.md", "anchor": "every night at 02:30 UTC", "min_support": 1}],
        "reference_answer": "Every night at 02:30 UTC.",
        "reference_facts": [],
        "absence_terms": [],
        "provenance": {"created_by": "llm-assisted", "method": "drafted", "created": "2026-11-27"},
        "review": {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""},
    }
    base.update(changes)
    return base


def no_answer(identifier: str) -> dict[str, Any]:
    return question(
        identifier,
        type="no_answer",
        answerable=False,
        gold=[],
        question="How large is the bonus?",
        reference_answer=None,
        absence_terms=["bonus"],
    )


def sheet(data: dict[str, Any]) -> list[dict[str, str]]:
    text = export_review_sheet(parse_dataset(data), CORPUS)
    return list(csv.DictReader(io.StringIO(text)))


def fill(rows: list[dict[str, str]], **decisions: dict[str, str]) -> str:
    for row in rows:
        row.update(decisions.get(row["id"], {}))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def test_the_sheet_shows_what_a_reviewer_needs_to_judge_each_question() -> None:
    rows = sheet(raw(question("q1"), no_answer("q2")))
    first = rows[0]
    assert first["id"] == "q1" and first["status"] == "unreviewed" and first["decision"] == ""
    assert "every night at 02:30 UTC" in first["gold"] and "a.md" in first["gold"]
    assert "kept for 14 days" in first["context"]  # the text around the anchor, from the corpus
    assert rows[1]["absence_terms"] == "bonus" and rows[1]["gold"] == ""


def test_decisions_become_reviews_signed_by_the_named_person_only() -> None:
    data = raw(
        question("q1"), question("q2", question="How long are copies kept?"), no_answer("q3")
    )
    decisions = fill(
        sheet(data),
        q1={"decision": "accept", "notes": "checked against a.md"},
        q2={
            "decision": "edit",
            "edited_question": "For how many days are copies kept?",
            "edited_reference_answer": "14 days.",
        },
        q3={"decision": "reject", "notes": "ambiguous"},
    )
    updated = apply_review(data, decisions, reviewer="Ada Lovelace", on=date(2026, 12, 1))
    by_id = {item["id"]: item for item in updated["questions"]}
    assert by_id["q1"]["review"] == {
        "status": "accepted",
        "reviewer": "Ada Lovelace",
        "date": "2026-12-01",
        "notes": "checked against a.md",
    }
    assert by_id["q2"]["review"]["status"] == "edited"
    assert by_id["q2"]["question"] == "For how many days are copies kept?"
    assert by_id["q2"]["reference_answer"] == "14 days."
    assert by_id["q3"]["review"]["status"] == "rejected"
    assert data["questions"][0]["review"]["status"] == "unreviewed"  # the input is not mutated
    assert validate_dataset(parse_dataset(updated), CORPUS).ok


def test_an_empty_decision_leaves_the_question_unreviewed() -> None:
    data = raw(question("q1"), question("q2"))
    updated = apply_review(
        data, fill(sheet(data), q1={"decision": "accept"}), reviewer="Ada", on=date(2026, 12, 1)
    )
    statuses = {q["id"]: q["review"]["status"] for q in updated["questions"]}
    assert statuses == {"q1": "accepted", "q2": "unreviewed"}


@pytest.mark.parametrize("name", ["", "   ", "review-bot", "gpt-4", "Claude", "AI assistant"])
def test_a_reviewer_must_be_a_person(name: str) -> None:
    data = raw(question("q1"))
    with pytest.raises(RagError) as info:
        apply_review(
            data, fill(sheet(data), q1={"decision": "accept"}), reviewer=name, on=date(2026, 12, 1)
        )
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


@pytest.mark.parametrize(
    "decisions",
    [
        {"q1": {"decision": "approve"}},
        {"q1": {"decision": "reject"}},  # a rejection needs a reason
        {"q1": {"decision": "edit"}},  # an edit needs a changed field
        {"q9": {"decision": "accept"}},
    ],
)
def test_unclear_decisions_are_refused_rather_than_guessed(
    decisions: dict[str, dict[str, str]],
) -> None:
    data = raw(question("q1"))
    rows = sheet(data)
    if "q9" in decisions:
        rows.append({**rows[0], "id": "q9", "decision": "accept"})
        text = fill(rows)
    else:
        text = fill(rows, **decisions)
    with pytest.raises(RagError) as info:
        apply_review(data, text, reviewer="Ada", on=date(2026, 12, 1))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
