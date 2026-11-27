"""Evaluation datasets, schema v2: strict parsing, validation and metric eligibility.

A record is a question with its gold evidence (document plus a verbatim anchor quote), its type
and split, and a review. Labels are not truth until a person has accepted them: metrics are
computed only from ``accepted`` and ``edited`` records, and a reviewer whose name looks like a
program is refused. The dataset is identified by ``name@vN`` and the SHA-256 of its canonical
JSON, so a run can state exactly which labels it used.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rag_engine.domain import ErrorCode, RagError

SPLITS = ("dev", "test")
TYPES = ("keyword", "paraphrase", "multi_doc", "no_answer", "conflicting", "out_of_scope")
LANGUAGES = ("ru", "en")
STATUSES = ("unreviewed", "accepted", "edited", "rejected")
COUNTED = ("accepted", "edited")
MIN_ACCEPTED = 30
PARAPHRASE_OVERLAP = 0.30
_BOT = re.compile(r"\b(bot|gpt|chatgpt|claude|llm|ai|assistant|copilot|gemini|codex|auto)\b", re.I)
_QUESTION_KEYS = {
    "id", "split", "type", "language", "question", "answerable", "gold", "reference_answer",
    "reference_facts", "absence_terms", "provenance", "review",
}  # fmt: skip
_REQUIRED = {
    "id", "split", "type", "language", "question", "answerable", "gold", "provenance", "review",
}  # fmt: skip


def looks_like_program(name: str) -> bool:
    """Whether a reviewer name reads like a tool or a model rather than a person."""
    return _BOT.search(name) is not None


@dataclass(frozen=True, slots=True)
class GoldRef:
    doc: str
    anchor: str
    min_support: int = 1


@dataclass(frozen=True, slots=True)
class Provenance:
    created_by: str
    method: str
    created: str


@dataclass(frozen=True, slots=True)
class Review:
    status: str
    reviewer: str | None
    date: str | None
    notes: str


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    split: str
    type: str
    language: str
    question: str
    answerable: bool
    gold: tuple[GoldRef, ...]
    reference_answer: str | None
    reference_facts: tuple[str, ...]
    absence_terms: tuple[str, ...]
    provenance: Provenance
    review: Review


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    version: int
    corpus: str
    questions: tuple[Question, ...]
    manifest_hash: str

    @property
    def identifier(self) -> str:
        return f"{self.name}@v{self.version}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _bad(message: str) -> RagError:
    return RagError(ErrorCode.E_CONFIG_INVALID, f"invalid evaluation dataset: {message}")


def _text(value: object, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _bad(f"{where} must be a non-empty string")
    return value


def _choice(value: object, allowed: Sequence[str], where: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _bad(f"{where} must be one of {', '.join(allowed)}")
    return value


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _bad(f"{where} must be a list of strings")
    return tuple(value)


def _optional_text(value: object, where: str) -> str | None:
    return None if value is None else _text(value, where, allow_empty=True)


def _gold(value: object, where: str) -> tuple[GoldRef, ...]:
    if not isinstance(value, list):
        raise _bad(f"{where} must be a list")
    items: list[GoldRef] = []
    for position, entry in enumerate(value):
        if not isinstance(entry, Mapping) or not {"doc", "anchor"} <= entry.keys():
            raise _bad(f"{where}[{position}] needs doc and anchor")
        support = entry.get("min_support", 1)
        if not isinstance(support, int) or isinstance(support, bool) or support < 1:
            raise _bad(f"{where}[{position}].min_support must be a positive integer")
        items.append(
            GoldRef(
                _text(entry["doc"], f"{where}[{position}].doc"),
                _text(entry["anchor"], f"{where}[{position}].anchor"),
                support,
            )
        )
    return tuple(items)


def _question(raw: object, position: int) -> Question:
    where = f"questions[{position}]"
    if not isinstance(raw, Mapping):
        raise _bad(f"{where} must be an object")
    missing, unknown = _REQUIRED - raw.keys(), raw.keys() - _QUESTION_KEYS
    if missing or unknown:
        raise _bad(f"{where} has missing {sorted(missing)} or unknown {sorted(unknown)} keys")
    provenance, review = raw["provenance"], raw["review"]
    if not isinstance(provenance, Mapping) or not isinstance(review, Mapping):
        raise _bad(f"{where}.provenance and .review must be objects")
    answerable = raw["answerable"]
    if not isinstance(answerable, bool):
        raise _bad(f"{where}.answerable must be true or false")
    return Question(
        id=_text(raw["id"], f"{where}.id"),
        split=_choice(raw["split"], SPLITS, f"{where}.split"),
        type=_choice(raw["type"], TYPES, f"{where}.type"),
        language=_choice(raw["language"], LANGUAGES, f"{where}.language"),
        question=_text(raw["question"], f"{where}.question"),
        answerable=answerable,
        gold=_gold(raw["gold"], f"{where}.gold"),
        reference_answer=_optional_text(raw.get("reference_answer"), f"{where}.reference_answer"),
        reference_facts=_strings(raw.get("reference_facts", []), f"{where}.reference_facts"),
        absence_terms=_strings(raw.get("absence_terms", []), f"{where}.absence_terms"),
        provenance=Provenance(
            _choice(provenance.get("created_by"), ("human", "llm-assisted"), f"{where}.provenance"),
            _text(provenance.get("method"), f"{where}.provenance.method"),
            _text(provenance.get("created"), f"{where}.provenance.created"),
        ),
        review=Review(
            _choice(review.get("status"), STATUSES, f"{where}.review.status"),
            _optional_text(review.get("reviewer"), f"{where}.review.reviewer"),
            _optional_text(review.get("date"), f"{where}.review.date"),
            _text(review.get("notes", ""), f"{where}.review.notes", allow_empty=True),
        ),
    )


def parse_dataset(data: Mapping[str, Any]) -> Dataset:
    if data.keys() != {"name", "version", "corpus", "questions"}:
        raise _bad("the top level must have exactly name, version, corpus and questions")
    version = data["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _bad("version must be a positive integer")
    if not isinstance(data["questions"], list) or not data["questions"]:
        raise _bad("questions must be a non-empty list")
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return Dataset(
        name=_text(data["name"], "name"),
        version=version,
        corpus=_text(data["corpus"], "corpus"),
        questions=tuple(_question(raw, position) for position, raw in enumerate(data["questions"])),
        manifest_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def load_dataset(path: Path) -> Dataset:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise _bad(f"{path.name} cannot be read") from None
    except ValueError:
        raise _bad(f"{path.name} is not valid JSON") from None
    if not isinstance(raw, Mapping):
        raise _bad("the file must contain one JSON object")
    return parse_dataset(raw)


def accepted_questions(dataset: Dataset) -> list[Question]:
    """The only records metrics may use."""
    return [item for item in dataset.questions if item.review.status in COUNTED]


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _groups(questions: Sequence[Question]) -> list[set[str]]:
    """Connected groups of gold documents: questions that share one are linked."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for question in questions:
        docs = [ref.doc for ref in question.gold]
        for doc in docs:
            find(doc)
        for other in docs[1:]:
            parent[find(other)] = find(docs[0])
    groups: dict[str, set[str]] = {}
    for doc in parent:
        groups.setdefault(find(doc), set()).add(doc)
    return list(groups.values())


def validate_dataset(dataset: Dataset, corpus: Mapping[str, str]) -> ValidationReport:
    """Check records against the corpus (document path to normalised text)."""
    errors: list[str] = []
    warnings: list[str] = []
    folded = {doc: _fold(text) for doc, text in corpus.items()}
    everything = " ".join(folded.values())
    seen: set[str] = set()
    for item in dataset.questions:
        if item.id in seen:
            errors.append(f"duplicate question id {item.id}")
        seen.add(item.id)
        _check_gold(item, folded, errors)
        _check_type(item, everything, errors)
        _check_review(item, errors)
        if item.type == "paraphrase" and item.gold:
            overlap = _overlap(item)
            if overlap > PARAPHRASE_OVERLAP:
                warnings.append(
                    f"{item.id}: paraphrase overlap {overlap:.2f} with its anchor "
                    f"exceeds {PARAPHRASE_OVERLAP}"
                )
    for group in _groups(dataset.questions):
        splits = {q.split for q in dataset.questions if any(r.doc in group for r in q.gold)}
        if len(splits) > 1:
            errors.append(
                f"split policy: questions sharing {sorted(group)} are in {sorted(splits)}"
            )
    counted = len(accepted_questions(dataset))
    if counted < MIN_ACCEPTED:
        warnings.append(f"anecdotal: only {counted} accepted questions ({MIN_ACCEPTED} needed)")
    return ValidationReport(tuple(errors), tuple(warnings))


def _check_gold(item: Question, folded: Mapping[str, str], errors: list[str]) -> None:
    for ref in item.gold:
        text = folded.get(ref.doc)
        if text is None:
            errors.append(f"{item.id}: gold document {ref.doc} is not in the corpus")
        elif _fold(ref.anchor) not in text:
            errors.append(f"{item.id}: anchor is not a verbatim quote of {ref.doc}")


def _check_type(item: Question, everything: str, errors: list[str]) -> None:
    if item.answerable:
        if not item.gold:
            errors.append(f"{item.id}: an answerable question needs gold evidence")
        if item.type in {"no_answer", "out_of_scope"}:
            errors.append(f"{item.id}: type {item.type} cannot be answerable")
        if item.type == "multi_doc" and len({ref.doc for ref in item.gold}) < 2:
            errors.append(f"{item.id}: multi_doc needs gold in at least two documents")
        return
    if item.gold:
        errors.append(f"{item.id}: an unanswerable question has no gold")
    if item.type not in {"no_answer", "out_of_scope"}:
        errors.append(f"{item.id}: an unanswerable question must have type no_answer")
    if not item.absence_terms:
        errors.append(f"{item.id}: an unanswerable question needs absence_terms")
    for term in item.absence_terms:
        if _fold(term) in everything:
            errors.append(f"{item.id}: absence term {term!r} appears in the corpus")


def _check_review(item: Question, errors: list[str]) -> None:
    review = item.review
    if review.reviewer and looks_like_program(review.reviewer):
        errors.append(f"{item.id}: the reviewer {review.reviewer!r} looks like a program")
    if review.status not in COUNTED:
        return
    if not review.reviewer or not review.reviewer.strip():
        errors.append(f"{item.id}: an {review.status} record needs a reviewer")
    try:
        date.fromisoformat(review.date or "")
    except ValueError:
        errors.append(f"{item.id}: an {review.status} record needs an ISO review date")


def _overlap(item: Question) -> float:
    words = set(re.findall(r"\w+", _fold(item.question)))
    anchor = set(re.findall(r"\w+", _fold(" ".join(ref.anchor for ref in item.gold))))
    return len(words & anchor) / len(words) if words else 0.0
