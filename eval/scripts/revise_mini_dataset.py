"""Version 2 of the small retrieval dataset: the corrections found by the question audit.

The first version was drafted by hand and has no generator, so its corrections are listed here as
data and applied to the frozen file; ``dataset_revision.revise`` records every change with its
reason in ``eval/datasets/CHANGES_v2.json``. Nothing in this file marks a question as reviewed.

    uv run python eval/scripts/revise_mini_dataset.py
"""

# ruff: noqa: E501  (reasons and terms are sentences)
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_revision import record, revise

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "eval" / "datasets" / "retrieval_mini_v1.json"
DATASET = ROOT / "eval" / "datasets" / "retrieval_mini_v2.json"
CHANGES = ROOT / "eval" / "datasets" / "CHANGES_v2.json"
VERSION = 2

# A multi-document answer is checked fact by fact: one fact per evidence quote, not one sentence.
FACTS: dict[str, list[str]] = {
    "en-m1": ["Daily backups 14 days", "application logs 30 days"],
    "en-m2": [
        "The on-call engineer within 5 minutes",
        "the release rolls back automatically above 2 percent errors",
    ],
    "en-m3": ["Backup copies are encrypted with AES-256", "exports are available for 7 days"],
    "ru-m1": ["Права раз в полгода", "восстановление раз в квартал"],
    "ru-m2": ["60 тысяч рублей в год на обучение", "суточные 1000 рублей в день"],
    "ru-m3": [
        "По понедельникам в 10:00",
        "в праздничные дни отключаются оповещения низкой важности",
    ],
}

# The corpus mixes English and Russian documents (and one mixed one), so "absent" must hold in both.
TERMS: dict[str, list[str]] = {
    "en-n1": ["собак"],
    "en-n2": ["премия", "премии"],
    "en-n3": ["облачн", "хостинг"],
    "en-n4": ["декрет", "родительск"],
    "ru-n1": ["bonus"],
    "ru-n2": ["canteen", "cafeteria"],
    "ru-n3": ["язык программирования", "бэкенд", "programming language", "backend"],
    "ru-n4": ["parental", "maternity"],
}

REASONS: dict[str, tuple[str, str]] = {
    **dict.fromkeys(
        FACTS,
        (
            "facts",
            "the expected answer joins two evidence quotes, so it is now listed as one fact per quote",
        ),
    ),
    **dict.fromkeys(
        TERMS,
        (
            "absence",
            "absence terms now cover both languages, because the corpus has documents in both",
        ),
    ),
}


def build() -> dict[str, object]:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    data = copy.deepcopy(previous)
    data["version"] = VERSION
    for question in data["questions"]:
        if question["id"] in FACTS:
            question["reference_facts"] = FACTS[question["id"]]
        if question["id"] in TERMS:
            question["absence_terms"] = [*question["absence_terms"], *TERMS[question["id"]]]
    return data  # type: ignore[no-any-return]


def main() -> None:
    from rag_engine.evaluation.dataset import parse_dataset, validate_dataset
    from rag_engine.evaluation.question_audit import Verdict, audit_source
    from rag_engine.evaluation.review_kit import SourceDataset

    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    dataset = build()
    changes = revise(previous, dataset, REASONS)
    folder = ROOT / str(dataset["corpus"])
    corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}
    parsed = parse_dataset(dataset)
    report = validate_dataset(parsed, corpus)
    if not report.ok:
        raise SystemExit("the dataset does not validate:\n" + "\n".join(report.errors))
    wrong = [
        a
        for a in audit_source(SourceDataset("retrieval_mini_v2", dataset, parsed, corpus))
        if a.verdict is Verdict.INCORRECT
    ]
    if wrong:
        raise SystemExit("wrong on their face:\n" + "\n".join(f"{a.key}: {a.hint}" for a in wrong))
    DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record(CHANGES, "retrieval-mini", previous["version"], VERSION, "2026-11-28", changes)
    print(f"{len(dataset['questions'])} questions, {len(changes)} changed")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
