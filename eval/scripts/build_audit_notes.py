"""Per-question notes for the review page: what version 2 changed, and what the deep audit found.

Three sources, merged into ``eval/datasets/audit_notes_v2.json``:

* the history of changes (``CHANGES_v2.json``): for a question that was corrected, what and why;
* ``READINGS`` below: the deep audit's reading of the questions that still carry a doubt;
* the audit report: for an unanswerable question whose nearest passage scores high, the passage that
  was checked.

A note is an assistant's reading of the evidence. It is shown to the reviewer as such, is never a
review, and changes no dataset.

    uv run python eval/scripts/build_audit_notes.py --audit eval_runs/question_audit_v2_semantic.json
"""

# ruff: noqa: E501  (the notes are sentences)
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARD, MINI = "retrieval_hard_v2", "retrieval_mini_v2"

READINGS: dict[str, str] = {
    f"{HARD}:en_vacation_v1-1-same": "A hard negative by design: the version 2 passage scores 0.887 against 0.894. The question says 'version 1' and the evidence is the version 1 document, so the label is right; only retrieval can be confused.",
    f"{HARD}:en_vacation_v1-1-cross": "Same pair as the English twin: the question asks for version 1, the evidence is version 1. The version 2 passage is the designed distractor. Label right.",
    f"{HARD}:en_onboarding-1-same": "The laptop-backup passage scores 0.877 against 0.884 because it also says 'laptops', but it is about backups, not about when a laptop is issued. Label right.",
    f"{HARD}:en_release-1-same": "Word overlap favours a backup passage ('night', 'time'); the embedding ranks the release document first (0.881 against 0.817). The evidence 'every Tuesday at 14:00 UTC' answers the question. Label right.",
    f"{HARD}:ru_travel-2-cross": "The per diem (суточные, 1200 rubles) is stated only in ru_travel; en_travel and en_meals give limits for hotels and meals, not a per diem. The question now says per diem in roubles. Label right; the margin is small only in the model's scores.",
    f"{HARD}:en_remote_2023-1-same": "Designed pair: the 2023 and 2024 policies. The question names 2023 and the evidence is the 2023 document. Label right.",
    f"{HARD}:en_remote_2023-1-cross": "Same pair as the English twin; the Russian question names 2023. Label right.",
    f"{HARD}:en_remote_2024-1-same": "Not a duplicate of conflicting-8: this question names 2024 and has one gold document; conflicting-8 has no year and expects both versions. The similarity is intended.",
    f"{HARD}:conflicting-8": "Not a duplicate of en_remote_2024-1-same: no year is given, so both the 2023 and the 2024 rule are needed, and the expected answer says which replaces which.",
    f"{HARD}:en_remote_2024-2-same": "Asks which policy replaces the 2023 one. The 2023 document ranks first because it contains '2023', but only the 2024 document says it replaces it. Label right.",
    f"{HARD}:en_remote_2024-2-cross": "Same as the English twin: the 2023 document ranks first on the word '2023', the replacement is stated only in the 2024 document. Label right.",
    f"{MINI}:en-k3": "The scores of the three nearest passages differ by 0.004, and none of the other two is about paging; the evidence says exactly 'the on-call engineer must be paged within 5 minutes'. Label right; the embedding is weak on very short questions.",
    f"{MINI}:en-p5": "'Write up what went wrong' means the postmortem, 'published within 5 business days' in en_incident_response; the embedding ranks it first. Label right.",
    f"{MINI}:en-p7": "'Login device' means the hardware security token of en_key_management; the competing passage only shares words. Label right.",
    f"{MINI}:ru-p7": "'Учётные данные' here means the password; ru_parole forbids sending a password by mail, so 'Нет, запрещено' is supported. Label right.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "eval/datasets/audit_notes_v2.json")
    args = parser.parse_args()
    notes: dict[str, str] = {}
    changes = json.loads((ROOT / "eval/datasets/CHANGES_v2.json").read_text(encoding="utf-8"))
    names = {"retrieval-hard": HARD, "retrieval-mini": MINI}
    for dataset, block in changes["datasets"].items():
        for change in block["changes"]:
            notes[f"{names[dataset]}:{change['id']}"] = (
                f"Changed in version 2 ({change['kind']}): {change['reason']}."
            )
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    for audit in report["audits"]:
        for flag in audit["flags"]:
            if flag["id"] == "dense-near-miss":
                found = re.match(r"(\S+) scores ([\d.]+)", flag["message"])
                where = f"{found.group(1)} ({found.group(2)})" if found else "another document"
                notes.setdefault(
                    audit["key"],
                    f"A near-miss topic, checked against the whole corpus: the nearest passage is {where} and it answers something else; the absence terms, in both languages, occur nowhere.",
                )
    for key, text in READINGS.items():
        notes[key] = text + (f" {notes[key]}" if key in notes else "")
    args.output.write_text(
        json.dumps(dict(sorted(notes.items())), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{len(notes)} notes written to {args.output}")


if __name__ == "__main__":
    main()
