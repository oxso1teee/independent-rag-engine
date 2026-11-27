"""Export an evaluation dataset for human review, and record a person's decisions.

    uv run python eval/scripts/review_sheet.py export --output eval_runs/review.csv
    uv run python eval/scripts/review_sheet.py apply --sheet eval_runs/review.csv \\
        --reviewer "Your Name" --output eval/datasets/retrieval_mini_v2.json

Export never changes the dataset. Apply refuses a reviewer name that looks like a program, leaves
rows without a decision unreviewed and writes a new file; it never overwrites the dataset it read.
Bump ``version`` in the output when the reviewed file replaces the old one (labels are immutable
per version).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rag_engine.domain import RagError
from rag_engine.evaluation.dataset import load_dataset, parse_dataset, validate_dataset
from rag_engine.evaluation.review import apply_review, export_review_sheet

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("action", choices=("export", "apply"))
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "eval/datasets/retrieval_mini_v1.json"
    )
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    corpus_dir = ROOT / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.md")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "export":
        args.output.write_text(export_review_sheet(dataset, corpus), encoding="utf-8")
        print(f"{len(dataset.questions)} questions written to {args.output}")
        return
    if args.sheet is None or not args.reviewer:
        sys.exit("apply needs --sheet and --reviewer")
    if args.output.resolve() == args.dataset.resolve():
        sys.exit(
            "write a new file; the reviewed dataset must not overwrite the one it was read from"
        )
    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    try:
        today = datetime.now().astimezone().date()
        sheet = args.sheet.read_text(encoding="utf-8")
        updated = apply_review(raw, sheet, reviewer=args.reviewer, on=today)
        report = validate_dataset(parse_dataset(updated), corpus)
    except RagError as error:
        sys.exit(error.message)
    if not report.ok:
        sys.exit("the reviewed dataset is invalid:\n" + "\n".join(report.errors))
    args.output.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counted = sum(q["review"]["status"] in {"accepted", "edited"} for q in updated["questions"])
    print(f"written to {args.output}: {counted} accepted or edited questions")
    for warning in report.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
