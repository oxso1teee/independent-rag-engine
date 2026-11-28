"""One review kit for the evaluation questions: build it, review in a browser, apply the result.

    uv run python eval/scripts/review_kit.py build --output eval_runs/review_kit/review_kit.html
    # open the HTML file, decide, press "Export decisions (CSV)"
    uv run python eval/scripts/review_kit.py apply --sheet ~/Downloads/review_decisions.csv \\
        --reviewer "Your Name" --outdir eval/datasets

``build`` changes no dataset. ``apply`` writes new versions next to the old ones (never over them)
and records only the decisions that are in the sheet, under the name given with ``--reviewer``.
By default the kit holds the two datasets that are awaiting review (``retrieval_hard_v1``, 141
questions, and ``retrieval_mini_v1``, 44); ``--dataset`` adds or replaces them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rag_engine.domain import RagError
from rag_engine.evaluation.review_kit import apply_kit_sheet, build_kit, load_source

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ("retrieval_hard_v2", "retrieval_mini_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "apply"))
    parser.add_argument("--dataset", action="append", type=Path, help="a dataset file (repeatable)")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "eval_runs/review_kit/review_kit.html"
    )
    parser.add_argument(
        "--audit", type=Path, help="a report of audit_questions.py: adds the audit column and table"
    )
    parser.add_argument(
        "--notes",
        type=Path,
        help="a JSON file of key -> note (a deep audit's reading per question)",
    )
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--outdir", type=Path, default=ROOT / "eval" / "datasets")
    args = parser.parse_args()
    paths = args.dataset or [ROOT / "eval/datasets" / f"{name}.json" for name in DEFAULT]
    try:
        sources = [load_source(path, ROOT) for path in paths]
        if args.action == "build":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            audits = None
            if args.audit is not None:
                report = json.loads(args.audit.read_text(encoding="utf-8"))
                audits = {item["key"]: item for item in report["audits"]}
            notes = None
            if args.notes is not None:
                notes = json.loads(args.notes.read_text(encoding="utf-8"))
            args.output.write_text(build_kit(sources, audits=audits, notes=notes), encoding="utf-8")
            total = sum(len(s.dataset.questions) for s in sources)
            print(f"{total} questions from {len(sources)} datasets written to {args.output}")
            return
        if args.sheet is None or not args.reviewer:
            sys.exit("apply needs --sheet and --reviewer")
        today = datetime.now().astimezone().date()
        for item in apply_kit_sheet(
            sources,
            args.sheet.read_text(encoding="utf-8"),
            reviewer=args.reviewer,
            outdir=args.outdir,
            on=today,
        ):
            print(
                f"{item.path.name}: {item.accepted} accepted, {item.edited} edited, "
                f"{item.rejected} rejected, {item.undecided} still unreviewed"
            )
    except RagError as error:
        sys.exit(error.message)


if __name__ == "__main__":
    main()
