"""Calibrate the sufficiency gate on the dev split and check it on the test split.

For every question the best dense cosine among the retrieved hits is the signal. The threshold is
chosen on dev so that at most 10 % of answerable questions are refused while as many unanswerable
ones as possible are; the same threshold is then applied to test without refitting. Like the
retrieval evaluation, an unreviewed dataset only gives an exploratory result.

    uv run python eval/scripts/gate_calibration.py --model-dir <e5 model directory>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from retrieval_eval import ROOT, write_config

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.domain import RetrievalMode, RetrievalQuery
from rag_engine.evaluation.dataset import Question, load_dataset, validate_dataset
from rag_engine.retrieval.sufficiency import calibrate


def best_dense_scores(
    container: DefaultContainer, config: Path, db: Path, questions: list[Question]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    with container.search(config, db) as searcher:
        for question in questions:
            request = RetrievalQuery(
                request_id=searcher.ids.request_id(),
                text=question.question,
                mode=RetrievalMode.DENSE,
                top_k=10,
                candidate_k=40,
            )
            hits = asyncio.run(searcher.retrieval.search(request)).result.hits
            scores[question.id] = max((h.scores.dense or 0.0 for h in hits), default=0.0)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "eval/datasets/retrieval_mini_v1.json"
    )
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    corpus_dir = ROOT / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.md")}
    if not validate_dataset(dataset, corpus).ok:
        sys.exit("dataset invalid")
    usable = [
        q
        for q in dataset.questions
        if q.review.status in {"accepted", "edited"}
        or (args.include_unreviewed and q.review.status != "rejected")
    ]
    if not usable:
        sys.exit("no accepted questions; review the dataset or use --include-unreviewed")

    container = DefaultContainer()
    with tempfile.TemporaryDirectory(prefix="rag-gate-") as folder:
        work = Path(folder)
        config, db = work / "rag.toml", work / "rag.db"
        write_config(config, args.model_dir.resolve(), 'mode = "dense"')
        prefix = ["--config", str(config), "--db", str(db)]
        for step in (["source", "add", "corpus", str(corpus_dir)], ["ingest", "corpus", "--json"]):
            if rag([*prefix, *step], container=container) != 0:
                sys.exit(f"rag {step[0]} failed")
        scores = best_dense_scores(container, config, db, usable)

    def split(name: str, answerable: bool) -> list[float]:
        return [scores[q.id] for q in usable if q.split == name and q.answerable is answerable]

    dev_yes, dev_no = split("dev", True), split("dev", False)
    fit = calibrate(dev_yes, dev_no)
    test_yes, test_no = split("test", True), split("test", False)
    held_out = (
        {
            "false_refusal_rate": sum(s < fit.threshold for s in test_yes) / len(test_yes),
            "correct_refusal_rate": sum(s < fit.threshold for s in test_no) / len(test_no),
            "answerable": len(test_yes),
            "unanswerable": len(test_no),
        }
        if test_yes and test_no
        else None
    )
    report = {
        "dataset": dataset.identifier,
        "exploratory": args.include_unreviewed,
        "signal": "best dense cosine among the hits",
        "dev": {
            "threshold": fit.threshold,
            "false_refusal_rate": fit.false_refusal_rate,
            "correct_refusal_rate": fit.correct_refusal_rate,
            "answerable": fit.positives,
            "unanswerable": fit.negatives,
            "meets_target": fit.meets_target,
        },
        "test_with_the_dev_threshold": held_out,
        "answerable_scores": {"min": min(dev_yes + test_yes), "max": max(dev_yes + test_yes)},
        "unanswerable_scores": {"min": min(dev_no + test_no), "max": max(dev_no + test_no)},
        "note": "fewer than 30 questions per class is anecdotal",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
