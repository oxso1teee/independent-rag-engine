"""Research on the sufficiency gate: which signal separates answerable from unanswerable questions.

The gate decides before the model is called, from retrieval signals only. This script collects the
signals of every question of the hard dataset with the real embedding model, **fits every rule on
the dev split only**, and applies the fitted rule to the test split exactly once (counted through
the same guard as ``rag eval``: pass ``--final``). Rules compared:

* dense: the best dense cosine among the hits (the rule of the first calibration);
* coverage: the lexical coverage of the top passage by the query's terms;
* margin: the gap between the best and the second-best document by dense cosine;
* and-or: a hit on either the dense or the coverage rule (as in the shipped gate);
* logistic: a logistic regression over all signals, thresholded for at most 10 % false refusals.

Targets: false refusals <= 10 % and correct refusals >= 90 %. With unreviewed labels and 15
unanswerable questions per split, the result is exploratory and cannot settle the criterion: one
question is 6.7 percentage points.

    uv run python eval/scripts/gate_research.py --embed-dir <e5 dir> [--final] \
        --output eval_runs/gate_research.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from retrieval_eval import ROOT, write_config

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.domain import RerankMode, RetrievalMode, RetrievalQuery
from rag_engine.evaluation.dataset import Question, load_dataset, validate_dataset
from rag_engine.evaluation.protocol import TestSplitGuard
from rag_engine.retrieval.presentation import load_details
from rag_engine.retrieval.sufficiency import calibrate, lexical_coverage

FEATURES = ("dense", "coverage", "margin", "lexical")
MAX_FALSE_REFUSAL = 0.10
MIN_CORRECT_REFUSAL = 0.90


@dataclass(frozen=True, slots=True)
class Sample:
    question_id: str
    split: str
    answerable: bool
    values: dict[str, float]


def collect(
    container: DefaultContainer, config: Path, db: Path, questions: list[Question]
) -> list[Sample]:
    samples: list[Sample] = []
    with container.search(config, db) as searcher:
        for question in questions:
            request = RetrievalQuery(
                request_id=searcher.ids.request_id(), text=question.question,
                mode=RetrievalMode.HYBRID, top_k=10, candidate_k=40, rerank=RerankMode.OFF,
            )  # fmt: skip
            outcome = asyncio.run(searcher.retrieval.search(request))
            hits = outcome.result.hits
            details = load_details(searcher.store, outcome.result)
            dense_by_doc: dict[str, float] = {}
            for hit, detail in zip(hits, details, strict=True):
                if hit.scores.dense is not None:
                    dense_by_doc[detail.path] = max(
                        dense_by_doc.get(detail.path, -1.0), hit.scores.dense
                    )
            ranked = sorted(dense_by_doc.values(), reverse=True)
            top_text = details[0].text if details else ""
            coverage = lexical_coverage(question.question, top_text, outcome.diagnostics.language)
            lexical = max((h.scores.lexical or 0.0 for h in hits), default=0.0)
            samples.append(
                Sample(
                    question.id,
                    question.split,
                    question.answerable,
                    {
                        "dense": ranked[0] if ranked else 0.0,
                        "coverage": coverage or 0.0,
                        "margin": (ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0,
                        "lexical": lexical,
                    },
                )
            )
    return samples


def rates(refused: np.ndarray, answerable: np.ndarray) -> dict[str, float | int] | None:
    if answerable.size == 0:
        return None  # the split was not read
    yes, no = answerable, ~answerable
    return {
        "false_refusal_rate": float(refused[yes].mean()) if yes.any() else float("nan"),
        "correct_refusal_rate": float(refused[no].mean()) if no.any() else float("nan"),
        "answerable": int(yes.sum()),
        "unanswerable": int(no.sum()),
        "false_refusals": int(refused[yes].sum()),
        "correct_refusals": int(refused[no].sum()),
    }


def matrix(samples: list[Sample], names: tuple[str, ...] = FEATURES) -> np.ndarray:
    return np.array([[s.values[n] for n in names] for s in samples], dtype=np.float64)


def fit_threshold(scores: np.ndarray, answerable: np.ndarray) -> float:
    """Threshold fitted on dev: refuse below it, most correct refusals within the allowance."""
    fit = calibrate(
        list(scores[answerable]), list(scores[~answerable]),
        max_false_refusal=MAX_FALSE_REFUSAL, min_correct_refusal=MIN_CORRECT_REFUSAL,
    )  # fmt: skip
    return fit.threshold


def logistic(
    train_x: np.ndarray, train_y: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-9
    x = (train_x - mean) / scale
    w, b = np.zeros(x.shape[1]), 0.0
    weights = np.where(train_y, 1.0, (train_y.sum() / max(1, (~train_y).sum())))  # balance classes
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        gradient = (p - train_y) * weights
        w -= 0.1 * (x.T @ gradient / len(x) + 0.01 * w)
        b -= 0.1 * gradient.mean()
    return w, b, mean, scale


def cross_validate(x: np.ndarray, y: np.ndarray, folds: int = 5) -> dict[str, object]:
    """Out-of-sample estimate on dev alone: each fold is scored by a rule fitted on the others."""
    refused = np.zeros(len(y), dtype=bool)
    order = np.argsort(np.arange(len(y)) % folds, kind="stable")
    assignment = np.empty(len(y), dtype=int)
    assignment[order] = np.arange(len(y)) % folds
    for fold in range(folds):
        train, held = assignment != fold, assignment == fold
        w, b, mean, scale = logistic(x[train], y[train])

        def probability(
            rows: np.ndarray,
            w: np.ndarray = w,
            b: float = b,
            mean: np.ndarray = mean,
            scale: np.ndarray = scale,
        ) -> np.ndarray:
            return np.asarray(1 / (1 + np.exp(-(((rows - mean) / scale) @ w + b))))

        cut = fit_threshold(probability(x[train]), y[train])
        refused[held] = probability(x[held]) < cut
    return {"cross_validated": rates(refused, y)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "eval/datasets/retrieval_hard_v1.json"
    )
    parser.add_argument("--final", action="store_true", help="also read the held-out test split")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    corpus_dir = ROOT / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.md")}
    if not validate_dataset(dataset, corpus).ok:
        sys.exit("dataset invalid")
    questions = [q for q in dataset.questions if q.review.status != "rejected"]
    dev_questions = [q for q in questions if q.split == "dev"]
    test_questions = [q for q in questions if q.split == "test"]
    # Without --final only dev is touched, so the rules can be developed freely; with it the test
    # split is read, counted, and the report says whether it was ever read before.
    reads = None
    if args.final:
        reads = TestSplitGuard(ROOT / "eval_runs" / "test_access.json").read(
            dataset.identifier, final=True
        )

    container = DefaultContainer()
    with tempfile.TemporaryDirectory(prefix="rag-gate-") as folder:
        work = Path(folder)
        config, db = work / "rag.toml", work / "rag.db"
        write_config(config, args.embed_dir.resolve(), 'mode = "hybrid"')
        prefix = ["--config", str(config), "--db", str(db)]
        for step in (["source", "add", "corpus", str(corpus_dir)], ["ingest", "corpus", "--json"]):
            if rag([*prefix, *step], container=container) != 0:
                sys.exit(f"rag {step[0]} failed")
        dev = collect(container, config, db, dev_questions)
        test = collect(container, config, db, test_questions) if args.final else []

    dev_y = np.array([s.answerable for s in dev])
    test_y = np.array([s.answerable for s in test], dtype=bool)
    dev_x = matrix(dev)
    test_x = matrix(test) if test else np.zeros((0, len(FEATURES)))
    results: dict[str, object] = {}

    thresholds: dict[str, float] = {}
    for index, name in enumerate(FEATURES):
        thresholds[name] = fit_threshold(dev_x[:, index], dev_y)
        results[f"single:{name}"] = {
            "threshold": thresholds[name],
            "dev": rates(dev_x[:, index] < thresholds[name], dev_y),
            "test": rates(test_x[:, index] < thresholds[name], test_y),
        }

    # shipped policy: accept when either the dense or the coverage rule accepts
    def either(x: np.ndarray) -> np.ndarray:
        return (x[:, 0] >= thresholds["dense"]) | (x[:, 1] >= thresholds["coverage"])

    results["and-or"] = {
        "dev": rates(~either(dev_x), dev_y),
        "test": rates(~either(test_x), test_y),
    }
    w, b, mean, scale = logistic(dev_x, dev_y)

    def score(x: np.ndarray) -> np.ndarray:
        return np.asarray(1 / (1 + np.exp(-(((x - mean) / scale) @ w + b))))

    cut = fit_threshold(score(dev_x), dev_y)
    results["logistic"] = {
        "weights": dict(zip(FEATURES, map(float, w), strict=True)), "threshold": cut,
        "dev": rates(score(dev_x) < cut, dev_y), "test": rates(score(test_x) < cut, test_y),
    }  # fmt: skip
    results["logistic:dev-cross-validated"] = cross_validate(dev_x, dev_y)
    report = {
        "dataset": dataset.identifier, "exploratory": True,
        "labels": "unreviewed (drafted by the assistant); not human-reviewed gold data",
        "test_reads": reads, "held_out": None if reads is None else reads <= 1,
        "test_split": "read (--final)" if args.final else "not read",
        "targets": {
            "max_false_refusal": MAX_FALSE_REFUSAL,
            "min_correct_refusal": MIN_CORRECT_REFUSAL,
        },
        "rules": results,
        "resolution": "one unanswerable question is %.1f percentage points"
        % (100 / max(1, int((~(test_y if test else dev_y)).sum()))),
    }  # fmt: skip
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
