"""Second study of the sufficiency gate, on a set nobody has looked at.

The first study (``gate_research.py``) fitted rules on one dataset and read its test split once; the
best held-out result was 73.3 % correct refusals. This one asks which *signals* separate the
questions that have an answer from the near misses, and answers it without touching the data it is
judged on:

1. Signals are computed for every question of the calibration sets (dev splits of the hard and the
   mini datasets), with the real embedding model and, optionally, the real cross-encoder:
   dense cosine, its margin between documents, the best lexical score, the IDF-weighted share of the
   query's terms found in the top passages, the rarest query term missing from them, and the
   reranker's logit and margin.
2. Candidate rules (one signal, two signals, all signals) are scored by five-fold cross-validation
   **on the calibration sets only**; the rule with the best out-of-fold correct refusals at no more
   than 10 % false refusals is chosen *before* the held-out set is opened.
3. The chosen rule is fitted on all calibration data and applied once, without refitting, to
   ``gate-heldout@v2`` (read through the counted test guard, ``--final``). The plain dense-cosine
   rule is reported next to it as the reference.

Targets: at most 10 % false refusals, at least 90 % correct refusals. All labels are unreviewed.

    uv run python eval/scripts/gate_research_v2.py --embed-dir <e5 dir> \
        [--rerank-dir <dir>] [--final]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from retrieval_eval import ROOT, write_config

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.domain import Language, RerankMode, RetrievalMode, RetrievalQuery
from rag_engine.evaluation.dataset import Question, load_dataset, validate_dataset
from rag_engine.evaluation.protocol import TestSplitGuard
from rag_engine.lexical.analysis import analyze
from rag_engine.retrieval.presentation import load_details
from rag_engine.retrieval.sufficiency import calibrate

BASE = ("dense", "margin", "lexical", "cover_idf", "missing_idf")
RERANK = ("rerank_top", "rerank_margin")
MAX_FALSE = 0.10
MIN_CORRECT = 0.90
FOLDS = 5
MIN_TERM = 3


@dataclass(frozen=True, slots=True)
class Sample:
    dataset: str
    question_id: str
    answerable: bool
    values: dict[str, float | None]


def script(text: str) -> str:
    cyr = sum("Ѐ" <= c <= "ԯ" for c in text)
    lat = sum(c.isascii() and c.isalpha() for c in text)
    return "cyr" if cyr > lat else "lat"


def terms(text: str) -> list[str]:
    stems = analyze(text, Language.MIXED).split()
    return [t for t in stems if len(t) >= MIN_TERM or any(c.isdigit() for c in t)]


def document_frequencies(corpus: dict[str, str]) -> tuple[dict[str, int], int]:
    df: dict[str, int] = {}
    pieces = [
        s for text in corpus.values() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()
    ]
    for piece in pieces:
        for t in set(terms(piece)):
            df[t] = df.get(t, 0) + 1
    return df, len(pieces)


def collect(
    container: DefaultContainer,
    config: Path,
    db: Path,
    dataset: str,
    questions: list[Question],
    corpus: dict[str, str],
    *,
    use_rerank: bool,
) -> list[Sample]:
    df, n = document_frequencies(corpus)

    def idf(term: str) -> float:
        return math.log((n + 1) / (df.get(term, 0) + 1))

    samples: list[Sample] = []
    with container.search(config, db) as searcher:
        for q in questions:
            request = RetrievalQuery(
                request_id=searcher.ids.request_id(),
                text=q.question,
                mode=RetrievalMode.HYBRID,
                top_k=10,
                candidate_k=40,
                rerank=RerankMode.ON if use_rerank else RerankMode.OFF,
            )
            outcome = asyncio.run(searcher.retrieval.search(request))
            hits = outcome.result.hits
            details = load_details(searcher.store, outcome.result)
            by_doc: dict[str, float] = {}
            rerank_by_doc: dict[str, float] = {}
            for hit, detail in zip(hits, details, strict=True):
                if hit.scores.dense is not None:
                    by_doc[detail.path] = max(by_doc.get(detail.path, -1.0), hit.scores.dense)
                if hit.scores.rerank is not None:
                    rerank_by_doc[detail.path] = max(
                        rerank_by_doc.get(detail.path, -1e9), hit.scores.rerank
                    )
            dense = sorted(by_doc.values(), reverse=True)
            rer = sorted(rerank_by_doc.values(), reverse=True)
            values: dict[str, float | None] = {
                "dense": dense[0] if dense else 0.0,
                "margin": dense[0] - dense[1] if len(dense) > 1 else 0.0,
                "lexical": max((h.scores.lexical or 0.0 for h in hits), default=0.0),
                "cover_idf": None,
                "missing_idf": None,
                "rerank_top": rer[0] if rer else None,
                "rerank_margin": rer[0] - rer[1] if len(rer) > 1 else None,
            }
            top_text = " ".join(d.text for d in details[:3])
            wanted = set(terms(q.question))
            if wanted and top_text and script(q.question) == script(top_text):
                have = set(terms(top_text))
                total = sum(idf(t) for t in wanted) or 1.0
                values["cover_idf"] = sum(idf(t) for t in wanted & have) / total
                values["missing_idf"] = max((idf(t) for t in wanted - have), default=0.0)
            samples.append(Sample(dataset, q.id, q.answerable, values))
    return samples


def build_workspace_samples(
    embed_dir: Path, rerank_dir: Path | None, dataset_path: Path, splits: set[str], name: str
) -> list[Sample]:
    dataset = load_dataset(dataset_path)
    corpus_dir = ROOT / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.md")}
    if not validate_dataset(dataset, corpus).ok:
        sys.exit(f"{name}: dataset invalid")
    questions = [
        q for q in dataset.questions if q.split in splits and q.review.status != "rejected"
    ]
    container = DefaultContainer()
    with tempfile.TemporaryDirectory(prefix="rag-gate2-") as folder:
        work = Path(folder)
        config, db = work / "rag.toml", work / "rag.db"
        extra = ""
        if rerank_dir is not None:
            from rag_engine.embedding.onnx import model_directory_digest

            extra = (
                '\n[rerank]\nenabled = true\nmodel = "BAAI/bge-reranker-v2-m3-int8"\n'
                f'model_dir = "{rerank_dir.resolve()}"\n'
                f'revision = "{model_directory_digest(rerank_dir)}"\n'
                "pool_size = 20\ntimeout_s = 120.0\nthreads = 4\n"
            )
        write_config(config, embed_dir.resolve(), 'mode = "hybrid"')
        if extra:
            config.write_text(config.read_text() + extra, encoding="utf-8")
        prefix = ["--config", str(config), "--db", str(db)]
        for step in (["source", "add", "corpus", str(corpus_dir)], ["ingest", "corpus", "--json"]):
            if rag([*prefix, *step], container=container) != 0:
                sys.exit(f"rag {step[0]} failed")
        return collect(
            container, config, db, name, questions, corpus, use_rerank=rerank_dir is not None
        )


# ------------------------------------------------------------------------------ the rules


class Rule:
    """A logistic score over some signals, thresholded for at most 10 % false refusals."""

    def __init__(self, features: tuple[str, ...]) -> None:
        self.features = features

    def fit(self, samples: list[Sample]) -> Rule:
        x = self._raw(samples)
        self.fill = np.nanmean(np.where(np.isnan(x), np.nan, x), axis=0)
        self.fill = np.where(np.isnan(self.fill), 0.0, self.fill)
        x = np.where(np.isnan(x), self.fill, x)
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0) + 1e-9
        y = np.array([s.answerable for s in samples])
        z = (x - self.mean) / self.scale
        w, b = np.zeros(z.shape[1]), 0.0
        weight = np.where(y, 1.0, y.sum() / max(1, (~y).sum()))
        for _ in range(2000):
            p = 1 / (1 + np.exp(-(z @ w + b)))
            g = (p - y) * weight
            w -= 0.1 * (z.T @ g / len(z) + 0.05 * w)
            b -= 0.1 * g.mean()
        self.w, self.b = w, b
        scores = self.score(samples)
        self.threshold = calibrate(
            list(scores[y]),
            list(scores[~y]),
            max_false_refusal=MAX_FALSE,
            min_correct_refusal=MIN_CORRECT,
        ).threshold
        return self

    def _raw(self, samples: list[Sample]) -> np.ndarray:
        return np.array(
            [
                [np.nan if s.values[f] is None else s.values[f] for f in self.features]
                for s in samples
            ],
            dtype=np.float64,
        )

    def score(self, samples: list[Sample]) -> np.ndarray:
        x = self._raw(samples)
        x = np.where(np.isnan(x), self.fill, x)
        return np.asarray(1 / (1 + np.exp(-(((x - self.mean) / self.scale) @ self.w + self.b))))

    def refuses(self, samples: list[Sample]) -> np.ndarray:
        return self.score(samples) < self.threshold


def rates(refused: np.ndarray, answerable: np.ndarray) -> dict[str, float | int]:
    yes, no = answerable, ~answerable
    return {
        "false_refusals": int(refused[yes].sum()),
        "answerable": int(yes.sum()),
        "correct_refusals": int(refused[no].sum()),
        "unanswerable": int(no.sum()),
        "false_refusal_rate": float(refused[yes].mean()) if yes.any() else float("nan"),
        "correct_refusal_rate": float(refused[no].mean()) if no.any() else float("nan"),
    }


def cross_validate(features: tuple[str, ...], samples: list[Sample]) -> dict[str, float | int]:
    order = np.random.default_rng(11).permutation(len(samples))
    refused = np.zeros(len(samples), dtype=bool)
    for fold in range(FOLDS):
        held = order[fold::FOLDS]
        train = [samples[i] for i in order if i not in set(held)]
        rule = Rule(features).fit(train)
        refused[held] = rule.refuses([samples[i] for i in held])
    return rates(refused, np.array([s.answerable for s in samples]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument("--rerank-dir", type=Path, default=None)
    parser.add_argument("--final", action="store_true", help="open the held-out set")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ed = ROOT / "eval/datasets"
    calibration = build_workspace_samples(
        args.embed_dir, args.rerank_dir, ed / "retrieval_hard_v1.json", {"dev"}, "hard-dev"
    )
    calibration += build_workspace_samples(
        args.embed_dir, args.rerank_dir, ed / "retrieval_mini_v1.json", {"dev"}, "mini-dev"
    )
    pool = BASE + (RERANK if args.rerank_dir else ())
    candidates = [(f,) for f in pool] + list(itertools.combinations(pool, 2)) + [pool]
    scored = []
    for features in candidates:
        cv = cross_validate(features, calibration)
        ok = cv["false_refusal_rate"] <= MAX_FALSE + 1e-9
        scored.append((features, cv, ok))
    eligible = [c for c in scored if c[2]] or scored
    chosen = max(eligible, key=lambda c: (c[1]["correct_refusal_rate"], -len(c[0])))
    report: dict[str, object] = {
        "calibration": {
            "datasets": ["hard-dev", "mini-dev"],
            "questions": len(calibration),
            "unanswerable": sum(not s.answerable for s in calibration),
        },
        "cross_validated": [
            {"features": list(f), **cv}
            for f, cv, _ in sorted(scored, key=lambda c: -c[1]["correct_refusal_rate"])[:12]
        ],
        "chosen_before_held_out": list(chosen[0]),
        "targets": {"max_false_refusal": MAX_FALSE, "min_correct_refusal": MIN_CORRECT},
        "labels": "unreviewed; drafted with a language model",
    }
    if args.final:
        reads = TestSplitGuard(ROOT / "eval_runs" / "test_access.json").read(
            "gate-heldout@v2", final=True
        )
        held = build_workspace_samples(
            args.embed_dir, args.rerank_dir, ed / "gate_heldout_v2.json", {"test"}, "gate-v2"
        )
        y = np.array([s.answerable for s in held])
        rule = Rule(chosen[0]).fit(calibration)
        baseline = Rule(("dense",)).fit(calibration)
        report["held_out"] = {
            "dataset": "gate-heldout@v2",
            "test_reads": reads,
            "questions": len(held),
            "chosen_rule": {"features": list(chosen[0]), **rates(rule.refuses(held), y)},
            "dense_reference": rates(baseline.refuses(held), y),
            "weights": dict(zip(chosen[0], map(float, rule.w), strict=True)),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cross_validated"}, indent=2))


if __name__ == "__main__":
    main()
