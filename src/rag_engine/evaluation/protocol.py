"""The evaluation protocol: manifests, the guarded test split, comparison, reports and baselines.

A run is a JSON document: a manifest that says exactly what was measured and with what, the
per-question scores (so any two runs can be compared pairwise) and a summary with 95 % intervals.
The test split is guarded: it is read only with an explicit ``final`` request, every read is
counted, and a second read is reported as such, because numbers computed on a split that has been
looked at repeatedly are no longer held-out. A baseline is a run kept in the repository; the
regression gate fails a change that makes a metric worse by more than a margin, judged on the
paired interval when the same questions were scored and on absolute floors otherwise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_engine.domain import ErrorCode, RagError
from rag_engine.evaluation.bootstrap import bootstrap_mean, paired_bootstrap
from rag_engine.evaluation.runner import QuestionScore
from rag_engine.evaluation.stats import (
    holm_adjust,
    paired_binary,
    sample_label,
    saturation,
    sign_test,
)

SCHEMA = 1
DEFAULT_DELTA = 0.03
RESAMPLES = 2000


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_for(dataset_manifest_hash: str) -> int:
    """The bootstrap seed of a dataset version: the start of its hash."""
    return int(dataset_manifest_hash[:8], 16)


# ----------------------------------------------------------------------------------- manifest


@dataclass(frozen=True, slots=True)
class Manifest:
    mode: str
    split: str
    final: bool
    dataset: str
    dataset_hash: str
    corpus_hash: str
    config_hash: str
    embedding_model: str | None
    llm: str | None
    judge: str | None
    context_window: int | None
    index_generation: int | None
    seed: int
    engine_version: str
    git_commit: str | None
    resources: Mapping[str, float]
    exploratory: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "split": self.split, "final": self.final,
            "dataset": self.dataset, "dataset_hash": self.dataset_hash,
            "corpus_hash": self.corpus_hash, "config_hash": self.config_hash,
            "embedding_model": self.embedding_model, "llm": self.llm, "judge": self.judge,
            "context_window": self.context_window, "index_generation": self.index_generation,
            "seed": self.seed, "engine_version": self.engine_version,
            "git_commit": self.git_commit, "resources": dict(self.resources),
            "exploratory": self.exploratory, "created_at": self.created_at,
        }  # fmt: skip


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------------- test-split guard


class TestSplitGuard:
    """Counts every read of a dataset's ``test`` split and refuses one that was not requested."""

    __test__ = False  # not a pytest class

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self, dataset: str, *, final: bool) -> int:
        """Record a read of the test split and return how many reads it has now had."""
        if not final:
            raise RagError(
                ErrorCode.E_QUERY_INVALID,
                "the test split is held out: pass --final to read it, once, at the end",
            )
        counts = self._load()
        counts[dataset] = counts.get(dataset, 0) + 1
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return counts[dataset]

    def reads(self, dataset: str) -> int:
        return self._load().get(dataset, 0)

    def _load(self) -> dict[str, int]:
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except ValueError:
            raise RagError(ErrorCode.E_INTEGRITY, "the test-access counter is unreadable") from None
        return {str(k): int(v) for k, v in data.items()}


# --------------------------------------------------------------------------------------- run


def build_run(
    manifest: Manifest,
    scores: Sequence[QuestionScore],
    metrics: Sequence[str],
    *,
    test_reads: int | None = None,
) -> dict[str, Any]:
    """The JSON document of a run: manifest, per-question scores and the summary."""
    groups: dict[str, list[QuestionScore]] = {"all": list(scores)}
    for item in scores:
        groups.setdefault(f"type:{item.type}", []).append(item)
        groups.setdefault(f"language:{item.language}", []).append(item)
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    saturated: list[str] = []
    for name, members in sorted(groups.items()):
        summary[name] = {}
        for metric in metrics:
            interval = bootstrap_mean(
                [m.metrics[metric] for m in members], seed=manifest.seed, resamples=RESAMPLES
            )
            summary[name][metric] = {
                "n": len(members),
                "mean": interval.mean,
                "low": interval.low,
                "high": interval.high,
            }
            if name == "all" and saturation(interval.mean, interval.high):
                saturated.append(metric)
    flags: dict[str, Any] = {
        "exploratory": manifest.exploratory,
        "sample": sample_label(len(scores)),
        "saturated_metrics": saturated,
    }
    if test_reads is not None:
        flags["test_reads"] = test_reads
        flags["held_out"] = test_reads <= 1
    return {
        "schema": SCHEMA,
        "manifest": manifest.to_dict(),
        "metrics": list(metrics),
        "questions": [
            {"id": s.question_id, "type": s.type, "language": s.language, "metrics": s.metrics}
            for s in scores
        ],
        "summary": summary,
        "flags": flags,
    }


def load_run(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, f"cannot read the run file {path.name}"
        ) from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or "questions" not in data:
        raise RagError(ErrorCode.E_CONFIG_INVALID, f"{path.name} is not a run of this version")
    return data


def write_run(run: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")


# ------------------------------------------------------------------------------- comparison


@dataclass(frozen=True, slots=True)
class Row:
    metric: str
    mean_diff: float
    low: float
    high: float
    sign_p: float
    sign_p_holm: float
    mcnemar_p: float | None
    significant: bool


def _scores(run: Mapping[str, Any], metric: str) -> dict[str, float]:
    return {q["id"]: float(q["metrics"][metric]) for q in run["questions"]}


def compare_runs(
    first: Mapping[str, Any], second: Mapping[str, Any], metrics: Sequence[str] | None = None
) -> list[Row]:
    """Paired comparison ``first - second`` over the questions both runs scored."""
    wanted = list(metrics or [m for m in first["metrics"] if m in second["metrics"]])
    seed = int(first["manifest"]["seed"])
    rows: list[tuple[str, float, float, float, float, float | None]] = []
    for metric in wanted:
        one, two = _scores(first, metric), _scores(second, metric)
        ids = sorted(one.keys() & two.keys())
        if not ids:
            raise ValueError("the runs share no questions")
        a, b = [one[i] for i in ids], [two[i] for i in ids]
        interval = paired_bootstrap(a, b, seed=seed, resamples=RESAMPLES)
        binary = all(v in (0.0, 1.0) for v in a + b)
        mcnemar = (
            paired_binary([v == 1.0 for v in a], [v == 1.0 for v in b]).p_value if binary else None
        )
        rows.append(
            (metric, interval.mean, interval.low, interval.high,
             sign_test([x - y for x, y in zip(a, b, strict=True)]), mcnemar)
        )  # fmt: skip
    adjusted = holm_adjust([r[4] for r in rows])
    return [
        Row(m, d, lo, hi, p, adj, mc, (lo > 0 or hi < 0) and adj < 0.05)
        for (m, d, lo, hi, p, mc), adj in zip(rows, adjusted, strict=True)
    ]


# ---------------------------------------------------------------------------------- reports


def render_report(run: Mapping[str, Any]) -> str:
    manifest, flags = run["manifest"], run["flags"]
    lines = [
        f"# Evaluation report: {manifest['dataset']} ({manifest['mode']}, {manifest['split']})",
        "",
        f"- sample: {flags['sample']}",
        f"- labels: {'EXPLORATORY (unreviewed labels)' if flags['exploratory'] else 'reviewed'}",
        f"- embedding model: {manifest['embedding_model']}; llm: {manifest['llm']}; "
        f"judge: {manifest['judge']}",
        f"- dataset hash: `{manifest['dataset_hash'][:12]}`, corpus hash: "
        f"`{manifest['corpus_hash'][:12]}`, seed {manifest['seed']}, "
        f"engine {manifest['engine_version']}, commit {manifest['git_commit'] or 'unknown'}",
    ]
    if "test_reads" in flags:
        lines.append(
            f"- test split reads: {flags['test_reads']} "
            f"({'held out' if flags['held_out'] else 'NOT held out any more: read repeatedly'})"
        )
    if flags["saturated_metrics"]:
        lines.append(
            "- **saturated** (improvements cannot be measured): "
            + ", ".join(flags["saturated_metrics"])
        )
    lines += ["", "| group | metric | n | mean | 95 % interval |", "|---|---|---:|---:|---|"]
    for group, metrics in run["summary"].items():
        for metric, value in metrics.items():
            note = " (anecdotal)" if value["n"] < 30 else ""
            lines.append(
                f"| {group} | {metric} | {value['n']}{note} | {value['mean']:.3f} | "
                f"[{value['low']:.3f}, {value['high']:.3f}] |"
            )
    return "\n".join(lines) + "\n"


def render_comparison(rows: Sequence[Row], first: str, second: str) -> str:
    lines = [
        f"# {first} minus {second}",
        "",
        "| metric | difference | 95 % interval | sign p (Holm) | McNemar p | verdict |",
        "|---|---:|---|---:|---:|---|",
    ]
    for r in rows:
        mc = "-" if r.mcnemar_p is None else f"{r.mcnemar_p:.3f}"
        verdict = "significant" if r.significant else "no evidence of a difference"
        lines.append(
            f"| {r.metric} | {r.mean_diff:+.3f} | [{r.low:+.3f}, {r.high:+.3f}] | "
            f"{r.sign_p_holm:.3f} | {mc} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------- baselines


@dataclass(frozen=True, slots=True)
class Violation:
    metric: str
    reason: str


def check_regression(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    delta: float = DEFAULT_DELTA,
    metrics: Sequence[str] | None = None,
) -> list[Violation]:
    """Metrics that got worse than the baseline by more than ``delta``.

    An absolute floor always applies: the mean must stay above the baseline mean minus ``delta``.
    When the same questions were scored, the paired interval also decides: a change that lowers
    the metric fails when the lower bound of the paired difference is below ``-delta``.
    """
    wanted = list(metrics or [m for m in baseline["metrics"] if m in current["metrics"]])
    same = {q["id"] for q in current["questions"]} == {q["id"] for q in baseline["questions"]}
    violations: list[Violation] = []
    paired = {r.metric: r for r in compare_runs(current, baseline, wanted)} if same else {}
    for metric in wanted:
        now = current["summary"]["all"][metric]["mean"]
        before = baseline["summary"]["all"][metric]["mean"]
        if now < before - delta:
            violations.append(
                Violation(metric, f"mean {now:.3f} is below the floor {before - delta:.3f}")
            )
        row = paired.get(metric)
        if row is not None and row.mean_diff < 0 and row.low < -delta:
            violations.append(
                Violation(metric, f"paired lower bound {row.low:+.3f} is below -{delta:.2f}")
            )
    return violations
