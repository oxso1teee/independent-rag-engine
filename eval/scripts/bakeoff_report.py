"""Summarise the NoMIRACL judged-pool bake-off with paired bootstrap intervals.

Reads ``results-*.json`` (and ``timing-*.json`` when present) written by
``bakeoff_nomiracl.py`` and prints Markdown. Resampling is seeded, and every model uses the same
resampled query indices, so differences are paired and the output is reproducible.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(__file__).resolve().parents[2] / "data/eval/nomiracl"
MODELS = ("e5-small", "qwen3-embedding-0.6b", "qwen3-embedding-4b")
METRICS = ("recall_at_1", "recall_at_5", "mrr_at_10", "ndcg_at_10")
RESAMPLES = 10_000
SEED = 20261126


def per_query(result: dict[str, Any], metric: str, lang: str | None) -> np.ndarray:
    langs = [lang] if lang else ["english", "russian"]
    return np.asarray(
        [
            value[metric]
            for item in langs
            for _, value in sorted(result["metrics"][item]["per_query"].items())
        ]
    )


def interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    results = {
        name: json.loads((BASE / f"results-{name}.json").read_text())
        for name in MODELS
        if (BASE / f"results-{name}.json").exists()
    }
    rng = np.random.default_rng(SEED)
    print("## Ranking quality among judged passages (mean, 95% bootstrap interval)\n")
    for scope, lang in (("English", "english"), ("Russian", "russian"), ("Pooled", None)):
        count = len(per_query(next(iter(results.values())), "ndcg_at_10", lang))
        indices = rng.integers(0, count, size=(RESAMPLES, count))
        print(f"### {scope} ({count} queries)\n")
        print("| Model | " + " | ".join(METRICS) + " |")
        print("|---|" + "---:|" * len(METRICS))
        for name, result in results.items():
            cells = []
            for metric in METRICS:
                values = per_query(result, metric, lang)
                low, high = interval(values[indices].mean(axis=1))
                cells.append(f"{values.mean():.3f} [{low:.3f}, {high:.3f}]")
            print(f"| {name} | " + " | ".join(cells) + " |")
        print("\nPaired difference of the first model minus the second:\n")
        print("| Pair | nDCG@10 diff [95% CI] | Recall@5 diff [95% CI] |")
        print("|---|---:|---:|")
        for first, second in combinations(results, 2):
            cells = []
            for metric in ("ndcg_at_10", "recall_at_5"):
                delta = per_query(results[first], metric, lang) - per_query(
                    results[second], metric, lang
                )
                low, high = interval(delta[indices].mean(axis=1))
                mark = " *" if low > 0 or high < 0 else ""
                cells.append(f"{delta.mean():+.3f} [{low:+.3f}, {high:+.3f}]{mark}")
            print(f"| {first} − {second} | " + " | ".join(cells) + " |")
        print("\n`*` the interval excludes zero.\n")
    timings = [json.loads(path.read_text()) for path in sorted(BASE.glob("timing-*.json"))]
    if timings:
        print("## Uncached timing on a fixed sample\n")
        print("| Model | Cold first query | Query p50 / p95 | Passages/s (batch 8) | Peak RSS |")
        print("|---|---:|---:|---:|---:|")
        for item in timings:
            rss = max(item["peak_process_rss_kib"], item["peak_ollama_runner_rss_kib"]) / 1024
            latency = item["query_latency_ms"]
            print(
                f"| {item['model']} | {item['cold_first_query_s']:.2f} s "
                f"| {latency['p50']:.0f} / {latency['p95']:.0f} ms "
                f"| {item['documents_per_second_batch8']:.2f} | {rss:,.0f} MiB |"
            )


if __name__ == "__main__":
    main()
