"""Small, reproducible RU/EN embedding probe using already installed Ollama models.

This diagnostic corpus is deliberately modest. Its scores describe this corpus only; they
must not be used as a general quality claim or as a substitute for a reviewed eval set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "embedding_probe_v1.json"


def sample_runner_rss(stop: threading.Event, peak_kib: list[int]) -> None:
    """Observe local Ollama runner RSS while inference is active."""
    while not stop.wait(0.05):
        total = 0
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                command = Path(entry.path, "cmdline").read_bytes()
                if b"/llama-server" not in command:
                    continue
                for line in Path(entry.path, "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
            except (OSError, ValueError):
                continue
        peak_kib[0] = max(peak_kib[0], total)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))]


def embed(client: httpx.Client, model: str, texts: list[str]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    response = client.post(
        "/api/embed",
        json={"model": model, "input": texts, "truncate": False, "keep_alive": "5m"},
        timeout=300,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("invalid embedding batch")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms <= 0).any():
        raise ValueError("zero-length embedding")
    return vectors / norms[:, None], elapsed


def run(model: str, dataset: Path) -> dict[str, Any]:
    raw = dataset.read_bytes()
    data = json.loads(raw)
    docs = data["documents"]
    questions = data["questions"]
    ids = [item["id"] for item in docs]
    runner_peak_kib = [0]
    stop = threading.Event()
    sampler = threading.Thread(target=sample_runner_rss, args=(stop, runner_peak_kib))
    with httpx.Client(base_url="http://127.0.0.1:11434", trust_env=False) as client:
        tags = client.get("/api/tags", timeout=10)
        tags.raise_for_status()
        installed = {item["name"]: item for item in tags.json()["models"]}
        if model not in installed:
            raise ValueError(f"model {model!r} is not installed locally")
        sampler.start()
        try:
            vectors, document_seconds = embed(client, model, [item["text"] for item in docs])
            query_times: list[float] = []
            scores: dict[str, list[tuple[float, float]]] = {"ru": [], "en": []}
            for question in questions:
                query, seconds = embed(client, model, [question["text"]])
                query_times.append(seconds * 1000)
                similarity = vectors @ query[0]
                ranked = sorted(range(len(ids)), key=lambda i: (-float(similarity[i]), ids[i]))
                gold = set(question["gold"])
                hits = [ids[i] for i in ranked[:10]]
                recall = len(gold.intersection(hits[:5])) / len(gold)
                dcg = sum(
                    1 / np.log2(rank + 2) for rank, doc_id in enumerate(hits) if doc_id in gold
                )
                ideal = sum(1 / np.log2(rank + 2) for rank in range(len(gold)))
                scores[question["language"]].append((recall, float(dcg / ideal)))
        finally:
            stop.set()
            sampler.join()
    return {
        "model": model,
        "digest": installed[model]["digest"],
        "model_size_bytes": installed[model]["size"],
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "documents": len(docs),
        "queries": len(questions),
        "dimensions": int(vectors.shape[1]),
        "ingestion_documents_per_second": len(docs) / document_seconds,
        "ingestion_seconds": document_seconds,
        "query_embedding_p50_ms": statistics.median(query_times),
        "query_embedding_p95_ms": percentile(query_times, 0.95),
        "runner_peak_rss_mib": runner_peak_kib[0] / 1024,
        "python_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "quality": {
            language: {
                "n": len(rows),
                "recall_at_5": statistics.mean(row[0] for row in rows),
                "ndcg_at_10": statistics.mean(row[1] for row in rows),
            }
            for language, rows in scores.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
