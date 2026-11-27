"""Reproducible, judged-pool NoMIRACL embedding comparison.

Download only the English and Russian NoMIRACL dev.relevant corpus, topics, and qrels at the
pinned revision documented in EMBEDDING_BAKEOFF.md. This evaluates ranking among each query's
human-judged passages; it is not a full-corpus MIRACL retrieval score.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import math
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from bakeoff_embedders import percentile, sample_runner_rss

ROOT = Path(__file__).resolve().parents[2]
REVISION = "ecd08778d0426a5ca28ac99763b0c9ddc2c78e68"
HASHES = {
    "english": (
        "f333c75491c8dac69f946b021511a67666318efb799dff3a6d927a841f8148eb",
        "25d9e6d17656674fd7aa3b67ee912c71545e664a842b147bfc17e9ff391db3da",
        "a0af836c300e85088c4d73449492e2378280f967cac032853c9e11f3bc8a8d81",
    ),
    "russian": (
        "8beb1a33ff56a914ac9e88362e1d580ae972ecf79872e1bf7d6d5672e130f88d",
        "91a7c3b6d21f79a12b5330f0a62811962f9370a0bf539cba8b3fecaa095f0616",
        "f877b43c7f53b76aab25e406e5a607ef1c5adba4ffaa11e1b252a4326735a19b",
    ),
}
QUERY_INSTRUCTION = "Instruct: Given a web search question, retrieve relevant passages.\nQuery: "


def load_sample(base: Path) -> dict[str, dict[str, Any]]:
    sample: dict[str, dict[str, Any]] = {}
    for lang in ("english", "russian"):
        folder = base / lang
        files = (
            folder / "corpus.jsonl.gz",
            folder / "topics/dev.relevant.tsv",
            folder / "qrels/dev.relevant.tsv",
        )
        for path, expected in zip(files, HASHES[lang], strict=True):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"NoMIRACL file differs from pinned revision: {path}")
        topics = dict(line.split("\t", 1) for line in files[1].read_text().splitlines())
        judged: dict[str, dict[str, int]] = {}
        for line in files[2].read_text().splitlines():
            query_id, _, doc_id, relevance = line.split("\t")
            judged.setdefault(query_id, {})[doc_id] = int(relevance)
        query_ids = sorted(
            (query_id for query_id in topics if set(judged[query_id].values()) == {0, 1}),
            key=int,
        )
        selected = [query_ids[(i * len(query_ids)) // 40] for i in range(40)]
        needed = {doc_id for query_id in selected for doc_id in judged[query_id]}
        documents: dict[str, str] = {}
        with gzip.open(files[0], "rt") as stream:
            for line in stream:
                item = json.loads(line)
                if item["docid"] in needed:
                    documents[item["docid"]] = f"{item['title']}\n{item['text']}"
        if needed != documents.keys():
            raise ValueError(f"NoMIRACL judged documents are missing for {lang}")
        for query_id in selected:
            labels = set(judged[query_id].values())
            if labels != {0, 1}:
                raise ValueError(
                    f"NoMIRACL query lacks positive and negative judgments: {query_id}"
                )
        sample[lang] = {
            "queries": {query_id: topics[query_id] for query_id in selected},
            "judged": {query_id: judged[query_id] for query_id in selected},
            "documents": documents,
        }
    return sample


class OllamaModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.client = httpx.Client(base_url="http://127.0.0.1:11434", trust_env=False)
        tags = self.client.get("/api/tags", timeout=30).json()["models"]
        match = next((item for item in tags if item["name"] == name), None)
        if match is None:
            raise ValueError(f"Ollama model is not installed: {name}")
        self.revision = match["digest"]
        self.bytes = match["size"]

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.post(
            "/api/embed",
            json={"model": self.name, "input": texts, "truncate": False, "keep_alive": "30m"},
            timeout=600,
        )
        response.raise_for_status()
        return np.asarray(response.json()["embeddings"], dtype=np.float32)

    def close(self) -> None:
        self.client.close()


class E5Model:
    def __init__(self, path: Path) -> None:
        # imported by name: the optional `onnx` extra is absent from the type-checking environment
        fastembed = importlib.import_module("fastembed")
        description = importlib.import_module("fastembed.common.model_description")
        TextEmbedding = fastembed.TextEmbedding
        ModelSource, PoolingType = description.ModelSource, description.PoolingType

        self.name = "intfloat/multilingual-e5-small"
        self.revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        self.bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        TextEmbedding.add_custom_model(
            model=self.name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=self.name),
            dim=384,
            license="mit",
        )
        self.backend = TextEmbedding(
            model_name=self.name,
            specific_model_path=str(path.resolve()),
            local_files_only=True,
            threads=4,
            providers=["CPUExecutionProvider"],
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.backend.embed(texts, batch_size=len(texts))), dtype=np.float32)

    def close(self) -> None:
        pass


def prefix_for(model: OllamaModel | E5Model, kind: str) -> str:
    if isinstance(model, E5Model):
        return "passage: " if kind == "documents" else "query: "
    return "" if kind == "documents" else QUERY_INSTRUCTION


def window_overflow(path: Path, sample: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Judged passages longer than the E5 window; FastEmbed truncates them silently."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.no_truncation()
    return {
        lang: sum(
            len(tokenizer.encode("passage: " + text).ids) > 512
            for text in data["documents"].values()
        )
        for lang, data in sample.items()
    }


def measure(
    model: OllamaModel | E5Model, sample: dict[str, dict[str, Any]], cache: Path
) -> dict[str, Any]:
    """Ranking quality among the judged passages; vectors are cached, so resuming is safe.

    Timing is deliberately absent: a resumed run only times what was missing. Use ``timing``.
    """
    existing: dict[str, np.ndarray] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            item = json.loads(line)
            existing[item["key"]] = np.asarray(item["vector"], dtype=np.float32)
    for lang, data in sample.items():
        for kind, items in (("documents", data["documents"]), ("queries", data["queries"])):
            rows = sorted(items.items(), key=lambda item: item[0])
            step = 8 if kind == "documents" else 1
            for offset in range(0, len(rows), step):
                missing = [
                    (key, text)
                    for key, text in rows[offset : offset + step]
                    if f"{lang}:{kind}:{key}" not in existing
                ]
                if not missing:
                    continue
                prefix = prefix_for(model, kind)
                vectors = model.embed([prefix + text for _, text in missing])
                if vectors.shape[0] != len(missing) or not np.isfinite(vectors).all():
                    raise ValueError("provider returned invalid vectors")
                norms = np.linalg.norm(vectors, axis=1)
                if (norms <= 0).any() or not np.isfinite(norms).all():
                    raise ValueError("provider returned zero vectors")
                vectors /= norms[:, None]
                with cache.open("a") as stream:
                    for (key, _), vector in zip(missing, vectors, strict=True):
                        identity = f"{lang}:{kind}:{key}"
                        existing[identity] = vector
                        stream.write(
                            json.dumps({"key": identity, "vector": vector.tolist()}) + "\n"
                        )
                if len(existing) % 80 < len(missing):
                    print(f"{model.name}: {len(existing)} vectors cached", flush=True)
    metrics: dict[str, Any] = {}
    for lang, data in sample.items():
        per_query: list[dict[str, float]] = []
        for query_id, labels in data["judged"].items():
            query = existing[f"{lang}:queries:{query_id}"]
            ranked = sorted(
                labels,
                key=lambda doc_id: (
                    -float(np.dot(query, existing[f"{lang}:documents:{doc_id}"])),
                    doc_id,
                ),
            )
            relevant = sum(labels.values())
            ranks = [rank for rank, doc_id in enumerate(ranked, 1) if labels[doc_id] > 0]
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant, 10) + 1))
            actual = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
            per_query.append(
                {
                    "recall_at_1": sum(rank <= 1 for rank in ranks) / relevant,
                    "recall_at_5": sum(rank <= 5 for rank in ranks) / relevant,
                    "recall_at_10": sum(rank <= 10 for rank in ranks) / relevant,
                    "mrr_at_10": 1 / min(ranks) if min(ranks) <= 10 else 0,
                    "ndcg_at_10": actual / ideal,
                }
            )
        metrics[lang] = {
            "queries": len(data["queries"]),
            "candidate_documents": len(data["documents"]),
            "mean_judged_per_query": statistics.mean(map(len, data["judged"].values())),
            "mean": {key: statistics.mean(row[key] for row in per_query) for key in per_query[0]},
            "per_query": dict(zip(data["queries"], per_query, strict=True)),
        }
    return {
        "dataset": "NoMIRACL dev.relevant judged-pool, 40 evenly spaced queries per language",
        "dataset_revision": REVISION,
        "model": model.name,
        "model_revision": model.revision,
        "model_bytes": model.bytes,
        "dimension": len(next(iter(existing.values()))),
        "metrics": metrics,
    }


def timing(model: OllamaModel | E5Model, sample: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Uncached latency, throughput and peak memory on a fixed, evenly spaced sample."""
    documents = sorted(
        (f"{lang}:{key}", text)
        for lang, data in sample.items()
        for key, text in data["documents"].items()
    )
    documents = documents[:: len(documents) // 64][:64]
    queries = sorted(
        (f"{lang}:{key}", text)
        for lang, data in sample.items()
        for key, text in data["queries"].items()
    )[::2]
    if isinstance(model, OllamaModel):
        subprocess.run(["ollama", "stop", model.name], check=False, capture_output=True)  # noqa: S603, S607
        time.sleep(2)
    peak_runner_kib = [0]
    stop = threading.Event()
    monitor = threading.Thread(target=sample_runner_rss, args=(stop, peak_runner_kib), daemon=True)
    monitor.start()
    try:
        started = time.perf_counter()
        model.embed([prefix_for(model, "queries") + queries[0][1]])
        cold_first_query_s = time.perf_counter() - started
        model.embed([prefix_for(model, "documents") + documents[0][1]])  # warm-up
        batches = [documents[i : i + 8] for i in range(0, len(documents), 8)]
        started = time.perf_counter()
        for batch in batches:
            model.embed([prefix_for(model, "documents") + text for _, text in batch])
        document_seconds = time.perf_counter() - started
        query_ms = []
        for _, text in queries:
            before = time.perf_counter()
            model.embed([prefix_for(model, "queries") + text])
            query_ms.append((time.perf_counter() - before) * 1000)
    finally:
        stop.set()
        monitor.join(timeout=1)
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {
        "model": model.name,
        "model_revision": model.revision,
        "documents_timed": len(documents),
        "queries_timed": len(queries),
        "cold_first_query_s": cold_first_query_s,
        "documents_per_second_batch8": len(documents) / document_seconds,
        "query_latency_ms": {
            "p50": percentile(query_ms, 0.5),
            "p95": percentile(query_ms, 0.95),
        },
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_ollama_runner_rss_kib": max(peak_runner_kib),
        "mem_available_kib_at_end": int(meminfo["MemAvailable"].split()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quality", "timing"))
    parser.add_argument("model", choices=("qwen3-embedding:0.6b", "qwen3-embedding:4b", "e5-small"))
    args = parser.parse_args()
    sample = load_sample(ROOT / "data/eval/nomiracl")
    name = args.model.replace(":", "-")
    e5_path = ROOT / "data/models/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3"
    loading = time.perf_counter()
    model = E5Model(e5_path) if args.model == "e5-small" else OllamaModel(args.model)
    model_load_s = time.perf_counter() - loading  # Ollama loads lazily: see cold_first_query_s
    base = ROOT / "data/eval/nomiracl"
    try:
        if args.mode == "quality":
            result = measure(model, sample, base / f"vectors-{name}.jsonl")
            if args.model == "e5-small":
                result["passages_over_512_tokens"] = window_overflow(e5_path, sample)
            output = base / f"results-{name}.json"
        else:
            result = {**timing(model, sample), "model_load_s": model_load_s}
            output = base / f"timing-{name}.json"
    finally:
        model.close()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    for lang, value in result.get("metrics", {}).items():
        print(lang, value["mean"])


if __name__ == "__main__":
    main()
