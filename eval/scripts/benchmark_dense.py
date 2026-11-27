"""Measure exact flat-vector search at 10k and 100k 1024-d; set OPENBLAS_NUM_THREADS=4."""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import resource
import statistics
import time

import numpy as np

from rag_engine.domain import ChunkId, EmbeddingModelId, IdFactory, Uuid7Generator
from rag_engine.storage.vector_flat import FlatVectorIndex


def current_rss_mib() -> float:
    for line in open("/proc/self/status", encoding="utf-8"):  # noqa: SIM115
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS unavailable")


def search_in_child(
    index: FlatVectorIndex, queries: np.ndarray, pipe: multiprocessing.connection.Connection
) -> None:
    """Measure the search worker after fork, with only its resident index and queries."""
    try:
        gc.collect()
        search_rss = current_rss_mib()
        times = []
        for query in queries:
            started = time.perf_counter()
            if len(index.search(query, 10)) != 10:
                raise RuntimeError("index returned fewer than ten hits")
            times.append((time.perf_counter() - started) * 1000)
        pipe.send(
            {
                "p50_ms": statistics.median(times),
                "p95_ms": sorted(times)[int(0.95 * len(times)) - 1],
                "search_rss_mib": search_rss,
                "search_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
        )
    finally:
        pipe.close()


def measure(count: int) -> dict[str, float | int]:
    rng = np.random.default_rng(104)
    vectors = rng.standard_normal((count, 1024), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    model = EmbeddingModelId.build(
        provider="fake", name="perf", revision="1", dimension=1024, normalized=True
    )
    version = IdFactory(
        Uuid7Generator(lambda: 1_767_225_600_000, lambda size: b"\0" * size)
    ).document_version_id()
    ids = [ChunkId.derive(version, "perf", number) for number in range(count)]
    index = FlatVectorIndex(model, 1024, max_memory_mb=300)
    index.upsert(ids, vectors)
    queries = vectors[:40].copy()
    del vectors, ids
    gc.collect()
    build_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    receiver, sender = multiprocessing.get_context("fork").Pipe(duplex=False)
    worker = multiprocessing.get_context("fork").Process(
        target=search_in_child, args=(index, queries, sender)
    )
    worker.start()
    sender.close()
    search = receiver.recv()
    receiver.close()
    worker.join()
    if worker.exitcode != 0:
        raise RuntimeError("search worker failed")
    return {
        "vectors": count,
        "dimension": 1024,
        "query_count": len(queries),
        "construction_peak_rss_mib": build_peak,
        **search,
    }


def main() -> None:
    if os.environ.get("OPENBLAS_NUM_THREADS") != "4":
        raise SystemExit("set OPENBLAS_NUM_THREADS=4 for the specified workload")
    for count in (10_000, 100_000):
        print(json.dumps(measure(count), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
