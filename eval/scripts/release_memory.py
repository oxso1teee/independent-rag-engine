"""Measure what ``OnnxEmbedder.release()`` actually gives back, instead of assuming it.

Dropping the reference only makes the model collectable; whether the process' resident memory
shrinks depends on the allocators underneath. This loads the real embedder, embeds a few texts,
releases it and reports the resident set size (``VmRSS``) at each step, then again after an
explicit ``malloc_trim``. The cross-encoder is not measured here: loading it needs about 2.5 GiB.

    uv run python eval/scripts/release_memory.py --embed-dir <e5 model directory>
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import json
import tempfile
from pathlib import Path

from retrieval_eval import write_config

from rag_engine.config import load_settings
from rag_engine.container import build_embedding
from rag_engine.embedding.onnx import OnnxEmbedder
from rag_engine.system import SystemClock

MIB = 1024 * 1024


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS not found")


def trim() -> bool:
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rag-release-") as folder:
        config = Path(folder) / "rag.toml"
        write_config(config, args.embed_dir.resolve(), 'mode = "dense"')
        built = build_embedding(load_settings(config).embedding, SystemClock())
        provider = built.provider
        assert isinstance(provider, OnnxEmbedder)  # noqa: S101 - the config selects ONNX
        steps = {"start_after_load_mib": rss_mib()}
        texts = [f"passage number {i} about backups and printers" * 4 for i in range(16)]
        asyncio.run(provider.embed_documents(texts))
        steps["after_embedding_mib"] = rss_mib()
        peak = steps["after_embedding_mib"]
        provider.release()
        gc.collect()
        steps["after_release_mib"] = rss_mib()
        steps["trim_returned_memory"] = float(trim())
        steps["after_malloc_trim_mib"] = rss_mib()
    result = {
        "steps": steps,
        "released_by_release_mib": round(peak - steps["after_release_mib"], 1),
        "released_with_trim_mib": round(peak - steps["after_malloc_trim_mib"], 1),
        "note": "one run, one process, no concurrent load control: an indication, not a benchmark",
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
