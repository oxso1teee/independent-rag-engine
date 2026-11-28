"""Soak test of the HTTP server: a steady trickle of real searches, memory and errors measured.

It builds an index of the hard corpus with the real embedding model, starts ``rag serve`` as a
separate process, sends one search every ``--interval`` seconds for ``--minutes`` minutes and
reads the server's resident memory from ``/proc``. Pass criteria: resident
memory grows by less than 10 % (from the third request to the last), fewer than 2 % of requests
fail, and the server never dies (no out-of-memory kill). The run is refused when the machine does
not have the memory to hold the model.

This soaks search (embedding, vectors, SQLite), not answers: no language model was ever run on this
machine (the memory gate refused it), and the report says so.

    uv run python eval/scripts/soak.py --embed-dir <e5 dir> --minutes 60 --interval 60
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from retrieval_eval import ROOT, write_config

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.evaluation.dataset import load_dataset

GIB = 1024**3
MIN_AVAILABLE = int(2.5 * GIB)
TOKEN = "soak-token-0123456789abcdef"  # noqa: S105 - a throwaway token for a loopback test


def meminfo(key: str) -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return 0


def rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def search(port: int, text: str) -> tuple[int, float]:
    body = json.dumps({"query": text, "mode": "dense", "top_k": 5}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/search",
        data=body,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    began = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            response.read()
            return response.status, time.perf_counter() - began
    except urllib.error.HTTPError as error:
        return error.code, time.perf_counter() - began
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, time.perf_counter() - began


def build_report(
    samples: list[dict[str, float]],
    failures: int,
    died: bool,
    min_available: int,
    args: argparse.Namespace,
    *,
    complete: bool,
) -> dict[str, object]:
    steady = samples[2:] or samples
    growth = (steady[-1]["rss_mib"] / steady[0]["rss_mib"] - 1) if len(steady) > 1 else 0.0
    latencies = sorted(s["seconds"] for s in samples)
    error_rate = failures / len(samples) if samples else 1.0
    return {
        "complete": complete,
        "requests": len(samples),
        "minutes": args.minutes,
        "interval_s": args.interval,
        "server_died": died,
        "error_rate": error_rate,
        "rss_first_mib": steady[0]["rss_mib"] if steady else None,
        "rss_last_mib": steady[-1]["rss_mib"] if steady else None,
        "rss_max_mib": max((s["rss_mib"] for s in samples), default=None),
        "rss_growth": growth,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_max_s": latencies[-1] if latencies else None,
        "min_mem_available_gib": round(min_available / GIB, 2),
        "criteria": {
            "rss_growth_below_10_percent": growth < 0.10,
            "errors_below_2_percent": error_rate < 0.02,
            "server_survived": not died,
        },
        "scope": "search only (real embedding model); no language model was run",
        "samples": samples,
    }


def write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=18799)
    parser.add_argument("--output", type=Path, default=ROOT / "eval_runs" / "soak.json")
    args = parser.parse_args()
    if meminfo("MemAvailable") < MIN_AVAILABLE:
        print(
            json.dumps(
                {
                    "result": "refused_by_resource_gate",
                    "mem_available_gib": round(meminfo("MemAvailable") / GIB, 2),
                }
            )
        )
        sys.exit(3)

    dataset = load_dataset(ROOT / "eval/datasets/retrieval_hard_v1.json")
    questions = [q.question for q in dataset.questions if q.answerable and q.split == "dev"]
    corpus_dir = ROOT / dataset.corpus
    with tempfile.TemporaryDirectory(prefix="rag-soak-") as folder:
        work = Path(folder)
        config, db = work / "rag.toml", work / "rag.db"
        write_config(config, args.embed_dir.resolve(), 'mode = "dense"')
        prefix = ["--config", str(config), "--db", str(db)]
        container = DefaultContainer()
        for step in (["source", "add", "corpus", str(corpus_dir)], ["ingest", "corpus", "--json"]):
            if rag([*prefix, *step], container=container) != 0:
                sys.exit(f"rag {step[0]} failed")
        env = {**os.environ, "RAG_API_TOKEN": TOKEN}
        server = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "rag_engine", *prefix, "serve", "--port", str(args.port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        samples: list[dict[str, float]] = []
        failures = 0
        started = time.time()
        deadline = started + args.minutes * 60
        min_available = meminfo("MemAvailable")
        died = False
        try:
            time.sleep(4)
            n = 0
            while time.time() < deadline:
                status, seconds = search(args.port, questions[n % len(questions)])
                rss = rss_mib(server.pid)
                if server.poll() is not None or rss is None:
                    died = True
                    break
                failures += status != 200
                min_available = min(min_available, meminfo("MemAvailable"))
                samples.append(
                    {
                        "n": n,
                        "t": round(time.time() - started),
                        "status": status,
                        "seconds": seconds,
                        "rss_mib": rss,
                    }
                )
                n += 1
                # a run can be watched, and an interrupted one still leaves its samples behind
                partial = build_report(samples, failures, died, min_available, args, complete=False)
                write_report(partial, args.output)
                time.sleep(max(0.0, args.interval - seconds))
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()

    report = build_report(samples, failures, died, min_available, args, complete=True)
    write_report(report, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
