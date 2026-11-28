"""Soak test of the HTTP server: a steady trickle of real searches, memory and errors measured.

It builds an index of the hard corpus with the real embedding model, starts ``rag serve`` as a
separate process, sends one search every ``--interval`` seconds for ``--minutes`` minutes and
reads the server's resident memory from ``/proc``. Pass criteria: resident
memory grows by less than 10 % (from the third request to the last), fewer than 2 % of requests
fail, and the server never dies (no out-of-memory kill). The run is refused when the machine does
not have the memory to hold the model.

Two options make it harder. ``--workers N`` sends N searches at once at every tick, and
``--churn-every S`` edits a document of a private copy of the corpus and runs ``rag ingest`` in a
second process every S seconds, so the server keeps reading while another writer publishes new
index generations. Beyond memory, the run follows open file descriptors, threads and the size of
the write-ahead log; a leak of any of them fails it, and so does a server that does not stop
cleanly on SIGTERM. A 503 ``E_STORE_LOCKED`` is the documented, retryable answer to a busy writer
and is counted apart from failures.

This soaks search (embedding, vectors, SQLite), not answers: no language model was ever run on this
machine (the memory gate refused it), and the report says so.

    uv run python eval/scripts/soak.py --embed-dir <e5 dir> --minutes 60 --interval 60
    uv run python eval/scripts/soak.py --embed-dir <e5 dir> --minutes 10 --interval 5 \\
        --workers 3 --churn-every 60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import resource_guard
from retrieval_eval import ROOT, write_config

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.evaluation.dataset import load_dataset

GIB = 1024**3
MIN_AVAILABLE = int(2.5 * GIB)
CHURN_MODEL_GIB = 1.2  # the model loaded again by the ingesting process
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


def open_fds(pid: int) -> int | None:
    try:
        return len(list(Path(f"/proc/{pid}/fd").iterdir()))
    except OSError:
        return None


def threads(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def wal_mib(db: Path) -> float:
    wal = db.with_name(db.name + "-wal")
    return wal.stat().st_size / 1024**2 if wal.exists() else 0.0


PROBLEMS: dict[str, int] = {}


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
        try:  # an RFC 9457 problem: its engine code says *why* (busy embedder, locked store, ...)
            code = str(json.loads(error.read()).get("code", "?"))
        except (ValueError, OSError, AttributeError):
            code = "?"
        PROBLEMS[f"{error.code} {code}"] = PROBLEMS.get(f"{error.code} {code}", 0) + 1
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
    locked: int = 0,
    churn: tuple[int, int] = (0, 0),
    clean_stop: bool | None = None,
) -> dict[str, object]:
    steady = samples[2:] or samples
    growth = (steady[-1]["rss_mib"] / steady[0]["rss_mib"] - 1) if len(steady) > 1 else 0.0
    latencies = sorted(s["seconds"] for s in samples)
    error_rate = failures / len(samples) if samples else 1.0
    fd_first, fd_last = (steady[0].get("fds", 0), steady[-1].get("fds", 0)) if steady else (0, 0)
    th_first, th_last = (
        (steady[0].get("threads", 0), steady[-1].get("threads", 0)) if steady else (0, 0)
    )
    wal_max = max((s.get("wal_mib", 0.0) for s in samples), default=0.0)
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
        "workers": args.workers,
        "churn_every_s": args.churn_every,
        "locked_503": locked,
        "problem_codes": dict(PROBLEMS),
        "churn_ingests": {"ok": churn[0], "failed": churn[1]},
        "fds_first": fd_first,
        "fds_last": fd_last,
        "threads_first": th_first,
        "threads_last": th_last,
        "wal_max_mib": round(wal_max, 2),
        "clean_stop": clean_stop,
        "criteria": {
            "rss_growth_below_10_percent": growth < 0.10,
            "errors_below_2_percent": error_rate < 0.02,
            "server_survived": not died,
            "fds_do_not_leak": fd_last <= max(fd_first * 1.1, fd_first + 8),
            "threads_do_not_leak": th_last <= th_first + 4,
            "wal_stays_below_64_mib": wal_max < 64,
            "churn_ingests_succeeded": churn[1] == 0,
            "stopped_cleanly": clean_stop is not False,
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
    parser.add_argument("--workers", type=int, default=1, help="searches sent at once per tick")
    parser.add_argument(
        "--churn-every", type=float, default=0.0, help="seconds between ingests by a second process"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "eval_runs" / "soak.json")
    args = parser.parse_args()
    # The server alone needs MIN_AVAILABLE. Concurrent indexing starts a second process that loads
    # the same model (about 1.2 GiB): two runs on a machine with 2 to 3 GiB free were stopped by the
    # watchdog at the first ingest, so that case needs the model's memory on top.
    needed = MIN_AVAILABLE + (int(CHURN_MODEL_GIB * GIB) if args.churn_every > 0 else 0)
    if meminfo("MemAvailable") < needed:
        print(
            json.dumps(
                {
                    "result": "refused_by_resource_gate",
                    "mem_available_gib": round(meminfo("MemAvailable") / GIB, 2),
                    "needed_gib": round(needed / GIB, 2),
                }
            )
        )
        sys.exit(3)

    dataset = load_dataset(ROOT / "eval/datasets/retrieval_hard_v1.json")
    questions = [q.question for q in dataset.questions if q.answerable and q.split == "dev"]
    with tempfile.TemporaryDirectory(prefix="rag-soak-") as folder:
        work = Path(folder)
        corpus_dir = work / "corpus"  # a private copy: churn edits it, the repository stays clean
        shutil.copytree(ROOT / dataset.corpus, corpus_dir)
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
        locked = 0
        churn = [0, 0]  # succeeded, failed
        stop = threading.Event()

        children: list[subprocess.Popen[bytes]] = []

        def trip() -> None:
            """The memory watchdog fired: stop everything this run started."""
            for child in [*children, server]:
                if child.poll() is None:
                    child.kill()

        resource_guard.watchdog(trip)

        def churner() -> None:
            document = sorted(corpus_dir.glob("*.md"))[0]
            number = 0
            while not stop.wait(args.churn_every):
                number += 1
                with document.open("a", encoding="utf-8") as handle:
                    handle.write(f"\nSoak edit number {number}: a paragraph that is new.\n")
                child = subprocess.Popen(  # noqa: S603
                    [sys.executable, "-m", "rag_engine", *prefix, "ingest", "corpus", "--json"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )  # fmt: skip
                children.append(child)
                try:
                    code = child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = 1
                churn[0 if code == 0 else 1] += 1

        worker = threading.Thread(target=churner, daemon=True) if args.churn_every > 0 else None
        started = time.time()
        deadline = started + args.minutes * 60
        min_available = meminfo("MemAvailable")
        died = False
        try:
            time.sleep(4)
            if worker is not None:
                worker.start()
            n = 0
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                while time.time() < deadline:
                    tick = time.perf_counter()
                    texts = [questions[(n + k) % len(questions)] for k in range(args.workers)]
                    results = list(pool.map(lambda text: search(args.port, text), texts))
                    rss = rss_mib(server.pid)
                    if server.poll() is not None or rss is None:
                        died = True
                        break
                    locked += sum(status == 503 for status, _ in results)
                    failures += sum(status not in (200, 503) for status, _ in results)
                    min_available = min(min_available, meminfo("MemAvailable"))
                    samples.append(
                        {
                            "n": n,
                            "t": round(time.time() - started),
                            "status": max(status for status, _ in results),
                            "seconds": max(seconds for _, seconds in results),
                            "rss_mib": rss,
                            "fds": open_fds(server.pid) or 0,
                            "threads": threads(server.pid) or 0,
                            "wal_mib": round(wal_mib(db), 3),
                        }
                    )
                    n += args.workers
                    # a run can be watched, and an interrupted one still leaves its samples behind
                    partial = build_report(
                        samples, failures, died, min_available, args, complete=False,
                        locked=locked, churn=(churn[0], churn[1]),
                    )  # fmt: skip
                    write_report(partial, args.output)
                    time.sleep(max(0.0, args.interval - (time.perf_counter() - tick)))
        finally:
            stop.set()
            if worker is not None and worker.is_alive():
                worker.join(timeout=620)
            clean = None
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)  # what a container stop sends
                try:
                    clean = server.wait(timeout=20) in (0, -signal.SIGTERM)
                except subprocess.TimeoutExpired:
                    server.kill()
                    clean = False

    report = build_report(
        samples, failures, died, min_available, args, complete=True,
        locked=locked, churn=(churn[0], churn[1]), clean_stop=clean,
    )  # fmt: skip
    write_report(report, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
