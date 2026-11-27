"""One real generation through the product pipeline, behind a memory gate.

The gate exists because this machine is small and shared: a model that does not fit is refused
before it is loaded, never attempted. The gate compares the model's size on disk (as Ollama
reports it) plus a fixed headroom for the KV cache and the runner with the memory the kernel
says is available, and it also requires some free swap, since a loaded model with nowhere to
page out is how an out-of-memory kill happens. A refusal is a result, not a failure of the
script: it exits with code 3 and says what was missing.

    uv run python eval/scripts/llm_smoke.py --model qwen3.5:4b --context-window 4096
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.domain import AnswerStatus, RagError
from rag_engine.generation.generator import AnswerGenerator, GenerationPolicy
from rag_engine.generation.resilience import ResiliencePolicy, ResilientLLM
from rag_engine.llm import OllamaChat
from rag_engine.system import system_id_factory
from rag_engine.testing import evidence_context

GIB = 1024**3
HEADROOM_BYTES = int(1.0 * GIB)
MIN_FREE_SWAP_BYTES = int(0.5 * GIB)
EVIDENCE = (
    "Backups run every night at 02:30 UTC and are kept for 14 days.",
    "Printers are on the third floor next to the kitchen.",
)
QUESTIONS = (
    ("answerable", "When do the backups run?"),
    ("unanswerable", "What is the guest wifi password?"),
)


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024
    return values


def gate(model_bytes: int) -> tuple[bool, dict[str, float]]:
    info = meminfo()
    need = model_bytes + HEADROOM_BYTES
    report = {
        "model_gib": round(model_bytes / GIB, 2),
        "need_gib": round(need / GIB, 2),
        "mem_available_gib": round(info["MemAvailable"] / GIB, 2),
        "swap_free_gib": round(info["SwapFree"] / GIB, 2),
    }
    ok = info["MemAvailable"] >= need and info["SwapFree"] >= MIN_FREE_SWAP_BYTES
    return ok, report


async def installed_size(base_url: str, model: str) -> int | None:
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0, trust_env=False) as client:
        tags = (await client.get("/api/tags")).json()
    for entry in tags.get("models", []):
        if entry.get("name") == model:
            return int(entry["size"])
    return None


async def run(args: argparse.Namespace) -> int:
    size = await installed_size(args.base_url, args.model)
    if size is None:
        print(json.dumps({"result": "model_not_installed", "model": args.model}))
        return 2
    ok, report = gate(size)
    if not ok:
        print(json.dumps({"result": "refused_by_resource_gate", **report}, indent=2))
        return 3
    counter = HeuristicTokenCounter()
    chat = OllamaChat(
        args.model,
        context_window=args.context_window,
        max_new_tokens=args.max_new,
        counter=counter,
        base_url=args.base_url,
        timeout_s=args.timeout,
        keep_alive="0",
    )
    llm = ResilientLLM(chat, ResiliencePolicy(timeout_s=args.timeout, max_retries=0))
    generator = AnswerGenerator(llm, policy=GenerationPolicy(max_new_tokens=args.max_new))
    ids = system_id_factory()
    context = evidence_context(
        EVIDENCE, document_id=ids.document_id(), version_id=ids.document_version_id()
    )
    request_id = ids.request_id()
    results: list[dict[str, object]] = []
    try:
        for kind, question in QUESTIONS:
            began = time.perf_counter()
            try:
                answer = await generator.answer(request_id, question, context)
                results.append(
                    {
                        "kind": kind,
                        "status": answer.status.value,
                        "correct": (answer.status is AnswerStatus.ANSWERED)
                        == (kind == "answerable"),
                        "text": answer.text,
                        "citations": [c.evidence_id for c in answer.citations],
                        "usage": [answer.usage.prompt_tokens, answer.usage.completion_tokens],
                        "warnings": list(answer.warnings),
                        "seconds": round(time.perf_counter() - began, 1),
                    }
                )
            except RagError as error:
                results.append({"kind": kind, "error": error.to_dict()})
    finally:
        await chat.aclose()
    after = meminfo()
    print(
        json.dumps(
            {
                "result": "ran",
                "model": args.model,
                "sentinel": REFUSAL_SENTINEL,
                "gate": report,
                "mem_available_after_gib": round(after["MemAvailable"] / GIB, 2),
                "answers": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
