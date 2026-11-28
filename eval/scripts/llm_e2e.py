# ruff: noqa: E501
"""End-to-end check of the whole pipeline against a real OpenAI-compatible chat provider.

It exercises what the unit tests only simulate: ingestion, retrieval, context assembly, generation,
citation validation, refusal on insufficient evidence, streaming, cancellation and the handling of
provider errors, all through the product's own code. Only the **public synthetic corpus** of the
repository (``eval/corpus``) may be indexed and sent: the script refuses any other directory. It
never prints, logs or stores the key (the environment variable is read by the adapter and nowhere
else) and checks that the key appears in none of its own output.

It talks to a paid or rate-limited service, so it refuses to run without ``--approved``, which says
that the owner agreed to use this provider. A run sends about 30 chat requests of at most 256 new
tokens each; the report gives the token totals so the cost can be read off.

A hosted provider says nothing about a local model server: this checks the OpenAI-compatible
adapter and the pipeline, **not** Ollama.

    export GROQ_API_KEY=...        # supplied by the owner; never committed
    uv run python eval/scripts/llm_e2e.py --approved --embed-dir <e5 dir> \\
        --base-url https://api.groq.com/openai/v1 --model <model> --api-key-env GROQ_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import resource_guard
from retrieval_eval import ROOT, write_config

from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.domain import AskStatus, RetrievalQuery
from rag_engine.evaluation.dataset import load_dataset
from rag_engine.evaluation.judge import LexicalJudge
from rag_engine.evaluation.llm_run import evaluate_answers, summarize_records
from rag_engine.pipeline import PipelineAnswer

PUBLIC = ROOT / "eval" / "corpus"
ANSWERABLE, UNANSWERABLE = 8, 6


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))


def llm_section(
    base_url: str, model: str, key_env: str, window: int, timeout: float, retries: int
) -> str:
    return (
        '[llm]\nenabled = true\nprovider = "openai_compat"\n'
        f'model = "{model}"\nbase_url = "{base_url}"\napi_key_env = "{key_env}"\n'
        f"context_window = {window}\nmax_new_tokens = 256\nrequest_timeout_s = {timeout}\n"
        f"max_retries = {retries}\n"
    )


def query_of(services: Any, text: str) -> RetrievalQuery:
    defaults = services.defaults
    return RetrievalQuery(
        request_id=services.ids.request_id(), text=text, mode=defaults.mode,
        top_k=defaults.top_k, candidate_k=max(defaults.candidate_k, defaults.top_k),
    )  # fmt: skip


def settle(answer: PipelineAnswer) -> str:
    return answer.status.value + (f":{answer.error.code.value}" if answer.error else "")


async def healthy(services: Any, text: str) -> bool:
    return (await services.pipeline.ask(query_of(services, text))).status is not AskStatus.ERROR


async def pipeline_checks(
    services: Any, report: Report, answerable: list[Any], unanswerable: list[Any]
) -> None:
    pipeline = services.pipeline
    judge = LexicalJudge()
    records = await evaluate_answers(
        pipeline, [*answerable, *unanswerable], judge, lambda q: query_of(services, q.question)
    )
    summary = summarize_records(records, judge_id=judge.judge_id)
    report.numbers["quality_exploratory"] = asdict(summary)
    statuses = {r.question_id: r.status for r in records}
    ok = [statuses[q.id] for q in answerable]
    report.add(
        "generation and citation validation",
        sum(s is AskStatus.ANSWERED for s in ok) >= max(1, len(ok) // 2),
        f"{sum(s is AskStatus.ANSWERED for s in ok)} of {len(ok)} answerable questions answered with valid citations",
    )
    errors = [s for s in statuses.values() if s is AskStatus.ERROR]
    report.add("no provider error in the normal run", not errors, f"{len(errors)} errors")
    refused = [statuses[q.id] for q in unanswerable]
    settled = sum(s in (AskStatus.INSUFFICIENT_EVIDENCE, AskStatus.UNVERIFIED) for s in refused)
    report.add(
        "insufficient evidence is reported, not invented",
        settled == len(refused),
        f"{settled} of {len(refused)} unanswerable questions ended as insufficient evidence or "
        "unverified (an 'answered' here is a finding to read, see the numbers)",
    )

    # streaming: verified text arrives once, the final event is authoritative
    first = answerable[0]
    kinds: list[str] = []
    final: PipelineAnswer | None = None
    async for event in pipeline.stream(query_of(services, first.question), verified_only=True):
        kinds.append(type(event).__name__)
        if type(event).__name__ == "FinalEvent":
            final = event.answer
    report.add(
        "streaming ends with one authoritative final event",
        final is not None and kinds.count("FinalEvent") == 1 and kinds[-1] == "FinalEvent",
        " > ".join(dict.fromkeys(kinds)),
    )

    # cancellation: stop reading after the first live event; the slot must come back
    async def read_one() -> None:
        async for _ in pipeline.stream(query_of(services, first.question), verified_only=False):
            await asyncio.sleep(3600)

    task = asyncio.create_task(read_one())
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
        cancelled = False
    except asyncio.CancelledError:
        cancelled = True
    again = await asyncio.wait_for(healthy(services, first.question), timeout=120)
    report.add(
        "cancellation releases the request and the pipeline still answers", cancelled and again
    )


async def error_checks(
    container: DefaultContainer,
    report: Report,
    workspace: Path,
    db: Path,
    args: argparse.Namespace,
    question: str,
    secret: str,
) -> None:
    variants: list[tuple[str, dict[str, str], str, str, float, int]] = [
        (
            "a wrong key",
            {args.api_key_env: "wrong-" + secret[:2]},
            args.base_url,
            args.model,
            60.0,
            0,
        ),
        ("an unknown model", {}, args.base_url, "no-such-model-" + args.model, 60.0, 0),
        ("an unreachable server", {}, "http://127.0.0.1:9/v1", args.model, 5.0, 0),
    ]
    seen: list[str] = []
    for label, env, base_url, model, timeout, retries in variants:
        config = workspace / "rag-error.toml"
        write_config(config, args.embed_dir.resolve(), 'mode = "dense"')
        config.write_text(
            config.read_text()
            + llm_section(base_url, model, args.api_key_env, args.context_window, timeout, retries),
            encoding="utf-8",
        )
        previous = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with container.ask(config, db) as services:
                answer = await services.pipeline.ask(query_of(services, question))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        text = json.dumps(
            {
                "status": answer.status.value,
                "error": None if answer.error is None else answer.error.message,
            }
        )
        seen.append(text)
        report.add(
            f"{label} is a clean error",
            answer.status is AskStatus.ERROR and answer.error is not None and secret not in text,
            settle(answer),
        )
    report.add("the key appears in no error text", not any(secret in text for text in seen))


def main(argv: list[str] | None = None, *, container: DefaultContainer | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", action="store_true", help="the owner approved this provider")
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="RAG_LLM_API_KEY")
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--corpus", type=Path, default=PUBLIC / "mini_v1")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "eval/datasets/retrieval_mini_v2.json"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-memory-gate", action="store_true", help="for tests with a mocked embedder"
    )
    args = parser.parse_args(argv)
    if not args.approved:
        print(
            "refused: pass --approved once the owner has agreed to use this provider",
            file=sys.stderr,
        )
        return 3
    corpus = args.corpus.resolve()
    if PUBLIC.resolve() not in corpus.parents:
        print(
            "refused: only a public corpus under eval/corpus may be sent to a provider",
            file=sys.stderr,
        )
        return 3
    secret = os.environ.get(args.api_key_env, "")
    if not secret:
        print(f"refused: the environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 3
    if not args.skip_memory_gate:
        resource_guard.require(1.2, what="the embedding model")
        resource_guard.watchdog()

    dataset = load_dataset(args.dataset)
    answerable = [q for q in dataset.questions if q.answerable][:ANSWERABLE]
    unanswerable = [q for q in dataset.questions if not q.answerable][:UNANSWERABLE]
    report = Report()
    container = container or DefaultContainer()
    with tempfile.TemporaryDirectory(prefix="rag-llm-e2e-") as folder:
        work = Path(folder)
        config, db = work / "rag.toml", work / "rag.db"
        write_config(config, args.embed_dir.resolve(), 'mode = "dense"')
        config.write_text(
            config.read_text()
            + llm_section(
                args.base_url, args.model, args.api_key_env, args.context_window, 120.0, 2
            ),
            encoding="utf-8",
        )
        prefix = ["--config", str(config), "--db", str(db)]
        ingest = [["source", "add", "corpus", str(corpus)], ["ingest", "corpus", "--json"]]
        report.add(
            "ingestion", all(rag([*prefix, *step], container=container) == 0 for step in ingest)
        )
        with container.ask(config, db) as services:
            query = query_of(services, answerable[0].question)
            outcome = asyncio.run(services.search.retrieval.search(query))
            wanted = {r.doc for r in answerable[0].gold}
            found = {h.chunk_ref.document_id for h in outcome.result.hits}
            report.add(
                "retrieval returns evidence", bool(outcome.result.hits), f"{len(found)} documents"
            )
            del wanted
            asyncio.run(pipeline_checks(services, report, answerable, unanswerable))
        asyncio.run(error_checks(container, report, work, db, args, answerable[0].question, secret))
    result = {
        "provider_base_url": args.base_url,
        "model": args.model,
        "corpus": str(corpus.relative_to(ROOT)),
        "all_passed": all(c.passed for c in report.checks),
        "checks": [asdict(c) for c in report.checks],
        "numbers": report.numbers,
        "note": "a hosted provider does not show that a local Ollama server works",
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    assert secret not in text  # noqa: S101 - the report must never carry the key
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
