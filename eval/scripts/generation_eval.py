"""Generation evaluation: citation validity, degenerate answers and refusal behaviour.

Two modes over the mini dataset:

* ``e2e``: the real pipeline (retrieval, context, gate, model) for every counted question;
* ``oracle``: the model is given the gold passages themselves, so retrieval cannot be blamed;
  only answerable questions take part.

``--llm ollama`` runs a real local model and only after the memory gate of ``llm_smoke.py`` passes;
otherwise the run is refused (exit 3) and nothing is loaded. ``--llm stub`` replaces the model by a
deterministic stub that quotes the first passage with a valid citation and never refuses. Its
numbers show that the wiring works and say nothing about any model's quality; the report says so.
As elsewhere, unreviewed labels only give an exploratory result.

    uv run python eval/scripts/generation_eval.py --embed-dir <e5 dir> --llm stub \
        --include-unreviewed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from llm_smoke import gate, installed_size
from retrieval_eval import ROOT, write_config

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.cli.main import main as rag
from rag_engine.container import DefaultContainer
from rag_engine.domain import (
    AnswerStatus,
    AskStatus,
    ErrorCode,
    RagError,
    RetrievalMode,
    RetrievalQuery,
)
from rag_engine.evaluation.dataset import (
    Question,
    accepted_questions,
    load_dataset,
    validate_dataset,
)
from rag_engine.evaluation.generation import GenerationSample, score_answer, summarize_generation
from rag_engine.evaluation.judge import LexicalJudge
from rag_engine.evaluation.llm_run import Asker, evaluate_answers, summarize_records
from rag_engine.generation.generator import AnswerGenerator, GenerationPolicy
from rag_engine.pipeline import PipelineAnswer, PipelineLimits, RagPipeline
from rag_engine.ports import GenerationRequest, GenerationResult, ModelLimits
from rag_engine.system import SystemClock, system_id_factory
from rag_engine.testing import Reply, ScriptedLLM, evidence_context


class Recorder:
    """Passes questions to a pipeline and keeps each reply, so one run feeds both reports."""

    def __init__(self, pipeline: Asker) -> None:
        self._pipeline = pipeline
        self.replies: list[PipelineAnswer] = []

    async def ask(self, query: RetrievalQuery) -> PipelineAnswer:
        reply = await self._pipeline.ask(query)
        self.replies.append(reply)
        return reply


def measure(
    pipeline: Asker, questions: list[Question]
) -> tuple[list[GenerationSample], dict[str, object]]:
    """One e2e pass: the mechanical samples and the quality metrics (lexical judge)."""
    recorder = Recorder(pipeline)
    judge = LexicalJudge()

    def make_query(question: Question) -> RetrievalQuery:
        return RetrievalQuery(
            request_id=system_id_factory().request_id(),
            text=question.question,
            mode=RetrievalMode.DENSE,
            top_k=5,
        )

    records = asyncio.run(evaluate_answers(recorder, questions, judge, make_query))
    samples = [score_answer(q, a) for q, a in zip(questions, recorder.replies, strict=True)]
    return samples, asdict(summarize_records(records, judge_id=judge.judge_id))


_FIRST_PASSAGE = re.compile(r"\[E1\][^\n]*\n([^\n]+)")


class CitingStub(ScriptedLLM):
    """Quotes the start of the first passage with a valid citation; never refuses."""

    def __init__(self) -> None:
        super().__init__([], limits=ModelLimits(8192, 512))

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        match = _FIRST_PASSAGE.search(request.prompt)
        quote = " ".join(match.group(1).split()[:14]) if match else "no passage"
        self._script.append(Reply(f"{quote} [E1]."))
        return await super().generate(request)


def gold_passages(question: Question, corpus: Path) -> list[str]:
    """The paragraph of each gold document that contains its anchor."""
    passages = []
    for ref in question.gold:
        text = (corpus / ref.doc).read_text(encoding="utf-8")
        paragraph = next((p for p in text.split("\n\n") if ref.anchor in p), ref.anchor)
        passages.append(" ".join(paragraph.split()))
    return passages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "eval/datasets/retrieval_mini_v1.json"
    )
    parser.add_argument("--mode", choices=("e2e", "oracle"), default="e2e")
    parser.add_argument("--llm", choices=("stub", "ollama", "openai_compat"), default="stub")
    parser.add_argument("--base-url", default=None, help="for openai_compat, e.g. https://host/v1")
    parser.add_argument("--api-key-env", default=None, help="name of the variable holding the key")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    corpus_dir = ROOT / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.md")}
    if not validate_dataset(dataset, corpus).ok:
        sys.exit("dataset invalid")
    pool = (
        [q for q in dataset.questions if q.review.status != "rejected"]
        if args.include_unreviewed
        else accepted_questions(dataset)
    )
    questions = [q for q in pool if q.answerable] if args.mode == "oracle" else list(pool)
    if not questions:
        sys.exit("no accepted questions; review the dataset or use --include-unreviewed")

    llm_section = ""
    if args.llm == "ollama":
        size = asyncio.run(installed_size("http://127.0.0.1:11434", args.model))
        if size is None:
            sys.exit(f"{args.model} is not installed")
        ok, report = gate(size)
        if not ok:
            print(json.dumps({"result": "refused_by_resource_gate", **report}, indent=2))
            sys.exit(3)
        llm_section = (
            f'[llm]\nenabled = true\nprovider = "ollama"\nmodel = "{args.model}"\n'
            f"context_window = {args.context_window}\nmax_new_tokens = 256\n"
        )

    if args.llm == "openai_compat":
        if not args.base_url:
            sys.exit("--base-url is required for openai_compat")
        key = f'api_key_env = "{args.api_key_env}"\n' if args.api_key_env else ""
        llm_section = (
            f'[llm]\nenabled = true\nprovider = "openai_compat"\nmodel = "{args.model}"\n'
            f'base_url = "{args.base_url}"\n{key}'
            f"context_window = {args.context_window}\nmax_new_tokens = 256\n"
        )

    container = DefaultContainer()
    samples: list[GenerationSample] = []
    quality: dict[str, object] | None = None
    with tempfile.TemporaryDirectory(prefix="rag-gen-") as folder:
        work = Path(folder)
        config, db = work / "rag.toml", work / "rag.db"
        write_config(config, args.embed_dir.resolve(), 'mode = "dense"')
        if llm_section:
            config.write_text(config.read_text() + llm_section, encoding="utf-8")
        prefix = ["--config", str(config), "--db", str(db)]
        for step in (["source", "add", "corpus", str(corpus_dir)], ["ingest", "corpus", "--json"]):
            if rag([*prefix, *step], container=container) != 0:
                sys.exit(f"rag {step[0]} failed")
        if args.llm != "stub":
            with container.ask(config, db) as services:
                samples, quality = measure(services.pipeline, questions)
        else:
            with container.search(config, db) as searcher:
                stub = CitingStub()
                generator = AnswerGenerator(
                    stub, policy=GenerationPolicy(unverified_on_invalid=True)
                )
                pipeline = RagPipeline(
                    retrieval=searcher.retrieval,
                    builder=searcher.context,
                    context_policy=searcher.context_policy,
                    counter=HeuristicTokenCounter(),
                    gate=searcher.gate,
                    generator=generator,
                    clock=SystemClock(),
                    limits=PipelineLimits(total_timeout_s=60.0),
                )
                if args.mode == "oracle":
                    samples = [oracle(generator, q, corpus_dir) for q in questions]
                else:
                    samples, quality = measure(pipeline, questions)

    summary = summarize_generation(samples)
    result = {
        "mode": args.mode,
        "llm": args.llm if args.llm == "stub" else args.model,
        "dataset": dataset.identifier,
        "exploratory": args.include_unreviewed,
        "summary": asdict(summary),
        "note": (
            "STUB: the model is a deterministic citing stub; these numbers test the wiring only "
            "and say nothing about the quality of any model."
            if args.llm == "stub"
            else "Real model run through the resource gate."
        ),
        "quality": quality,
        "quality_caveat": (
            "answer correctness is a match against reference facts; faithfulness, hallucination "
            "and citation precision use the lexical judge (a screen, not an entailment check, "
            "uncalibrated against people) and unreviewed labels"
        ),
        "not_measured": (
            ["faithfulness", "hallucination rate", "answer correctness"] if quality is None else []
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def oracle(generator: AnswerGenerator, question: Question, corpus: Path) -> GenerationSample:
    ids = system_id_factory()
    context = evidence_context(
        gold_passages(question, corpus),
        document_id=ids.document_id(),
        version_id=ids.document_version_id(),
    )
    try:
        generated = asyncio.run(generator.answer(ids.request_id(), question.question, context))
    except RagError as error:
        degenerate = error.code is ErrorCode.E_DEGENERATE_ANSWER
        return GenerationSample(
            question.id, question.answerable, AskStatus.ERROR, True, degenerate, error.code.value
        )
    status = {
        AnswerStatus.ANSWERED: AskStatus.ANSWERED,
        AnswerStatus.REFUSED: AskStatus.INSUFFICIENT_EVIDENCE,
        AnswerStatus.UNVERIFIED: AskStatus.UNVERIFIED,
    }[generated.status]
    return GenerationSample(question.id, question.answerable, status, True, False, None)


if __name__ == "__main__":
    main()
