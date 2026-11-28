"""The real-provider check script, run against a fake OpenAI-compatible server.

This tests the script, not a provider: a scripted endpoint answers with citations, refuses what the
dataset marks unanswerable, rejects a wrong key or an unknown model, and the script has to report
every stage, keep the key out of its output and refuse to start without approval or on a corpus that
is not public.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.evaluation.dataset import load_dataset
from rag_engine.testing import HashingEmbedder

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
KEY = "-".join(["fake", "credential", "for", "tests"])  # built, so no scanner mistakes it for a key
DIGEST = "9" * 64
MODEL = "fake-model"
FIRST = re.compile(r"\[E1\][^\n]*\n([^\n]+)")


def load_script() -> Any:
    sys.path.insert(0, str(ROOT / "eval" / "scripts"))
    spec = importlib.util.spec_from_file_location("llm_e2e", ROOT / "eval/scripts/llm_e2e.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_e2e"] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def install(monkeypatch: pytest.MonkeyPatch, unanswerable: list[str]) -> list[httpx.Request]:
    embedder = HashingEmbedder(32)
    seen: list[httpx.Request] = []

    def completion(body: dict[str, Any]) -> str:
        prompt = " ".join(str(m.get("content", "")) for m in body["messages"])
        if any(q in prompt for q in unanswerable):
            return REFUSAL_SENTINEL
        match = FIRST.search(prompt)
        quote = " ".join(match.group(1).split()[:14]) if match else "nothing"
        return f"{quote} [E1]."

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "127.0.0.1" and request.url.port == 9:
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        if path == "/api/embed":
            texts = json.loads(request.content)["input"]
            with ThreadPoolExecutor(max_workers=1) as pool:
                vectors = pool.submit(asyncio.run, embedder.embed_documents(texts)).result()
            return httpx.Response(200, json={"embeddings": vectors.tolist()})
        assert path == "/v1/chat/completions"
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})
        body = json.loads(request.content)
        if body["model"] != MODEL:
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        text = completion(body)
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": text[i : i + 8]}, "finish_reason": None}]}
                for i in range(0, len(text), 8)
            ]
            lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
            return httpx.Response(200, content="".join(lines) + "data: [DONE]\n\n")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 12},
            },
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(
            base_url=kw["base_url"],
            headers=kw.get("headers"),
            transport=httpx.MockTransport(handler),
        ),
    )
    return seen


def configure(script: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def write_config(path: Path, _model_dir: Path, retrieval: str) -> None:
        path.write_text(
            '[parsing]\nisolate = []\n[embedding]\nenabled = true\nprovider = "ollama"\n'
            f'model = "test"\nrevision = "{DIGEST}"\ndimension = 32\nbase_url = "http://embed.test"\n'
            f"[retrieval]\n{retrieval}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(script, "write_config", write_config)


def args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--approved", "--embed-dir", str(tmp_path), "--base-url", "http://llm.test/v1",
        "--model", MODEL, "--api-key-env", "FAKE_KEY", "--skip-memory-gate", *extra,
    ]  # fmt: skip


def test_the_script_checks_every_stage_and_never_prints_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load_script()
    configure(script, monkeypatch)
    monkeypatch.setenv("FAKE_KEY", KEY)
    dataset = load_dataset(ROOT / "eval/datasets/retrieval_mini_v2.json")
    install(monkeypatch, [q.question for q in dataset.questions if not q.answerable])
    out = tmp_path / "report.json"
    code = script.main(args(tmp_path, "--output", str(out)))
    printed = capsys.readouterr()
    report = json.loads(out.read_text())
    checks = {c["name"]: c for c in report["checks"]}
    assert KEY not in printed.out and KEY not in printed.err and KEY not in out.read_text()
    for name in (
        "ingestion", "retrieval returns evidence", "generation and citation validation",
        "insufficient evidence is reported, not invented",
        "streaming ends with one authoritative final event",
        "cancellation releases the request and the pipeline still answers",
        "a wrong key is a clean error", "an unknown model is a clean error",
        "an unreachable server is a clean error", "the key appears in no error text",
    ):  # fmt: skip
        assert checks[name]["passed"], checks[name]
    assert code == 0 and report["all_passed"] and "Ollama" in report["note"]
    assert report["numbers"]["quality_exploratory"]["prompt_tokens"] > 0


def test_it_refuses_without_approval_a_private_corpus_or_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load_script()
    monkeypatch.setenv("FAKE_KEY", KEY)
    no_approval = [a for a in args(tmp_path) if a != "--approved"]
    assert script.main(no_approval) == 3 and "--approved" in capsys.readouterr().err
    private = tmp_path / "private"
    private.mkdir()
    assert script.main(args(tmp_path, "--corpus", str(private))) == 3
    assert "public corpus" in capsys.readouterr().err
    monkeypatch.delenv("FAKE_KEY")
    assert script.main(args(tmp_path)) == 3 and "is not set" in capsys.readouterr().err
