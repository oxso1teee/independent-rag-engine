"""``rag ask`` end to end: a real index, a scripted chat server, every outcome."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from rag_engine.cli.main import main
from rag_engine.container import DefaultContainer
from rag_engine.testing import HashingEmbedder

DIGEST = "c" * 64
DIMENSION = 32
QUESTION = "how are chunks stored in SQLite"
GOOD = "Chunks are stored in SQLite [E1]."

DOCUMENTS = {
    "alpha.md": "# Storage\n\nThe indexing pipeline stores chunks in SQLite and builds a vector "
    "matrix for exact search.\n",
    "beta/ranking.md": "# Ranking\n\nReciprocal rank fusion merges the dense and the lexical "
    "rankings into one list.\n",
    "menu.md": "# Menu\n\nWeekly menu: soup, bread and a salad with olive oil.\n",
}

type Reply = str | Callable[[], httpx.Response]


class Chat:
    """The scripted chat endpoint of the mocked Ollama; records every request body."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.bodies: list[dict[str, object]] = []

    def respond(self, body: dict[str, object]) -> httpx.Response:
        self.bodies.append(body)
        reply = self.replies.pop(0)
        if callable(reply):
            return reply()
        done = {"done": True, "done_reason": "stop", "prompt_eval_count": 120, "eval_count": 9}
        if body["stream"]:
            pieces = [reply[i : i + 6] for i in range(0, len(reply), 6)]
            lines = [{"message": {"content": p}, "done": False} for p in pieces]
            lines.append({"message": {"content": ""}, **done})
            return httpx.Response(200, content="".join(json.dumps(x) + "\n" for x in lines))
        return httpx.Response(200, json={"message": {"content": reply}, **done})


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], DefaultContainer, Chat]:
    embedder = HashingEmbedder(DIMENSION)
    chat = Chat()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        if request.url.path == "/api/chat":
            return chat.respond(json.loads(request.content))
        texts = json.loads(request.content)["input"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            vectors = pool.submit(asyncio.run, embedder.embed_documents(texts)).result()
        return httpx.Response(200, json={"embeddings": vectors.tolist()})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            base_url=kwargs["base_url"], transport=httpx.MockTransport(handler)
        ),
    )
    root = tmp_path / "notes"
    for name, text in DOCUMENTS.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    config = tmp_path / "rag.toml"
    config.write_text(
        "[parsing]\nisolate = []\n[embedding]\nenabled = true\n"
        f'provider = "ollama"\nmodel = "test"\nrevision = "{DIGEST}"\n'
        f'dimension = {DIMENSION}\nbase_url = "http://test"\n'
        '[llm]\nenabled = true\nprovider = "ollama"\nmodel = "chat-test"\n'
        'context_window = 4096\nmax_new_tokens = 256\nbase_url = "http://test"\n'
    )
    container = DefaultContainer()
    prefix = ["--config", str(config), "--db", str(tmp_path / "rag.db")]
    assert main([*prefix, "source", "add", "notes", str(root)], container=container) == 0
    assert main([*prefix, "ingest", "notes", "--json"], container=container) == 0
    return prefix, container, chat


Workspace = tuple[list[str], DefaultContainer, Chat]


def ask(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str, str]:
    prefix, container, _ = workspace
    capsys.readouterr()
    code = main([*prefix, "ask", *args], container=container)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_an_answer_names_its_sources_and_exits_zero(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.append(GOOD)
    code, out, _ = ask(workspace, capsys, QUESTION, "--mode", "dense")
    assert code == 0 and GOOD in out and "Sources:" in out and "notes/alpha.md" in out
    body = workspace[2].bodies[0]
    assert body["think"] is False and body["keep_alive"] == "0"  # the safe resource defaults
    assert "Question: " + QUESTION in body["messages"][1]["content"]  # type: ignore[index]


def test_json_document_has_status_citations_provenance_and_diagnostics(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.append(GOOD)
    code, out, _ = ask(workspace, capsys, QUESTION, "--mode", "dense", "--json", "--sources")
    data = json.loads(out)
    assert code == 0 and data["status"] == "answered" and data["text"] == GOOD
    (citation,) = data["citations"]
    assert citation["evidence_id"] == "E1" and citation["path"] == "alpha.md"
    assert citation["heading_path"] == ["Storage"] and citation["char_start"] < citation["char_end"]
    assert data["sources"][0]["cited"] is True
    diag = data["diagnostics"]
    assert diag["llm_called"] and diag["usage"]["prompt_tokens"] == 120
    assert {"retrieval", "context", "generation"} <= set(diag["stage_ms"])
    assert diag["gate"]["reasons"] == ["uncalibrated"] and diag["context"]["evidence"] >= 1


def test_the_model_declining_is_insufficient_evidence_with_its_own_exit_code(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.append("<<INSUFFICIENT>>")
    code, out, _ = ask(workspace, capsys, "what is the wifi password")
    assert code == 10 and "do not contain enough information" in out and "[E" not in out


def test_a_question_nothing_matches_never_calls_the_model(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = ask(workspace, capsys, "qqqqzzzz", "--mode", "lexical", "--json")
    assert code == 10 and json.loads(out)["reasons"] == ["no_hits"]
    assert workspace[2].bodies == []  # no request ever reached the chat server


def test_fabricated_citations_are_unverified_and_the_text_is_hidden_unless_asked(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.extend(
        ["Chunks are stored in SQLite [E9].", "Chunks live in SQLite [E9]."]
    )
    code, out, _ = ask(workspace, capsys, QUESTION, "--mode", "dense")
    assert code == 11 and "No verified answer" in out and "Chunks are stored" not in out
    workspace[2].replies.extend(
        ["Chunks are stored in SQLite [E9].", "Chunks live in SQLite [E9]."]
    )
    code, out, _ = ask(workspace, capsys, QUESTION, "--mode", "dense", "--show-unverified")
    assert code == 11 and "do not rely on it" in out and "Chunks live in SQLite" in out


def test_a_server_failure_is_an_error_with_the_provider_exit_code(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.extend([lambda: httpx.Response(404, text="model not found")])
    code, out, err = ask(workspace, capsys, QUESTION, "--mode", "dense")
    assert code == 6 and "E_PROVIDER_UNAVAILABLE" in err and out == ""


def test_a_verified_stream_prints_only_the_checked_text(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.append(GOOD)
    code, out, _ = ask(workspace, capsys, QUESTION, "--mode", "dense", "--stream")
    assert code == 0 and out.count(GOOD) == 1 and "Sources:" in out


def test_a_streamed_json_run_is_one_event_per_line_ending_in_final(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.append(GOOD)
    code, out, _ = ask(workspace, capsys, QUESTION, "--mode", "dense", "--stream", "--json")
    events = [json.loads(line) for line in out.splitlines()]
    kinds = [e["event"] for e in events]
    assert code == 0 and kinds[0] == "evidence" and kinds[-1] == "final"
    assert kinds[-2] == "citations" and kinds.count("delta") == 1
    assert events[kinds.index("delta")]["provisional"] is False
    assert events[-1]["status"] == "answered" and events[-1]["text"] == GOOD


def test_a_live_stream_shows_a_draft_and_withdraws_it_when_the_citations_fail(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace[2].replies.append("Chunks are stored in SQLite [E9].")
    code, out, err = ask(workspace, capsys, QUESTION, "--mode", "dense", "--stream", "--live")
    assert code == 11 and "Chunks are stored" in out and "withdrawn" in err


def test_without_a_configured_model_the_command_explains_how_to_enable_one(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, container, _ = workspace
    bare = tmp_path / "bare.toml"
    bare.write_text("[parsing]\nisolate = []\n")
    capsys.readouterr()
    code = main(
        ["--config", str(bare), "--db", str(tmp_path / "rag.db"), "ask", "q"], container=container
    )
    assert code == 3 and "[llm]" in capsys.readouterr().err


@pytest.mark.parametrize("args", [("   ",), (QUESTION, "--live"), (QUESTION, "--mode", "fuzzy")])
def test_bad_arguments_are_usage_errors(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], args: tuple[str, ...]
) -> None:
    code, _, err = ask(workspace, capsys, *args)
    assert code == 2 and "E_QUERY_INVALID" in err


def test_a_missing_database_is_reported_not_crashed(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    prefix, container, _ = workspace
    capsys.readouterr()
    code = main(
        [prefix[0], prefix[1], "--db", str(tmp_path / "none.db"), "ask", "q"], container=container
    )
    assert code == 4
