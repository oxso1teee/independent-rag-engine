"""A real container behind the HTTP API, with the embedding and chat servers mocked."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.cli.main import main
from rag_engine.config import load_settings
from rag_engine.container import DefaultContainer
from rag_engine.testing import HashingEmbedder

DIGEST = "c" * 64
DIMENSION = 32
TOKEN = "t" * 32

type Reply = str | Callable[[], httpx.Response]

DOCUMENTS = {
    "alpha.md": "# Storage\n\nThe indexing pipeline stores chunks in SQLite and builds a vector "
    "matrix for exact search.\n",
    "beta/ranking.md": "# Ranking\n\nReciprocal rank fusion merges the dense and the lexical "
    "rankings into one list.\n",
    "menu.md": "# Menu\n\nWeekly menu: soup, bread and a salad with olive oil.\n",
}


@dataclass
class Chat:
    replies: list[Reply] = field(default_factory=list)
    bodies: list[dict[str, object]] = field(default_factory=list)

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


@dataclass
class Api:
    client: TestClient
    chat: Chat
    root: Path
    config: Path
    container: DefaultContainer
    prefix: list[str]

    def seed(self) -> None:
        """Register and ingest the sample documents with the command line (a CLI-managed source)."""
        for name, text in DOCUMENTS.items():
            target = self.root / "notes" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        assert main([*self.prefix, "source", "add", "notes", str(self.root / "notes")],
                    container=self.container) == 0  # fmt: skip
        assert main([*self.prefix, "ingest", "notes", "--json"], container=self.container) == 0


def write_config(path: Path, *, llm: bool = True, api: str = "") -> None:
    text = (
        "[parsing]\nisolate = []\n[embedding]\nenabled = true\n"
        f'provider = "ollama"\nmodel = "test"\nrevision = "{DIGEST}"\n'
        f'dimension = {DIMENSION}\nbase_url = "http://test"\n'
    )
    if llm:
        text += (
            '[llm]\nenabled = true\nprovider = "ollama"\nmodel = "chat-test"\n'
            'context_window = 4096\nmax_new_tokens = 256\nbase_url = "http://test"\n'
        )
    path.write_text(text + api)


def make_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm: bool = True,
    api: str = "",
    token: str | None = None,
) -> Iterator[Api]:
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
    config = tmp_path / "rag.toml"
    write_config(config, llm=llm, api=api)
    db = tmp_path / "data" / "rag.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    container = DefaultContainer()
    settings = load_settings(config)
    app = create_app(container, config, db, settings, token=token)
    with TestClient(app, raise_server_exceptions=False) as client:
        if token is not None:
            client.headers["Authorization"] = f"Bearer {token}"
        yield Api(
            client, chat, tmp_path, config, container, ["--config", str(config), "--db", str(db)]
        )


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Api]:
    yield from make_api(tmp_path, monkeypatch)
