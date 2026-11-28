"""A document with a table border or a rule must index; it used to end as a dead job."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from rag_engine.cli.main import main
from rag_engine.container import DefaultContainer
from rag_engine.testing import HashingEmbedder

DIGEST = "f" * 64
BORDER = "\n".join("+" + "-" * 70 + "+" for _ in range(80))


def test_a_grid_table_border_does_not_stop_a_document_from_indexing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    embedder = HashingEmbedder(32)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        texts = json.loads(request.content)["input"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            vectors = pool.submit(asyncio.run, embedder.embed_documents(texts)).result()
        return httpx.Response(200, json={"embeddings": vectors.tolist()})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(base_url=kw["base_url"], transport=httpx.MockTransport(handler)),
    )
    root = tmp_path / "notes"
    root.mkdir()
    (root / "grid.md").write_text(
        "# Report\n\nThe backup policy keeps copies for two weeks.\n\n## Table\n\n"
        f"{BORDER}\n\n## After\n\nRestores are tested every month.\n"
    )
    config = tmp_path / "rag.toml"
    config.write_text(
        "[parsing]\nisolate = []\n[chunking]\ntarget_tokens = 60\nmax_tokens = 80\nmin_tokens = 8\n"
        '[embedding]\nenabled = true\nprovider = "ollama"\nmodel = "test"\n'
        f'revision = "{DIGEST}"\ndimension = 32\nbase_url = "http://test"\n'
    )
    prefix = ["--config", str(config), "--db", str(tmp_path / "rag.db")]
    container = DefaultContainer()
    assert main([*prefix, "source", "add", "notes", str(root)], container=container) == 0
    assert main([*prefix, "ingest", "notes", "--json"], container=container) == 0
    capsys.readouterr()
    assert main([*prefix, "search", "how long are backups kept", "--json"],
                container=container) == 0  # fmt: skip
    hits = json.loads(capsys.readouterr().out)
    assert hits["hits"], "the real text is searchable"
