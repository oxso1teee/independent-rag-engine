"""``rag search`` end to end: a real index built by ``rag ingest``, then every mode and filter."""

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

DIGEST = "c" * 64
DIMENSION = 32

DOCUMENTS = {
    "alpha.md": "# Storage\n\nThe indexing pipeline stores chunks in SQLite and builds a vector "
    "matrix for exact search.\n",
    "beta/ranking.md": "# Ranking\n\nReciprocal rank fusion merges the dense and the lexical "
    "rankings into one list.\n",
    "beta/serials.md": "# Calibration\n\nXylophone calibration table for serial number ZX4471 "
    "instruments.\n",
    "menu.md": "# Menu\n\nWeekly menu: soup, bread and a salad with olive oil.\n",
}


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], DefaultContainer]:
    embedder = HashingEmbedder(DIMENSION)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        texts = json.loads(request.content)["input"]
        with ThreadPoolExecutor(max_workers=1) as pool:  # the handler runs inside an event loop
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
    )
    container = DefaultContainer()
    prefix = ["--config", str(config), "--db", str(tmp_path / "rag.db")]
    assert main([*prefix, "source", "add", "notes", str(root)], container=container) == 0
    assert main([*prefix, "ingest", "notes", "--json"], container=container) == 0
    return prefix, container


def search(
    workspace: tuple[list[str], DefaultContainer],
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, dict[str, object]]:
    prefix, container = workspace
    capsys.readouterr()
    code = main([*prefix, "search", *args, "--json"], container=container)
    return code, json.loads(capsys.readouterr().out)


def paths(data: dict[str, object]) -> list[str]:
    hits = data["hits"]
    assert isinstance(hits, list)
    return [hit["path"] for hit in hits]


def test_dense_search_finds_the_passage_by_meaning(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    code, data = search(workspace, capsys, "--mode", "dense", "how are chunks stored in SQLite")
    assert code == 0 and paths(data)[0] == "alpha.md"
    first = data["hits"][0]  # type: ignore[index]
    assert first["rank"] == 1 and first["source"] == "notes"
    assert first["scores"]["dense"] is not None and first["scores"]["lexical"] is None
    assert "SQLite" in first["text"] and first["heading_path"] == ["Storage"]
    assert data["index_generation"] >= 1 and data["embedding_model"]  # type: ignore[operator]


def test_hybrid_search_finds_a_rare_identifier(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    code, data = search(workspace, capsys, "--mode", "hybrid", "ZX4471")
    assert code == 0 and paths(data)[0] == "beta/serials.md"
    scores = data["hits"][0]["scores"]  # type: ignore[index]
    assert scores["lexical"] is not None and scores["fused"] is not None


def test_lexical_search_uses_no_embedding_and_filters_narrow_the_result(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    code, data = search(workspace, capsys, "--mode", "lexical", "rankings lists")
    assert code == 0 and paths(data) == ["beta/ranking.md"]
    code, data = search(workspace, capsys, "--mode", "hybrid", "list", "--path-prefix", "menu")
    assert code == 0 and set(paths(data)) <= {"menu.md"}
    code, data = search(workspace, capsys, "--mode", "dense", "x", "--path-glob", "beta/*.md")
    assert code == 0 and set(paths(data)) <= {"beta/ranking.md", "beta/serials.md"}
    code, data = search(workspace, capsys, "--mode", "dense", "x", "--source", "nowhere")
    assert code == 0 and data["hits"] == [] and "filter_matched_zero" in data["fallbacks"]  # type: ignore[operator]


def test_debug_adds_diagnostics_and_a_plain_run_does_not(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    _, plain = search(workspace, capsys, "--mode", "hybrid", "ZX4471")
    assert "diagnostics" not in plain
    _, debug = search(workspace, capsys, "--mode", "hybrid", "ZX4471", "--debug")
    diagnostics = debug["diagnostics"]
    assert diagnostics["mode"] == "hybrid" and diagnostics["fused_candidates"] >= 1  # type: ignore[index]
    assert set(diagnostics["timings_ms"]) >= {"dense", "lexical", "fuse"}  # type: ignore[index]


def test_the_same_search_returns_the_same_hits(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    _, first = search(workspace, capsys, "--mode", "hybrid", "indexing pipeline")
    _, second = search(workspace, capsys, "--mode", "hybrid", "indexing pipeline")
    assert (
        first["hits"] == second["hits"] and first["index_generation"] == second["index_generation"]
    )


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (["--mode", "semantic", "text"], "E_QUERY_INVALID"),
        (["   "], "E_QUERY_INVALID"),
        (["text", "--language", "xx"], "E_FILTER_INVALID"),
        (["text", "--kind", "poem"], "E_FILTER_INVALID"),
        (["text", "--modified-after", "yesterday"], "E_FILTER_INVALID"),
        (
            ["text", "--modified-after", "2026-02-01", "--modified-before", "2026-01-01"],
            "E_FILTER_INVALID",
        ),
        (["text", "--doc-id", "not-an-id"], "E_FILTER_INVALID"),
        (["text", "--path-glob", "[unterminated"], "E_FILTER_INVALID"),
    ],
)
def test_bad_input_is_a_user_error_with_a_stable_code(
    workspace: tuple[list[str], DefaultContainer],
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    code: str,
) -> None:
    exit_code, data = search(workspace, capsys, *args)
    assert exit_code == 2
    assert data["error"]["code"] == code  # type: ignore[index]


def test_text_output_lists_rank_score_path_and_a_preview(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    prefix, container = workspace
    capsys.readouterr()
    assert main([*prefix, "search", "--mode", "hybrid", "ZX4471"], container=container) == 0
    out = capsys.readouterr().out
    assert " 1 " in out and "notes/beta/serials.md" in out and "[Calibration]" in out
    assert "ZX4471" in out


def test_searching_before_anything_is_indexed_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--db", str(tmp_path / "missing.db"), "search", "text", "--json"],
        container=DefaultContainer(),
    )
    assert code != 0
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "E_SOURCE_NOT_FOUND"


def test_forcing_reranking_without_a_reranker_is_a_noted_fallback_not_an_error(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    code, forced = search(workspace, capsys, "--mode", "hybrid", "ZX4471", "--rerank")
    assert code == 0 and paths(forced)[0] == "beta/serials.md"
    assert "rerank_skipped:not_configured" in forced["fallbacks"]  # type: ignore[operator]
    code, off = search(workspace, capsys, "--mode", "hybrid", "ZX4471", "--no-rerank")
    assert code == 0 and off["fallbacks"] == [] and off["hits"] == forced["hits"]


def run_context(
    workspace: tuple[list[str], DefaultContainer],
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, dict[str, object]]:
    prefix, container = workspace
    capsys.readouterr()
    code = main([*prefix, "context", *args, "--json"], container=container)
    return code, json.loads(capsys.readouterr().out)


def test_context_reports_the_measured_budget_and_the_evidence(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_engine.chunking.tokens import HeuristicTokenCounter
    from rag_engine.context.prompt import system_prompt_tokens

    code, data = run_context(
        workspace, capsys, "--mode", "hybrid", "ZX4471", "--window", "4096", "--max-new", "256"
    )
    assert code == 0
    budget: dict[str, int] = data["budget"]  # type: ignore[assignment]
    counter = HeuristicTokenCounter()
    assert budget["system"] == system_prompt_tokens(counter)
    assert budget["question"] == counter.count("ZX4471")
    assert budget["margin"] == 410  # 10 % of 4096 is above the 256 floor
    expected = 4096 - budget["system"] - budget["question"] - 256 - 410
    assert budget["evidence_budget"] == expected
    assert budget["tokens_used"] <= budget["evidence_budget"]
    first = data["evidence"][0]  # type: ignore[index]
    assert first["id"] == "E1" and first["path"] == "beta/serials.md"
    assert first["heading_path"] == ["Calibration"] and "ZX4471" in first["text"]
    report = data["report"]
    shown = {c for item in data["evidence"] for c in item["matched"]}  # type: ignore[attr-defined]
    dropped = {item["chunk_id"] for item in report["dropped"]}  # type: ignore[index]
    assert not shown & dropped and len(data["context_hash"]) == 64  # type: ignore[arg-type]


def test_a_window_that_leaves_no_room_is_a_resource_error_with_a_hint(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    code, data = run_context(workspace, capsys, "ZX4471", "--window", "300", "--max-new", "100")
    assert code == 7
    assert data["error"]["code"] == "E_CONTEXT_BUDGET"  # type: ignore[index]
    assert "hint" in data["error"]["context"]  # type: ignore[index]


def test_context_needs_a_window_from_the_command_or_the_configuration(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    code, data = run_context(workspace, capsys, "ZX4471")
    assert code == 3 and data["error"]["code"] == "E_CONFIG_INVALID"  # type: ignore[index]
    code, data = run_context(workspace, capsys, "ZX4471", "--window", "500", "--max-new", "600")
    assert code == 3
    code, data = run_context(workspace, capsys, "ZX4471", "--window", "4096", "--order", "random")
    assert code == 2 and data["error"]["code"] == "E_QUERY_INVALID"  # type: ignore[index]


def test_context_text_output_shows_the_budget_line_and_numbered_evidence(
    workspace: tuple[list[str], DefaultContainer], capsys: pytest.CaptureFixture[str]
) -> None:
    prefix, container = workspace
    capsys.readouterr()
    code = main(
        [*prefix, "context", "ZX4471", "--mode", "hybrid", "--window", "4096"], container=container
    )
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("budget: window 4096")
    assert "E1 " in out and "notes/beta/serials.md" in out
