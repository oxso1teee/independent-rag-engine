"""``rag init``, ``rag config``, ``rag backup`` and ``rag serve``."""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from rag_engine.cli.context import AskServices, Processing, SearchServices, Services
from rag_engine.cli.main import main as _main
from rag_engine.container import DefaultContainer

pytestmark = pytest.mark.e2e


def main(args: list[str], container: DefaultContainer | None = None) -> int:
    return _main(args, container=container or DefaultContainer())


class SpyServer(DefaultContainer):
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int | None]] = []

    def serve(
        self, config: Path | None, db: Path | None, *, host: str | None, port: int | None
    ) -> int:
        self.calls.append((host, port))
        return 0


def test_init_creates_a_valid_starter_config_and_refuses_to_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", "--dir", str(tmp_path / "proj")]) == 0
    config = tmp_path / "proj" / "rag.toml"
    assert config.is_file() and (tmp_path / "proj" / "data").is_dir()
    assert main(["--config", str(config), "config", "check"]) == 0
    capsys.readouterr()
    assert main(["init", "--dir", str(tmp_path / "proj"), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "E_CONFIG_INVALID"
    config.write_text("# mine\n")
    assert main(["init", "--dir", str(tmp_path / "proj"), "--force"]) == 0
    assert "[storage]" in config.read_text()


def test_config_check_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.toml"
    good.write_text(
        '[llm]\nenabled = true\nprovider = "ollama"\nmodel = "m"\ncontext_window = 4096\n'
    )
    capsys.readouterr()
    assert main(["--config", str(good), "config", "check", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"valid": True, "enabled": ["llm"]}
    assert main(["--config", str(good), "config", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["llm"]["keep_alive"] == "0" and shown["api"]["host"] == "127.0.0.1"
    bad = tmp_path / "bad.toml"
    bad.write_text("[llm]\nenabled = true\n")
    assert main(["--config", str(bad), "config", "check"]) == 3
    assert "context_window" in capsys.readouterr().err


def test_serving_beyond_the_loopback_without_a_token_refuses_to_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "public.toml"
    config.write_text('[api]\nhost = "0.0.0.0"\n')
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    capsys.readouterr()
    assert main(["--config", str(config), "serve"]) == 3  # nothing was bound
    assert "non-loopback" in capsys.readouterr().err
    assert main(["--config", str(config), "serve", "--host", "0.0.0.0"]) == 3  # noqa: S104


def test_backup_writes_a_readable_copy_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "rag.toml"
    config.write_text("[parsing]\nisolate = []\n")
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nSome text for the backup test.\n")
    prefix = ["--config", str(config), "--db", str(tmp_path / "rag.db")]
    assert main([*prefix, "source", "add", "notes", str(root)]) == 0
    destination = tmp_path / "backups" / "copy.db"
    capsys.readouterr()
    assert main([*prefix, "backup", str(destination), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bytes"] > 0
    with sqlite3.connect(destination) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT name FROM sources").fetchall() == [("notes",)]
    assert main([*prefix, "backup", str(destination)]) == 9  # integrity: refuses to overwrite
    assert (
        main([*prefix[:2], "--db", str(tmp_path / "none.db"), "backup", str(tmp_path / "x.db")])
        == 4
    )


def test_serve_hands_the_options_to_the_container() -> None:
    container = SpyServer()
    assert main(["serve", "--host", "127.0.0.1", "--port", "9000"], container) == 0
    assert container.calls == [("127.0.0.1", 9000)]


def _types(
    _: AbstractContextManager[AskServices], __: Processing, ___: SearchServices, ____: Services
) -> None:
    return None  # keeps the imported names used by the type checker
