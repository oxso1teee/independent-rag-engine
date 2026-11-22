from __future__ import annotations

import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

import rag_engine
from rag_engine import container as composition_root
from rag_engine.cli import main
from rag_engine.cli.context import Services


class NeverOpened:
    """A container for commands that must not touch any infrastructure."""

    def open(self, config: Path | None, db: Path | None, *, read_only: bool) -> Iterator[Services]:
        raise AssertionError("this command must not open the services")


def run(*args: str) -> int:
    return main(list(args), container=NeverOpened())  # type: ignore[arg-type]


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--version") == 0
    assert capsys.readouterr().out.strip() == f"rag {rag_engine.__version__}"


def test_no_arguments_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert run() == 0
    output = capsys.readouterr().out
    assert "ingest" in output and "source" in output and "status" in output


def test_unknown_command_is_a_usage_error() -> None:
    assert run("nonsense") == 2


def test_package_metadata_is_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["name"] == "independent-rag-engine"
    assert project["requires-python"] == ">=3.12,<3.14"
    assert project["scripts"] == {"rag": "rag_engine.container:main"}
    assert "version" in project["dynamic"]


def test_the_composition_root_is_the_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    assert composition_root.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"rag {rag_engine.__version__}"
