from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import rag_engine
from rag_engine.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"rag {rag_engine.__version__}"


def test_no_arguments_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "ingest" in output and "source" in output and "status" in output


def test_unknown_command_is_a_usage_error() -> None:
    assert main(["nonsense"]) == 2


def test_package_metadata_is_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["name"] == "independent-rag-engine"
    assert project["requires-python"] == ">=3.12,<3.14"
    assert project["scripts"] == {"rag": "rag_engine.cli:main"}
    assert "version" in project["dynamic"]
