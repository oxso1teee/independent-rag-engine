"""The documentation names every endpoint and command; drift fails the build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_engine.api.openapi import build_document
from rag_engine.cli.main import build_app
from rag_engine.container import DefaultContainer

pytestmark = pytest.mark.api

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _shape(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_every_endpoint_is_in_the_api_documentation() -> None:
    text = _shape((DOCS / "API.md").read_text(encoding="utf-8"))
    for path, operations in build_document()["paths"].items():
        for method in operations:
            documented = re.findall(rf"{method.upper()}[^`|]*?{re.escape(_shape(path))}\b", text)
            assert documented or f"`{method.upper()} {_shape(path)}" in text, (
                f"{method.upper()} {path} is undocumented"
            )


def test_every_command_is_in_the_cli_documentation() -> None:
    text = (DOCS / "CLI.md").read_text(encoding="utf-8")
    result = CliRunner().invoke(build_app(DefaultContainer()), ["--help"])
    commands = re.findall(r"^  ([a-z-]+)\s", result.output.split("Commands:")[1], re.M)
    assert len(commands) >= 12
    for command in commands:
        assert f"rag {command}" in text, f"`rag {command}` is undocumented"


def test_the_documented_json_examples_are_valid() -> None:
    text = (DOCS / "API.md").read_text(encoding="utf-8")
    for block in re.findall(r"```json\n(.*?)\n```", text, re.S):
        assert isinstance(json.loads(block), dict)
