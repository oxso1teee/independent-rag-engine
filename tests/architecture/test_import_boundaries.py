"""The import-linter configuration must reject real violations."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]


def run_linter(
    config: Path, cwd: Path, extra_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    code = (
        "import sys; from importlinter.cli import lint_imports; "
        f"sys.exit(lint_imports(config_filename={str(config)!r}))"
    )
    env_path = str(extra_path) if extra_path else ""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=cwd,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_project_architecture_is_valid() -> None:
    result = run_linter(PROJECT / ".importlinter", PROJECT, PROJECT / "src")
    assert result.returncode == 0, result.stdout + result.stderr


def _make_package(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")


CONFIG = """
[importlinter]
root_package = demo
include_external_packages = True

[importlinter:contract:layers]
name = layers
type = layers
layers =
    demo.app
    demo.ports
    demo.domain

[importlinter:contract:pure-domain]
name = domain stays pure
type = forbidden
source_modules = demo.domain
forbidden_modules =
    sqlite3
    demo.app
"""


@pytest.mark.parametrize(
    ("violating_file", "content"),
    [
        ("demo/domain/models.py", "from demo import app\n"),
        ("demo/domain/models.py", "import sqlite3\n"),
        ("demo/ports/store.py", "from demo.app import main\n"),
        ("demo/domain/a.py", "from demo.ports import store\n"),
    ],
)
def test_violations_fail_the_linter(tmp_path: Path, violating_file: str, content: str) -> None:
    files = {
        "demo/__init__.py": "",
        "demo/domain/__init__.py": "",
        "demo/domain/models.py": "",
        "demo/domain/a.py": "",
        "demo/ports/__init__.py": "",
        "demo/ports/store.py": "",
        "demo/app/__init__.py": "",
        "demo/app/main.py": "",
        violating_file: content,
        "setup.cfg": CONFIG,
    }
    _make_package(tmp_path, files)
    result = run_linter(tmp_path / "setup.cfg", tmp_path, tmp_path)
    assert result.returncode != 0, result.stdout


def test_clean_package_passes(tmp_path: Path) -> None:
    _make_package(
        tmp_path,
        {
            "demo/__init__.py": "",
            "demo/domain/__init__.py": "",
            "demo/ports/__init__.py": "from demo.domain import models\n",
            "demo/domain/models.py": "import dataclasses\n",
            "demo/app/__init__.py": "from demo.ports import store\n",
            "demo/ports/store.py": "",
            "setup.cfg": CONFIG,
        },
    )
    result = run_linter(tmp_path / "setup.cfg", tmp_path, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("module", "content"),
    [
        ("chunking/offender.py", "from rag_engine.parsing import txt\n"),
        ("parsing/offender.py", "from rag_engine.chunking import fixed\n"),
        ("chunking/offender.py", "from rag_engine.storage import sqlite\n"),
        ("parsing/offender.py", "from rag_engine.config import load_settings\n"),
        ("chunking/offender.py", "from rag_engine.ingestion import processing\n"),
        ("parsing/offender.py", "import sqlite3\n"),
        ("chunking/offender.py", "import urllib.request\n"),
    ],
)
def test_parsing_and_chunking_stay_independent_compute(
    tmp_path: Path, module: str, content: str
) -> None:
    package = tmp_path / "rag_engine"
    shutil.copytree(
        PROJECT / "src" / "rag_engine", package, ignore=shutil.ignore_patterns("__pycache__")
    )
    clean = run_linter(PROJECT / ".importlinter", tmp_path, tmp_path)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    (package / module).write_text(content, encoding="utf-8")
    result = run_linter(PROJECT / ".importlinter", tmp_path, tmp_path)
    assert result.returncode != 0, result.stdout
