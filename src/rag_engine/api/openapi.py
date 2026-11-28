"""The OpenAPI document of the API, generated without starting a server.

``docs/openapi.json`` is a committed snapshot of it; a test fails when the API changes without the
snapshot, so every change to the contract shows up in review. Regenerate it with::

    uv run python -m rag_engine.api.openapi docs/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

from rag_engine.api.app import create_app
from rag_engine.cli.context import Container
from rag_engine.config import Settings


def build_document() -> dict[str, Any]:
    # Describing the contract calls no route, so no services are needed behind it.
    app = create_app(cast(Container, object()), None, Path("rag.db"), Settings(), token=None)
    document = app.openapi()
    assert isinstance(document, dict)  # noqa: S101 - FastAPI returns a plain dict
    return document


def render() -> str:
    return json.dumps(build_document(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m rag_engine.api.openapi OUTPUT.json\n")
        return 2
    Path(args[0]).write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
