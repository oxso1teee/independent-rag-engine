"""Minimal ``rag`` command; subcommands are added together with their features."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rag_engine import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Modular, local-first RAG engine.",
    )
    parser.add_argument("--version", action="version", version=f"rag {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
