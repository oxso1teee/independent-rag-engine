"""The isolated parsing worker (``python -m rag_engine.parsing.worker MEMORY_MB``).

The worker reads one request at a time from standard input (a JSON header, then the raw bytes of
the document), parses it with the parser the header names and writes the reply to the real
standard output. File descriptor 1 is pointed at ``/dev/null`` before anything else runs, so a
library that prints can never corrupt the protocol. The data segment is capped at start (a
document that needs more ends as a ``memory`` failure) and core dumps are disabled.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from importlib import import_module
from typing import Any, BinaryIO

from rag_engine.domain import ErrorCode, ParseReason, RagError, parse_failed
from rag_engine.parsing.wire import (
    decode_hints,
    dumps,
    encode_document,
    encode_error,
    frame,
    loads,
    read_frame,
)
from rag_engine.ports import Parser

EXIT_AFTER_MEMORY_ERROR = True


def limit_memory(megabytes: int) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - not available on this platform
        return
    limit = megabytes * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def build_parser(factory: str, args: Sequence[Any]) -> Parser:
    """``module:callable`` called with ``args``; the parent decides what the worker may load."""
    module_name, _, attribute = factory.partition(":")
    parser: Parser = getattr(import_module(module_name), attribute)(*args)
    return parser


def handle(
    header: dict[str, Any], data: bytes, parsers: dict[str, Parser]
) -> tuple[dict[str, Any], bool]:
    """``(reply, stop)``: the reply message and whether the worker must exit afterwards."""
    key = f"{header['factory']}|{dumps(header['args']).decode()}"
    try:
        parser = parsers.get(key)
        if parser is None:
            parser = parsers[key] = build_parser(header["factory"], header["args"])
        document = parser.parse(data, decode_hints(header["hints"]))
        return {"ok": encode_document(document)}, False
    except RagError as error:
        return {"error": encode_error(error)}, False
    except MemoryError:
        exhausted = parse_failed(ParseReason.MEMORY, "the document needs too much memory")
        return {"error": encode_error(exhausted)}, EXIT_AFTER_MEMORY_ERROR
    except RecursionError:
        too_deep = parse_failed(ParseReason.TOO_DEEP, "the document is nested too deeply")
        return {"error": encode_error(too_deep)}, False
    except ImportError:
        unsupported = RagError(
            ErrorCode.E_UNSUPPORTED_FORMAT, "the parser is not installed in the worker"
        )
        return {"error": encode_error(unsupported)}, False
    except Exception as exc:
        corrupt = parse_failed(
            ParseReason.CORRUPT, "the document could not be parsed", error_type=type(exc).__name__
        )
        return {"error": encode_error(corrupt)}, False


def serve(stdin: BinaryIO, stdout: BinaryIO) -> None:
    parsers: dict[str, Parser] = {}
    stdout.write(frame(dumps({"ready": True})))
    stdout.flush()
    while True:
        try:
            header = loads(read_frame(stdin))
            data = read_frame(stdin)
        except EOFError:
            return
        reply, stop = handle(header, data, parsers)
        try:
            payload = dumps({**reply, "stop": stop})
        except MemoryError:
            exhausted = parse_failed(ParseReason.MEMORY, "the result needs too much memory")
            payload, stop = dumps({"error": encode_error(exhausted), "stop": True}), True
        stdout.write(frame(payload))
        stdout.flush()
        if stop:
            return


def main(argv: Sequence[str]) -> int:
    memory_mb = int(argv[1]) if len(argv) > 1 else 1024
    stdout = os.fdopen(os.dup(1), "wb")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    limit_memory(memory_mb)
    serve(sys.stdin.buffer, stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
