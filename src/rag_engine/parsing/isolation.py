"""Running a parser in a separate worker process.

Formats whose libraries parse untrusted binary or markup input (PDF, DOCX, HTML) can run in a
long-lived worker. A document that exhausts memory, loops, crashes the interpreter or takes too
long ends only the worker: the caller gets ``E_PARSE_FAILED`` with the reason (``memory``,
``timeout`` or ``worker_failed``) and the next document starts a fresh one.

The worker gets a minimal environment (no secrets), a capped data segment and no core dumps; it
is not a syscall sandbox. Messages are length-prefixed JSON (see :mod:`rag_engine.parsing.wire`).
"""

from __future__ import annotations

import contextlib
import os
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Sequence
from typing import Any

from rag_engine.domain import ParsedDocument, ParseReason, RagError, parse_failed
from rag_engine.parsing.common import check_input_size
from rag_engine.parsing.wire import (
    MAX_FRAME,
    decode_document,
    decode_error,
    dumps,
    encode_hints,
    frame,
    loads,
)
from rag_engine.ports import ParseHints, Parser

_HEADER = struct.Struct(">Q")
DEFAULT_GRACE_S = 5.0  # added to the parse time limit: the parser enforces its own limit first
_START_TIMEOUT_S = 30.0


class _TimedOut(Exception):
    pass


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Kill the worker and collect it; safe to call twice."""
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):  # a killed process does exit
        process.wait(timeout=5)
    for stream in (process.stdin, process.stdout):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


class ParserWorker:
    """One worker process shared by every isolated parser; started on the first document."""

    def __init__(
        self,
        memory_mb: int = 1024,
        max_documents: int = 200,
        grace_s: float = DEFAULT_GRACE_S,
    ) -> None:
        self._memory_mb = memory_mb
        self._max_documents = max_documents
        self._grace_s = grace_s
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._finalizer: weakref.finalize[..., Any] | None = None
        self._served = 0

    # -- lifecycle -----------------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C.UTF-8",
        }

    def _start(self) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(  # noqa: S603 - our own interpreter and module, fixed argv
            [sys.executable, "-m", "rag_engine.parsing.worker", str(self._memory_mb)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._environment(),
            cwd=tempfile.gettempdir(),
            close_fds=True,
            start_new_session=True,
        )
        if self._finalizer is not None:
            self._finalizer.detach()
        self._finalizer = weakref.finalize(self, _stop, process)
        self._process, self._served = process, 0
        try:
            ready = loads(self._read(process, time.monotonic() + _START_TIMEOUT_S))
        except (_TimedOut, EOFError, ValueError, OSError):
            self._discard()
            raise parse_failed(
                ParseReason.WORKER_FAILED, "the parsing worker did not start"
            ) from None
        if not (isinstance(ready, dict) and ready.get("ready") is True):
            self._discard()
            raise parse_failed(ParseReason.WORKER_FAILED, "the parsing worker did not start")
        return process

    def _discard(self) -> int | None:
        process, self._process = self._process, None
        if process is None:
            return None
        _stop(process)
        return process.returncode

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()  # the worker leaves when its input ends
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                _stop(process)

    # -- one document --------------------------------------------------------------

    @staticmethod
    def _read(process: subprocess.Popen[bytes], deadline: float) -> bytes:
        assert process.stdout is not None  # noqa: S101 - created with a pipe
        fd = process.stdout.fileno()
        buffer = bytearray()
        size: int | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise _TimedOut
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise EOFError
            buffer += chunk
            if size is None and len(buffer) >= _HEADER.size:
                (size,) = _HEADER.unpack_from(buffer)
                if size > MAX_FRAME:
                    raise ValueError("frame too large")
            if size is not None and len(buffer) >= _HEADER.size + size:
                return bytes(buffer[_HEADER.size : _HEADER.size + size])

    def call(
        self, factory: str, args: Sequence[object], data: bytes, hints: ParseHints
    ) -> ParsedDocument:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                self._discard()
                process = self._start()
            deadline = time.monotonic() + hints.limits.timeout_s + self._grace_s
            header = dumps({"factory": factory, "args": list(args), "hints": encode_hints(hints)})
            try:
                assert process.stdin is not None  # noqa: S101 - created with a pipe
                process.stdin.write(frame(header) + frame(data))
                process.stdin.flush()
                reply = loads(self._read(process, deadline))
            except _TimedOut:
                self._discard()
                raise parse_failed(
                    ParseReason.TIMEOUT, "parsing took too long", limit=int(hints.limits.timeout_s)
                ) from None
            except (EOFError, OSError, ValueError):
                code = self._discard()
                raise parse_failed(
                    ParseReason.WORKER_FAILED,
                    "the parsing worker stopped unexpectedly",
                    exit_code=code if code is not None else 0,
                ) from None
            self._served += 1
            stops = isinstance(reply, dict) and reply.get("stop") is True
            if stops or self._served >= self._max_documents:
                self._discard()  # the worker asked to go, or a fresh one keeps long runs clean
            return self._result(reply)

    @staticmethod
    def _result(reply: Any) -> ParsedDocument:
        try:
            if isinstance(reply, dict) and "error" in reply:
                raise decode_error(reply["error"])
            return decode_document(reply["ok"])
        except RagError:
            raise
        except Exception:
            raise parse_failed(
                ParseReason.WORKER_FAILED, "the parsing worker sent an invalid reply"
            ) from None


class IsolatedParser:
    """A parser whose ``parse`` runs in the worker; its identity is that of the wrapped parser."""

    def __init__(
        self, inner: Parser, worker: ParserWorker, factory: str, args: Sequence[object]
    ) -> None:
        self._inner = inner
        self._worker = worker
        self._factory = factory
        self._args = tuple(args)

    @property
    def parser_id(self) -> str:
        return self._inner.parser_id

    @property
    def version(self) -> str:
        return self._inner.version

    @property
    def normalizer_version(self) -> str:
        return self._inner.normalizer_version

    def supports(self, mime: str | None, extension: str | None) -> bool:
        return self._inner.supports(mime, extension)

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        check_input_size(data, hints)
        return self._worker.call(self._factory, self._args, data, hints)
