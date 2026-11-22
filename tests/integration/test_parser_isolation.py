"""Parsers in a separate worker process: results, failures, limits and recovery."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from rag_engine.domain import ErrorCode, ParsedDocument, RagError
from rag_engine.parsing.isolation import IsolatedParser, ParserWorker
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseHints, ParseLimits, Parser
from tests import isolation_fakes
from tests.docx_builder import build_docx, encrypted_docx_stub
from tests.pdf_builder import build_pdf, encrypt_pdf

HINTS = ParseHints(mime="text/plain", limits=ParseLimits(timeout_s=30))
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def worker() -> Iterator[ParserWorker]:
    instance = ParserWorker(memory_mb=700, grace_s=0.3)
    yield instance
    instance.close()


def fake(name: str, worker: ParserWorker) -> IsolatedParser:
    inner: Parser = getattr(isolation_fakes, name)()
    return IsolatedParser(inner, worker, f"tests.isolation_fakes:{name}", [])


def failure(parser: Parser, data: bytes = b"text", hints: ParseHints = HINTS) -> RagError:
    with pytest.raises(RagError) as info:
        parser.parse(data, hints)
    return info.value


def pid_of(document: ParsedDocument) -> int:
    assert document.title is not None
    return int(document.title.removeprefix("pid:"))


def test_the_result_is_the_document_the_parser_made_in_the_worker(worker: ParserWorker) -> None:
    isolated = fake("Rich", worker)
    local = isolation_fakes.Rich().parse(b"", HINTS)
    assert isolated.parse(b"", HINTS) == local


def test_the_worker_is_long_lived_and_never_the_calling_process(worker: ParserWorker) -> None:
    echo = fake("Echo", worker)
    first, second = echo.parse(b"one", HINTS), echo.parse(b"two", HINTS)
    assert pid_of(first) == pid_of(second) != os.getpid()
    assert [b.text for b in (first.blocks[0], second.blocks[0])] == ["one", "two"]


def test_the_wrapper_has_the_identity_of_the_parser_it_wraps(worker: ParserWorker) -> None:
    echo = fake("Echo", worker)
    assert (echo.parser_id, echo.version, echo.normalizer_version) == ("echo", "1", "norm/1")
    assert echo.supports("text/plain", "txt")


def test_the_input_size_limit_is_checked_before_a_worker_is_started(worker: ParserWorker) -> None:
    hints = ParseHints(limits=ParseLimits(max_input_bytes=5))
    error = failure(fake("Echo", worker), b"more than five bytes", hints)
    assert error.context["reason"] == "too_large"
    assert worker._process is None


def test_a_parser_that_hangs_is_killed_and_the_next_document_still_works(
    worker: ParserWorker,
) -> None:
    hints = ParseHints(limits=ParseLimits(timeout_s=0.5))
    echo = fake("Echo", worker)
    before = pid_of(echo.parse(b"warm", HINTS))
    error = failure(fake("Sleeper", worker), hints=hints)
    assert error.code is ErrorCode.E_PARSE_FAILED and error.context["reason"] == "timeout"
    assert pid_of(echo.parse(b"after", HINTS)) != before


def test_a_crashing_parser_is_reported_and_the_worker_is_replaced(worker: ParserWorker) -> None:
    error = failure(fake("Crasher", worker))
    assert error.context["reason"] == "worker_failed" and error.context["exit_code"] == 7
    assert fake("Echo", worker).parse(b"alive", HINTS).blocks[0].text == "alive"


def test_a_parser_that_needs_too_much_memory_fails_with_the_memory_reason(
    worker: ParserWorker,
) -> None:
    error = failure(fake("Greedy", worker))
    assert error.code is ErrorCode.E_PARSE_FAILED and error.context["reason"] == "memory"
    assert fake("Echo", worker).parse(b"alive", HINTS).blocks[0].text == "alive"


def test_deep_nesting_and_bugs_are_reported_without_the_document_text(
    worker: ParserWorker,
) -> None:
    assert failure(fake("Deep", worker)).context["reason"] == "too_deep"
    buggy = failure(fake("Buggy", worker))
    assert buggy.context["reason"] == "corrupt" and buggy.context["error_type"] == "ValueError"
    assert "private text" not in repr(buggy.to_dict())


def test_a_parser_that_is_not_installed_in_the_worker_is_unsupported(
    worker: ParserWorker,
) -> None:
    missing = IsolatedParser(isolation_fakes.Echo(), worker, "no_such_module_anywhere:Parser", [])
    assert failure(missing).code is ErrorCode.E_UNSUPPORTED_FORMAT
    unknown = IsolatedParser(isolation_fakes.Echo(), worker, "tests.isolation_fakes:Nope", [])
    assert failure(unknown).context["reason"] == "corrupt"


def test_output_of_the_parser_cannot_corrupt_the_protocol(worker: ParserWorker) -> None:
    assert fake("Loud", worker).parse(b"fine", HINTS).blocks[0].text == "fine"
    assert fake("Loud", worker).parse(b"again", HINTS).blocks[0].text == "again"


def test_the_worker_does_not_inherit_secrets(
    worker: ParserWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LLM__API_KEY", "super-secret")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "super-secret")
    document = fake("Environment", worker).parse(b"", HINTS)
    names = set(document.normalized_text.split())
    assert not {"RAG_LLM__API_KEY", "MY_SERVICE_TOKEN"} & names
    assert "PATH" in names
    assert document.title != os.getcwd()  # not the caller's directory


def test_a_worker_is_replaced_after_a_number_of_documents() -> None:
    short_lived = ParserWorker(memory_mb=700, max_documents=2)
    try:
        echo = fake("Echo", short_lived)
        pids = [pid_of(echo.parse(b"x", HINTS)) for _ in range(5)]
    finally:
        short_lived.close()
    assert pids[0] == pids[1] != pids[2] == pids[3] != pids[4]


def test_a_closed_worker_starts_again_when_needed(worker: ParserWorker) -> None:
    echo = fake("Echo", worker)
    first = pid_of(echo.parse(b"x", HINTS))
    worker.close()
    assert pid_of(echo.parse(b"x", HINTS)) != first


def test_an_invalid_reply_is_a_worker_failure() -> None:
    replies: list[object] = [{"ok": {"text": 5}}, {"ok": None}, {"error": {"code": "x"}}, [], None]
    for reply in replies:
        with pytest.raises(RagError) as info:
            ParserWorker._result(reply)
        assert info.value.context["reason"] == "worker_failed"


# ---------------------------------------------------------------- the real parsers


@pytest.fixture
def parsers() -> Iterator[tuple[dict[str, Parser], dict[str, Parser]]]:
    shared = ParserWorker(memory_mb=900, grace_s=1)
    isolated = default_parsers(isolate=("pdf", "docx", "html"), worker=shared)
    yield (
        {p.parser_id: p for p in isolated},
        {p.parser_id: p for p in default_parsers()},
    )
    shared.close()


def test_only_the_named_formats_run_in_the_worker() -> None:
    parsers = default_parsers(isolate=("pdf", "docx", "html"))
    kinds = {p.parser_id: isinstance(p, IsolatedParser) for p in parsers}
    assert kinds == {"md": False, "html": True, "txt": False, "pdf": True, "docx": True}
    assert not any(isinstance(p, IsolatedParser) for p in default_parsers())


def test_real_parsers_give_the_same_documents_in_the_worker(
    parsers: tuple[dict[str, Parser], dict[str, Parser]],
) -> None:
    isolated, local = parsers
    samples = {
        "html": (b"<main><h1>T</h1><p>Body &amp; more.</p><ul><li>a</li></ul></main>", "text/html"),
        "pdf": (build_pdf([["Первая страница."], ["Page two."]], title="X"), "application/pdf"),
        "docx": (build_docx([("title", "Doc"), ("p", "Text."), ("table", [["a"]])]), DOCX),
    }
    for parser_id, (data, mime) in samples.items():
        hints = ParseHints(mime=mime, limits=ParseLimits(timeout_s=30))
        assert isolated[parser_id].parse(data, hints) == local[parser_id].parse(data, hints)


def test_real_parser_failures_keep_their_reasons_in_the_worker(
    parsers: tuple[dict[str, Parser], dict[str, Parser]],
) -> None:
    isolated, _ = parsers
    pdf = ParseHints(mime="application/pdf", limits=ParseLimits(timeout_s=30))
    locked = encrypt_pdf(build_pdf([["Secret text."]]), "pw")
    assert failure(isolated["pdf"], locked, pdf).context["reason"] == "encrypted"
    assert failure(isolated["pdf"], build_pdf([None]), pdf).context["reason"] == "no_text_layer"
    docx = ParseHints(mime=DOCX, limits=ParseLimits(timeout_s=30))
    assert failure(isolated["docx"], encrypted_docx_stub(), docx).context["reason"] == "encrypted"
    assert failure(isolated["docx"], b"PK\x03\x04junk", docx).context["reason"] == "corrupt"
