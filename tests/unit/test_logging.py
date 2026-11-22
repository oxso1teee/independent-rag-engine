from __future__ import annotations

import io
import json
from datetime import timedelta

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.observability import (
    REDACTED,
    Timer,
    configure_logging,
    get_logger,
    is_sensitive_key,
    request_scope,
    stage_scope,
    text_digest,
)
from rag_engine.testing import FakeClock
from tests.factories import make_ids


def capture(level: str = "DEBUG", clock: FakeClock | None = None) -> io.StringIO:
    stream = io.StringIO()
    configure_logging(level, stream=stream, clock=clock)
    return stream


def events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_events_are_json_with_level_and_timestamp() -> None:
    clock = FakeClock()
    stream = capture(clock=clock)
    get_logger("test").info("ingest.started", documents=3)
    (event,) = events(stream)
    assert event["event"] == "ingest.started"
    assert event["documents"] == 3
    assert event["level"] == "info"
    assert event["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_request_id_is_present_on_every_event_of_a_request() -> None:
    stream = capture()
    request_id = make_ids().request_id()
    logger = get_logger("pipeline")
    with request_scope(request_id):
        logger.info("one")
        with stage_scope(logger, "retrieve"):
            logger.info("inside")
        logger.warning("two")
    logger.info("outside")
    recorded = events(stream)
    inside = [e for e in recorded if e["event"] != "outside"]
    assert len(inside) == 5
    assert all(e["request_id"] == str(request_id) for e in inside)
    assert "request_id" not in recorded[-1]


def test_stage_scope_correlates_stage_events() -> None:
    clock = FakeClock()
    stream = capture(clock=clock)
    logger = get_logger("pipeline")
    with request_scope(make_ids().request_id()), stage_scope(logger, "embed", clock=clock, n=2):
        clock.advance(timedelta(milliseconds=42))
    start, done = events(stream)
    assert (start["event"], start["stage"], start["n"]) == ("stage.start", "embed", 2)
    assert (done["event"], done["stage"], done["duration_ms"]) == ("stage.done", "embed", 42)
    assert start["request_id"] == done["request_id"]


def test_stage_failure_logs_error_code_without_the_message() -> None:
    stream = capture()
    logger = get_logger("pipeline")
    error = RagError(ErrorCode.E_PROVIDER_TIMEOUT, "details with private text")
    with pytest.raises(RagError), stage_scope(logger, "embed"):
        raise error
    failed = events(stream)[-1]
    assert failed["event"] == "stage.failed"
    assert failed["error_code"] == "E_PROVIDER_TIMEOUT"
    assert failed["error_category"] == "provider"
    assert failed["retryable"] is True
    assert "private" not in json.dumps(failed)


def test_stage_failure_with_unexpected_exception() -> None:
    stream = capture()
    with pytest.raises(ZeroDivisionError), stage_scope(get_logger("x"), "parse"):
        1 / 0  # noqa: B018
    failed = events(stream)[-1]
    assert failed["error_type"] == "ZeroDivisionError"


def test_level_filtering() -> None:
    stream = capture("WARNING")
    logger = get_logger("x")
    logger.info("hidden")
    logger.error("shown")
    assert [e["event"] for e in events(stream)] == ["shown"]


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("LOUD")


def test_sensitive_values_never_reach_the_log() -> None:
    stream = capture()
    document_text = "CONFIDENTIAL DOCUMENT BODY " * 3
    get_logger("x").info(
        "chunk.created",
        text=document_text,
        chunk_text=document_text,
        prompt=document_text,
        question="what is the salary?",
        api_key="sk-live-123",
        authorization="Bearer abc",
        token="tok",
        nested={"answer": "private", "safe": 1},
        chunks=1,
        token_count=7,
        prompt_tokens=9,
    )
    raw = stream.getvalue()
    for secret in ("CONFIDENTIAL", "salary", "sk-live-123", "Bearer", 'tok"', "private"):
        assert secret not in raw
    (event,) = events(stream)
    assert event["text"] == REDACTED
    assert event["nested"] == {"answer": REDACTED, "safe": 1}
    assert (event["chunks"], event["token_count"], event["prompt_tokens"]) == (1, 7, 9)


def test_long_strings_under_unexpected_keys_are_truncated() -> None:
    stream = capture()
    get_logger("x").info("odd", payload="x" * 500, short="ok")
    (event,) = events(stream)
    assert event["payload"] == "[TRUNCATED 500 chars]"
    assert event["short"] == "ok"


def test_deeply_nested_values_are_collapsed() -> None:
    stream = capture()
    get_logger("x").info("deep", data={"a": {"b": {"c": {"d": {"e": "leak"}}}}})
    assert "leak" not in stream.getvalue()


def test_sensitive_key_detection() -> None:
    assert all(
        is_sensitive_key(k)
        for k in ("text", "Prompt", "chunk_text", "system_prompt", "x_api_key", "DB_PASSWORD")
    )
    assert not any(
        is_sensitive_key(k)
        for k in ("token_count", "prompt_tokens", "chunks", "stage", "duration_ms")
    )


def test_text_digest_is_a_stable_short_fingerprint() -> None:
    assert text_digest("abc") == text_digest("abc") != text_digest("abd")
    assert len(text_digest("abc")) == 12
    assert "abc" not in text_digest("abc")


def test_timer_uses_the_injected_clock() -> None:
    clock = FakeClock()
    timer = Timer(clock)
    clock.advance(timedelta(milliseconds=1500))
    assert timer.elapsed_ms() == 1500


def test_console_rendering_is_supported() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_output=False, stream=stream)
    get_logger("x").info("hello", n=1)
    assert "hello" in stream.getvalue()


# ------------------------------------------------ security regression (exceptions, events)

PRIVATE = "CONFIDENTIAL DOCUMENT BODY"


def _single_event(stream: io.StringIO) -> dict[str, object]:
    recorded = events(stream)
    assert len(recorded) == 1
    return recorded[0]


def _assert_no_leak(stream: io.StringIO) -> None:
    assert "CONFIDENTIAL" not in stream.getvalue()


def _raise_private() -> None:
    raise ValueError(f"{PRIVATE} " + "x" * 400)


@pytest.mark.parametrize("style", ["exc_info_true", "exception_method", "instance", "tuple"])
def test_exceptions_never_leak_their_message(style: str) -> None:
    import sys

    stream = capture()
    logger = get_logger("t")
    try:
        _raise_private()
    except ValueError as caught:
        if style == "exc_info_true":
            logger.error("parse.failed", exc_info=True)
        elif style == "exception_method":
            logger.exception("parse.failed")
        elif style == "instance":
            logger.error("parse.failed", exc_info=caught)
        else:
            logger.error("parse.failed", exc_info=sys.exc_info())
    _assert_no_leak(stream)
    event = _single_event(stream)
    assert event["exception_type"] == "builtins.ValueError"
    assert "exc_info" not in event
    assert "exception" not in event
    frames = event["traceback"]
    assert isinstance(frames, list)
    assert any("_raise_private" in frame for frame in frames)
    assert all("/" not in frame for frame in frames)  # file names only, no directories


def test_rag_error_diagnostics_are_preserved() -> None:
    stream = capture()
    request_id = make_ids().request_id()
    error = RagError(ErrorCode.E_PROVIDER_TIMEOUT, f"{PRIVATE} timed out", context={"k": PRIVATE})
    with request_scope(request_id):
        try:
            raise error
        except RagError:
            get_logger("t").error("embed.failed", exc_info=True)
    _assert_no_leak(stream)
    event = _single_event(stream)
    assert event["error_code"] == "E_PROVIDER_TIMEOUT"
    assert event["error_category"] == "provider"
    assert event["retryable"] is True
    assert event["request_id"] == str(request_id)
    assert event["exception_type"].endswith("RagError")  # type: ignore[attr-defined]


def test_exception_chain_reports_types_only() -> None:
    stream = capture()
    try:
        try:
            _raise_private()
        except ValueError as inner:
            raise RagError(ErrorCode.E_PARSE_FAILED, "wrapped") from inner
    except RagError:
        get_logger("t").error("parse.failed", exc_info=True)
    _assert_no_leak(stream)
    assert _single_event(stream)["exception_chain"] == ["builtins.ValueError"]


def test_exc_info_without_an_active_exception_is_harmless() -> None:
    stream = capture()
    get_logger("t").error("odd.event", exc_info=True)
    event = _single_event(stream)
    assert "exception_type" not in event
    assert "exc_info" not in event


def test_manually_supplied_exception_text_is_redacted_and_capped() -> None:
    stream = capture()
    get_logger("t").error("x.failed", exception=f"Traceback ... {PRIVATE}", stack=PRIVATE * 20)
    _assert_no_leak(stream)


@pytest.mark.parametrize(
    "event",
    [
        f"parsed {PRIVATE}",
        f"{PRIVATE}",
        "Has Spaces",
        "UPPER.case",
        "x" * 200,
        "",
        "résumé.parsed",
    ],
)
def test_non_constant_event_names_are_replaced_by_a_digest(event: str) -> None:
    stream = capture()
    get_logger("t").info(event, chunks=2)
    _assert_no_leak(stream)
    recorded = _single_event(stream)
    assert recorded["event"] == "log.event_redacted"
    assert recorded["event_digest"] == text_digest(event)
    assert recorded["event_chars"] == len(event)
    assert recorded["chunks"] == 2


@pytest.mark.parametrize(
    "event", ["stage.start", "ingest.document_parsed", "a", "retry:2", "x-y_z.1"]
)
def test_constant_event_names_pass_unchanged(event: str) -> None:
    stream = capture()
    get_logger("t").info(event)
    assert _single_event(stream)["event"] == event


def test_non_string_event_is_handled() -> None:
    stream = capture()
    get_logger("t").info({"text": PRIVATE})  # type: ignore[arg-type]
    _assert_no_leak(stream)
    assert _single_event(stream)["event"] == "log.event_redacted"


def test_stage_failure_keeps_request_stage_and_code_without_content() -> None:
    stream = capture()
    logger = get_logger("pipeline")
    request_id = make_ids().request_id()
    with (
        pytest.raises(ValueError, match="CONFIDENTIAL"),
        request_scope(request_id),
        stage_scope(logger, "parse"),
    ):
        _raise_private()
    _assert_no_leak(stream)
    failed = events(stream)[-1]
    assert failed["event"] == "stage.failed"
    assert failed["request_id"] == str(request_id)
    assert failed["stage"] == "parse"
    assert failed["error_type"] == "ValueError"


def test_console_renderer_also_hides_exception_content() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_output=False, stream=stream)
    try:
        _raise_private()
    except ValueError:
        get_logger("t").exception("parse.failed")
    assert "CONFIDENTIAL" not in stream.getvalue()
    assert "ValueError" in stream.getvalue()
