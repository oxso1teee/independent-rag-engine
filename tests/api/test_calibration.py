"""The API and the CLI agree about the gate: one calibration, one decision, the same warnings."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.cli.main import main
from tests import calibration_support as cal
from tests.api.conftest import Api

pytestmark = pytest.mark.api


@pytest.fixture
def indexed(api: Api) -> Api:
    notes = api.root / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    for name, text in cal.documents().items():
        (notes / name).write_text(text)
    api.seed()
    return api


def test_an_uncalibrated_gate_is_reported_in_the_answer(indexed: Api) -> None:
    indexed.chat.replies.append("The protocol is used [E1].")
    body = indexed.client.post("/v1/answers", json={"question": cal.unanswerable(1)}).json()
    gate = body["diagnostics"]["gate"]
    assert gate["calibrated"] is False and gate["reasons"] == ["uncalibrated"]
    assert any("rag calibrate" in w for w in body["diagnostics"]["warnings"])


def test_a_calibration_made_with_the_cli_refuses_weak_questions_over_http(
    indexed: Api, tmp_path: Path
) -> None:
    questions = cal.write_questions(tmp_path / "q.jsonl", range(0, 24))
    assert main([*indexed.prefix, "calibrate", "run", str(questions), "--json"],
                container=indexed.container) == 0  # fmt: skip
    calls = len(indexed.chat.bodies)
    weak = indexed.client.post("/v1/answers", json={"question": cal.unanswerable(2)}).json()
    assert weak["status"] == "insufficient_evidence"
    assert weak["reasons"] == ["gate:dense_below_threshold"]
    assert weak["diagnostics"]["gate"]["calibrated"] is True
    assert len(indexed.chat.bodies) == calls  # the model was never called
    indexed.chat.replies.append("The protocol is used [E1].")
    strong = indexed.client.post("/v1/answers", json={"question": cal.answerable(5)}).json()
    assert strong["status"] == "answered"
    assert not any("not calibrated" in w for w in strong["diagnostics"]["warnings"])


def test_a_request_with_another_setup_is_let_through_with_a_warning(
    indexed: Api, tmp_path: Path
) -> None:
    questions = cal.write_questions(tmp_path / "q.jsonl", range(0, 24))
    assert main([*indexed.prefix, "calibrate", "run", str(questions), "--json"],
                container=indexed.container) == 0  # fmt: skip
    indexed.chat.replies.append("The protocol is used [E1].")
    body = indexed.client.post(
        "/v1/answers", json={"question": cal.unanswerable(2), "top_k": 3}
    ).json()
    assert body["diagnostics"]["gate"]["reasons"] == ["calibration_stale"]
    assert any("another retrieval setup" in w for w in body["diagnostics"]["warnings"])
