"""``rag calibrate`` from indexing to ``rag ask``: a real index, mocked model servers."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag_engine.cli.main import main
from rag_engine.container import DefaultContainer
from rag_engine.testing import HashingEmbedder
from tests import calibration_support as cal

DIGEST = "d" * 64
DIMENSION = 64
ANSWER = "The protocol is used [E1]."

type Run = Callable[..., tuple[int, str]]


class Chat:
    def __init__(self) -> None:
        self.calls = 0

    def respond(self) -> httpx.Response:
        self.calls += 1
        done = {"done": True, "done_reason": "stop", "prompt_eval_count": 50, "eval_count": 9}
        return httpx.Response(200, json={"message": {"content": ANSWER}, **done})


@pytest.fixture
def lab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[Run, Chat, Path]:
    embedder = HashingEmbedder(DIMENSION)
    chat = Chat()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        if request.url.path == "/api/chat":
            return chat.respond()
        texts = json.loads(request.content)["input"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            vectors = pool.submit(asyncio.run, embedder.embed_documents(texts)).result()
        return httpx.Response(200, json={"embeddings": vectors.tolist()})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(base_url=kw["base_url"], transport=httpx.MockTransport(handler)),
    )
    root = tmp_path / "notes"
    root.mkdir()
    for name, text in cal.documents().items():
        (root / name).write_text(text)
    config = tmp_path / "rag.toml"
    config.write_text(
        '[parsing]\nisolate = []\n[retrieval]\nmode = "dense"\n[embedding]\nenabled = true\n'
        f'provider = "ollama"\nmodel = "test"\nrevision = "{DIGEST}"\ndimension = {DIMENSION}\n'
        'base_url = "http://test"\n[llm]\nenabled = true\nprovider = "ollama"\n'
        'model = "chat-test"\ncontext_window = 4096\nmax_new_tokens = 256\nbase_url = "http://test"\n'
    )
    db = tmp_path / "rag.db"
    container = DefaultContainer()
    prefix = ["--config", str(config), "--db", str(db)]

    def run(*args: str) -> tuple[int, str]:
        capsys.readouterr()
        code = main([*prefix, *args], container=container)
        return code, capsys.readouterr().out

    assert run("source", "add", "notes", str(root))[0] == 0
    assert run("ingest", "notes", "--json")[0] == 0
    return run, chat, db


def ask_json(run: Run, question: str, *extra: str) -> tuple[int, dict[str, object]]:
    code, out = run("ask", question, "--json", *extra)
    return code, json.loads(out)


def gate_of(body: dict[str, object]) -> dict[str, object]:
    diagnostics = body["diagnostics"]
    assert isinstance(diagnostics, dict)
    gate = diagnostics["gate"]
    assert isinstance(gate, dict)
    return gate


def warnings_of(body: dict[str, object]) -> list[str]:
    diagnostics = body["diagnostics"]
    assert isinstance(diagnostics, dict)
    return list(diagnostics["warnings"])


def test_the_whole_story_from_ingest_to_a_refused_question(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, chat, db = lab
    weak = cal.unanswerable(0)
    strong = cal.answerable(3)

    # 1. before any calibration the gate is open, and says so everywhere
    code, body = ask_json(run, weak)
    assert code == 0 and body["status"] == "answered"  # the model was asked and "answered"
    assert gate_of(body)["calibrated"] is False and gate_of(body)["reasons"] == ["uncalibrated"]
    assert any("not calibrated" in w and "rag calibrate" in w for w in warnings_of(body))
    _, doctor = run("doctor", "--integrity", "--json")
    assert any("not calibrated" in w for w in json.loads(doctor)["warnings"])
    _, shown = run("calibrate", "show", "--json")
    assert json.loads(shown)["state"] == "uncalibrated"

    # 2. calibrate on separate question files; the holdout is read once and reported separately
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    held = cal.write_questions(tmp_path / "held.jsonl", range(24, 48), offset=100)
    code, out = run("calibrate", "run", str(fit), "--holdout", str(held), "--json")
    result = json.loads(out)
    assert code == 0 and result["written"] is True and result["signal"] == "dense"
    assert result["holdout"]["correct_refusal"] == 1.0  # the corpus is separable on purpose
    assert result["estimate"]["false_refusal"] == 0.0 and result["meets_target"] is True
    assert cal.unanswerable(0) not in out  # no question text in the output

    # 3. the calibration is stored, applies, and is shown
    _, shown = run("calibrate", "show", "--json")
    info = json.loads(shown)
    assert info["state"] == "calibrated" and info["problems"] == []
    with closing(sqlite3.connect(db)) as conn:
        stored = json.loads(conn.execute("SELECT calibration_json FROM index_state").fetchone()[0])
    assert stored["sufficiency"]["algorithm"] == 2 and stored["sufficiency"]["binding"]
    assert weak not in json.dumps(stored)  # the record holds no question text

    # 4. now the weak question is refused before the model is called; the strong one is answered
    calls = chat.calls
    code, refused = ask_json(run, weak)
    assert code == 10 and refused["status"] == "insufficient_evidence"
    assert refused["reasons"] == ["gate:dense_below_threshold"]
    assert gate_of(refused)["calibrated"] is True and chat.calls == calls
    code, answered = ask_json(run, strong)
    assert code == 0 and answered["status"] == "answered" and chat.calls == calls + 1
    assert not any("not calibrated" in w for w in warnings_of(answered))

    # 5. doctor is quiet about the calibration
    _, doctor = run("doctor", "--integrity", "--json")
    assert not any("calibrat" in w for w in json.loads(doctor)["warnings"])


def test_a_different_retrieval_setup_is_flagged_instead_of_silently_applied(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, _ = lab
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    assert run("calibrate", "run", str(fit), "--json")[0] == 0
    # same setup: applies; another depth or mode: the gate lets it through and says why
    _, same = ask_json(run, cal.unanswerable(1))
    assert gate_of(same)["calibrated"] is True
    _, deeper = ask_json(run, cal.unanswerable(1), "--top-k", "3")
    assert gate_of(deeper)["reasons"] == ["calibration_stale"]
    assert any("another retrieval setup" in w for w in warnings_of(deeper))
    _, lexical = ask_json(run, cal.unanswerable(1), "--mode", "hybrid")
    assert gate_of(lexical)["reasons"] == ["calibration_stale"]


def test_a_calibration_below_the_targets_is_written_and_flagged(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, _ = lab
    # contradictory labels the corpus cannot separate: answerable questions that no document has
    rows = []
    for i in range(25):
        rows.append({"question": cal.unanswerable(i), "answerable": True})
        rows.append({"question": cal.unanswerable(i + 50), "answerable": False})
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    code, out = run("calibrate", "run", str(bad), "--json")
    result = json.loads(out)
    assert code == 0 and result["written"] and result["meets_target"] is False
    assert any("below its targets" in w for w in result["warnings"])
    _, body = ask_json(run, cal.answerable(2))
    assert any("below the targets" in w or "targets" in w for w in warnings_of(body))
    _, doctor = run("doctor", "--integrity", "--json")
    assert any("below its targets" in w for w in json.loads(doctor)["warnings"])


def test_require_target_writes_nothing_and_exits_12(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, _ = lab
    rows = [{"question": cal.unanswerable(i), "answerable": i % 2 == 0} for i in range(48)]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    code, out = run("calibrate", "run", str(bad), "--require-target", "--json")
    assert code == 12 and json.loads(out)["written"] is False
    assert json.loads(run("calibrate", "show", "--json")[1])["state"] == "uncalibrated"


def test_dry_run_writes_nothing_and_clear_removes_the_record(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, _ = lab
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    code, out = run("calibrate", "run", str(fit), "--dry-run", "--json")
    assert code == 0 and json.loads(out)["written"] is False
    assert json.loads(run("calibrate", "show", "--json")[1])["state"] == "uncalibrated"
    assert run("calibrate", "run", str(fit), "--json")[0] == 0
    assert json.loads(run("calibrate", "clear", "--json")[1]) == {"cleared": True}
    assert json.loads(run("calibrate", "clear", "--json")[1]) == {"cleared": False}
    assert json.loads(run("calibrate", "show", "--json")[1])["state"] == "uncalibrated"


def test_overlapping_holdout_and_too_little_data_are_refused(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, _ = lab
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    same = cal.write_questions(tmp_path / "same.jsonl", range(12, 36))  # shares questions 12..23
    code, out = run("calibrate", "run", str(fit), "--holdout", str(same), "--json")
    assert code == 2 and "holdout" in json.loads(out)["error"]["message"]
    small = cal.write_questions(tmp_path / "small.jsonl", range(0, 5))
    code, out = run("calibrate", "run", str(small), "--json")
    assert code == 2 and "at least 20" in json.loads(out)["error"]["message"]
    assert json.loads(run("calibrate", "show", "--json")[1])["state"] == "uncalibrated"


def test_a_changed_index_invalidates_the_estimate_not_the_gate(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, db = lab
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    assert run("calibrate", "run", str(fit), "--json")[0] == 0
    # shrink the recorded size to simulate a much larger index since the calibration
    with closing(sqlite3.connect(db)) as conn:
        record = json.loads(conn.execute("SELECT calibration_json FROM index_state").fetchone()[0])
        record["sufficiency"]["details"]["documents"] = 4
        conn.execute("UPDATE index_state SET calibration_json = ?", (json.dumps(record),))
        conn.commit()
    _, doctor = run("doctor", "--integrity", "--json")
    assert any("changed a lot" in w for w in json.loads(doctor)["warnings"])
    info = json.loads(run("calibrate", "show", "--json")[1])
    assert info["state"] == "stale"


def _stored(db: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(db)) as conn:
        value = json.loads(conn.execute("SELECT calibration_json FROM index_state").fetchone()[0])
    assert isinstance(value, dict)
    return value


def test_saving_is_atomic_and_refuses_an_index_that_moved_on(
    lab: tuple[Run, Chat, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, db = lab
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    assert run("calibrate", "run", str(fit), "--json")[0] == 0
    before = _stored(db)

    from rag_engine.domain import ErrorCode, RagError
    from rag_engine.evaluation import gate_calibration as gc
    from rag_engine.retrieval.sufficiency import thresholds_from_metadata
    from rag_engine.storage.sqlite import open_store
    from rag_engine.system import SystemClock

    sufficiency = before["sufficiency"]
    assert isinstance(sufficiency, dict)
    thresholds = thresholds_from_metadata(before)
    assert thresholds is not None
    result = gc.CalibrationResult(
        thresholds, "dense", gc.Rates(1, 1, 0, 1), gc.Rates(1, 1, 0, 1), None, True, False, ()
    )
    model = str(sufficiency["model_id"])
    with open_store(db, SystemClock()) as store:
        generation = store.reader().get_index_metadata().index_generation  # type: ignore[union-attr]

        # a failure in the middle of the transaction leaves the old record whole
        def boom(_: object) -> dict[str, object]:
            raise RuntimeError("disk full")

        monkeypatch.setattr(gc, "thresholds_to_metadata", boom)
        with pytest.raises(RuntimeError):
            gc.save_calibration(store, result, model_id=model, index_generation=generation)
        monkeypatch.undo()
        assert _stored(db) == before

        # an index that was published again, or switched, since the measurement is refused
        with pytest.raises(RagError) as moved:
            gc.save_calibration(store, result, model_id=model, index_generation=generation + 1)
        assert moved.value.code is ErrorCode.E_STORE_LOCKED and moved.value.retryable
        with pytest.raises(RagError):
            gc.save_calibration(store, result, model_id="another", index_generation=generation)
        assert _stored(db) == before


def test_other_calibration_keys_survive_a_new_calibration(
    lab: tuple[Run, Chat, Path], tmp_path: Path
) -> None:
    run, _, db = lab
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("UPDATE index_state SET calibration_json = ?", (json.dumps({"other": 1}),))
        conn.commit()
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    assert run("calibrate", "run", str(fit), "--json")[0] == 0
    stored = _stored(db)
    assert stored["other"] == 1 and "sufficiency" in stored
    run("calibrate", "clear", "--json")
    assert _stored(db) == {"other": 1}


def test_the_text_output_says_what_was_done(lab: tuple[Run, Chat, Path], tmp_path: Path) -> None:
    run, _, _ = lab
    fit = cal.write_questions(tmp_path / "fit.jsonl", range(0, 24))
    code, out = run("calibrate", "run", str(fit))
    assert code == 0
    assert "cross-validated estimate" in out and "optimistic" in out
    assert "targets: correct refusals at least 90%" in out and "calibration written" in out
    _, shown = run("calibrate", "show")
    assert "gate calibration: calibrated" in shown and "signal dense" in shown
    _, dry = run("calibrate", "run", str(fit), "--dry-run")
    assert "nothing written" in dry
    _, cleared = run("calibrate", "clear")
    assert "calibration removed" in cleared
    assert "there was no calibration" in run("calibrate", "clear")[1]
    assert "run: rag calibrate run" in run("calibrate", "show")[1]
