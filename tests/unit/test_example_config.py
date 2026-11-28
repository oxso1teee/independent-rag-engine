"""``rag.example.toml`` mentions every setting, so the example cannot drift from the code."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from rag_engine.config import Settings, load_settings

pytestmark = pytest.mark.unit

EXAMPLE = Path(__file__).resolve().parents[2] / "rag.example.toml"


def test_every_setting_is_documented_in_the_example() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    missing = []
    for section, field in Settings.model_fields.items():
        nested = getattr(field.annotation, "model_fields", None)
        if nested is None:
            if not re.search(rf"^{section}\s*=", text, re.M):
                missing.append(section)
            continue
        if not re.search(rf"^\[{section}\]", text, re.M):
            missing.append(f"[{section}]")
        missing += [f"{section}.{n}" for n in nested if not re.search(rf"\b{n}\b", text)]
    assert not missing, f"undocumented settings: {missing}"


def test_the_example_loads_and_keeps_the_safe_defaults() -> None:
    tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    settings = load_settings(EXAMPLE)
    assert settings.llm.keep_alive == "0" and settings.api.host == "127.0.0.1"
    assert settings.rerank.enabled is False and settings.llm.enabled is False
