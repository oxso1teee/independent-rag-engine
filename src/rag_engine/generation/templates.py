"""Prompt rendering for ``answer.v1``.

The layout is fixed so that a run can be reproduced from the prompt hash: the evidence block, one
numbered passage after another, then the question. Evidence and question are data. Marker-like text
inside a passage is neutralised so a document cannot invent a citation, and the question is
flattened to one line so it cannot open a section of its own.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag_engine.context.prompt import PROMPT_ID, SYSTEM_PROMPT
from rag_engine.domain import Context

_MARKER_LIKE = re.compile(r"\[(E[1-9][0-9]*)\]")


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    template_id: str
    system_prompt: str
    prompt: str
    prompt_sha256: str


def _defuse(text: str) -> str:
    return _MARKER_LIKE.sub(r"(\1)", text)


def render_answer(question: str, context: Context) -> RenderedPrompt:
    if not context.evidence:
        raise ValueError("an answer prompt needs at least one piece of evidence")
    passages = [
        f"[{item.evidence_id}] {item.locator.source_key}\n{_defuse(item.text)}"
        for item in context.evidence
    ]
    flat = " ".join(question.split())
    prompt = "Evidence:\n" + "\n\n".join(passages) + f"\n\nQuestion: {_defuse(flat)}"
    digest = hashlib.sha256(f"{PROMPT_ID}\n{SYSTEM_PROMPT}\n{prompt}".encode()).hexdigest()
    return RenderedPrompt(PROMPT_ID, SYSTEM_PROMPT, prompt, digest)
