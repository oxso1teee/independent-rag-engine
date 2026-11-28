# ruff: noqa: E501  (the page below is HTML, CSS and JavaScript in one string)
"""A self-contained kit for a person to review evaluation questions, and the way back.

``build_kit`` turns one or more datasets into **one HTML file** that works offline: for every
question the source documents, the expected answer, the evidence quoted with its context, the
closest passages for questions that have no answer, and controls to accept, edit or reject it. The
person's decisions are exported as a CSV sheet, and ``apply_kit_sheet`` records them, under the
reviewer's own name, into new versions of the datasets. Nothing here ever marks a question as
reviewed: building the kit changes no dataset, and a row without a decision stays a draft.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rag_engine.domain import ErrorCode, Language, RagError
from rag_engine.evaluation.dataset import Dataset, Question, parse_dataset, validate_dataset
from rag_engine.evaluation.review import COLUMNS, apply_review
from rag_engine.lexical.analysis import analyze

SHEET_COLUMNS = ("dataset", *COLUMNS)
_CONTEXT_CHARS = 400
_NEIGHBOURS = 3


@dataclass(frozen=True, slots=True)
class SourceDataset:
    """A dataset as it is on disk, with its corpus."""

    name: str  # the file stem, used to route decisions back
    raw: Mapping[str, Any]
    dataset: Dataset
    corpus: Mapping[str, str]


def load_source(path: Path, root: Path) -> SourceDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = parse_dataset(raw)
    corpus_dir = root / dataset.corpus
    corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}
    report = validate_dataset(dataset, corpus)
    if not report.ok:
        raise RagError(
            ErrorCode.E_INTEGRITY,
            f"{path.name} does not validate",
            context={"errors": list(report.errors)[:5]},
        )
    return SourceDataset(path.stem, raw, dataset, corpus)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _terms(text: str) -> set[str]:
    return {t for t in analyze(text, Language.MIXED).split() if len(t) >= 3 or t.isdigit()}


def _around(text: str, anchor: str) -> dict[str, str]:
    start = text.casefold().find(anchor.casefold())
    if start < 0:
        return {"before": "", "anchor": anchor, "after": ""}
    low = max(0, start - _CONTEXT_CHARS)
    high = min(len(text), start + len(anchor) + _CONTEXT_CHARS)
    return {
        "before": ("…" if low else "") + text[low:start],
        "anchor": text[start : start + len(anchor)],
        "after": text[start + len(anchor) : high] + ("…" if high < len(text) else ""),
    }


def _closest(question: Question, corpus: Mapping[str, str]) -> list[dict[str, str]]:
    """The corpus sentences that share most terms with the question (no model: stems only)."""
    wanted = _terms(question.question)
    scored = []
    for doc, text in corpus.items():
        for sentence in _sentences(text):
            overlap = len(wanted & _terms(sentence))
            if overlap:
                scored.append((overlap, doc, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"doc": d, "text": s} for _, d, s in scored[:_NEIGHBOURS]]


def question_card(source: SourceDataset, question: Question) -> dict[str, Any]:
    evidence = [
        {"doc": ref.doc, **_around(source.corpus.get(ref.doc, ""), ref.anchor)}
        for ref in question.gold
    ]
    card: dict[str, Any] = {
        "key": f"{source.name}:{question.id}",
        "dataset": source.name,
        "id": question.id,
        "split": question.split,
        "type": question.type,
        "language": question.language,
        "question": question.question,
        "answerable": question.answerable,
        "reference_answer": question.reference_answer or "",
        "reference_facts": list(question.reference_facts),
        "evidence": evidence,
        "gold": "\n".join(f"{ref.doc} :: {ref.anchor}" for ref in question.gold),
        "absence_terms": list(question.absence_terms),
        "closest": [] if question.answerable else _closest(question, source.corpus),
        "status": question.review.status,
        "provenance": f"{question.provenance.created_by}: {question.provenance.method}",
    }
    return card


def build_kit(
    sources: Sequence[SourceDataset], *, title: str = "Evaluation question review"
) -> str:
    cards = [question_card(s, q) for s in sources for q in s.dataset.questions]
    documents = {f"{s.name}/{doc}": text for s in sources for doc, text in s.corpus.items()}
    payload = {
        "title": title,
        "columns": list(SHEET_COLUMNS),
        "cards": cards,
        "documents": documents,
        "datasets": {s.name: s.dataset.identifier for s in sources},
    }
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


# --------------------------------------------------------------------------- applying a sheet


@dataclass(frozen=True, slots=True)
class AppliedDataset:
    name: str
    version: int
    path: Path
    accepted: int
    edited: int
    rejected: int
    undecided: int


def apply_kit_sheet(
    sources: Sequence[SourceDataset], sheet: str, *, reviewer: str, outdir: Path, on: date
) -> list[AppliedDataset]:
    """Record the sheet's decisions into new dataset versions in ``outdir``; never overwrite."""
    rows = list(csv.DictReader(io.StringIO(sheet)))
    if rows and "dataset" not in rows[0]:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "review sheet: missing the dataset column")
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    known = {s.name for s in sources}
    unknown = set(by_dataset) - known
    if unknown:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID, f"review sheet: unknown dataset {sorted(unknown)}"
        )
    applied: list[AppliedDataset] = []
    outdir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        decided = by_dataset.get(source.name, [])
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(COLUMNS), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(decided)
        updated = apply_review(source.raw, buffer.getvalue(), reviewer=reviewer, on=on)
        version = int(source.raw["version"]) + 1
        updated["version"] = version
        if not validate_dataset(parse_dataset(updated), source.corpus).ok:
            raise RagError(ErrorCode.E_INTEGRITY, f"{source.name}: the reviewed dataset is invalid")
        target = outdir / f"{source.name.rsplit('_v', 1)[0]}_v{version}.json"
        if target.exists():
            raise RagError(ErrorCode.E_INTEGRITY, f"{target.name} already exists; not overwriting")
        target.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        statuses = [q["review"]["status"] for q in updated["questions"]]
        applied.append(
            AppliedDataset(
                source.name, version, target,
                statuses.count("accepted"), statuses.count("edited"), statuses.count("rejected"),
                statuses.count("unreviewed"),
            )
        )  # fmt: skip
    return applied


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1c1e21;--muted:#5f6670;--line:#d7dbe0;--card:#f7f8fa;--ok:#1a7f37;--bad:#b42318;--edit:#9a6700;--mark:#fff3a3}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e8eb;--muted:#9aa3ad;--line:#33383f;--card:#1e2126;--ok:#4ac26b;--bad:#ff7b72;--edit:#e3b341;--mark:#5c4b00}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 6px}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input,textarea,button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.primary{background:var(--ok);color:#fff;border-color:var(--ok)}
main{max-width:980px;margin:0 auto;padding:12px 16px 80px}
.card{border:1px solid var(--line);background:var(--card);border-radius:8px;margin:14px 0;padding:12px 14px}
.card.done{border-left:5px solid var(--ok)}.card.rej{border-left:5px solid var(--bad)}.card.edt{border-left:5px solid var(--edit)}
.meta{color:var(--muted);font-size:13px}.badge{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 8px;margin-right:6px;font-size:12px}
.q{font-size:17px;font-weight:600;margin:6px 0}.lbl{font-weight:600;margin-top:8px;font-size:13px;color:var(--muted);text-transform:uppercase}
.ev{border-left:3px solid var(--line);padding:4px 10px;margin:4px 0;white-space:pre-wrap}mark{background:var(--mark);color:inherit}
details{margin:6px 0}summary{cursor:pointer;color:var(--muted)}pre{white-space:pre-wrap;margin:6px 0}
.decide{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;align-items:center}textarea{width:100%;min-height:48px}
.hint{color:var(--muted);font-size:13px}.warn{color:var(--bad);font-size:13px}
</style></head><body>
<header><h1>__TITLE__</h1>
<div class="bar">
 <label>Dataset <select id="f-dataset"></select></label>
 <label>Split <select id="f-split"><option value="">all</option><option>dev</option><option>test</option></select></label>
 <label>Type <select id="f-type"></select></label>
 <label>Language <select id="f-lang"><option value="">all</option><option>en</option><option>ru</option></select></label>
 <label>Show <select id="f-state"><option value="">all</option><option value="todo">not decided</option><option value="done">decided</option></select></label>
 <input id="f-text" placeholder="search the questions" size="22">
 <span id="progress" class="hint"></span>
 <button id="export" class="primary">Export decisions (CSV)</button>
</div></header>
<main>
<p class="hint">Decide each question: <b>accept</b> if the question and its labels are right as they are; <b>edit</b> if they are right after your change; <b>reject</b> (with a reason) if it should not be used. Leave it undecided if you are unsure: it stays a draft. Your choices are saved in this browser as you go; export them when you are done.</p>
<div id="cards"></div>
<details><summary>All source documents</summary><div id="docs"></div></details>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const KEY = 'review-kit:' + D.cards.length + ':' + D.cards[0].key + ':' + D.cards[D.cards.length-1].key;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { state = {}; }
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function st(key) { return state[key] || {decision:'', eq:'', ea:'', notes:''}; }
function progress() {
  const done = D.cards.filter(c => st(c.key).decision).length;
  $('progress').textContent = done + ' of ' + D.cards.length + ' decided';
}
function opts(sel, values, first) { sel.innerHTML = '<option value="">' + first + '</option>' + values.map(v => '<option>' + esc(v) + '</option>').join(''); }
opts($('f-dataset'), Object.keys(D.datasets), 'all');
opts($('f-type'), [...new Set(D.cards.map(c => c.type))], 'all');
function evidence(c) {
  if (!c.evidence.length) return '<div class="hint">No evidence: the documents must not answer this question.</div>';
  return c.evidence.map(e => '<div class="ev"><b>' + esc(e.doc) + '</b><br>' + esc(e.before) + '<mark>' + esc(e.anchor) + '</mark>' + esc(e.after) + '</div>').join('');
}
function absent(c) {
  if (c.answerable) return '';
  const near = c.closest.length ? c.closest.map(n => '<div class="ev"><b>' + esc(n.doc) + '</b>: ' + esc(n.text) + '</div>').join('') : '<div class="hint">Nothing in the corpus shares terms with the question.</div>';
  return '<div class="lbl">Terms that must not occur in the corpus</div><div>' + esc(c.absence_terms.join(', ')) + '</div>' +
         '<div class="lbl">Closest passages in the corpus (is any of them an answer?)</div>' + near;
}
function render() {
  const f = {dataset:$('f-dataset').value, split:$('f-split').value, type:$('f-type').value, lang:$('f-lang').value, state:$('f-state').value, text:$('f-text').value.toLowerCase()};
  const list = D.cards.filter(c => (!f.dataset || c.dataset === f.dataset) && (!f.split || c.split === f.split) && (!f.type || c.type === f.type) &&
    (!f.lang || c.language === f.lang) && (!f.state || (f.state === 'done') === !!st(c.key).decision) && (!f.text || c.question.toLowerCase().includes(f.text)));
  $('cards').innerHTML = list.map(c => {
    const s = st(c.key); const cls = s.decision === 'accept' ? 'done' : s.decision === 'edit' ? 'edt' : s.decision === 'reject' ? 'rej' : '';
    const radios = ['accept','edit','reject'].map(v => '<label><input type="radio" name="d-' + esc(c.key) + '" data-key="' + esc(c.key) + '" data-f="decision" value="' + v + '"' + (s.decision === v ? ' checked' : '') + '> ' + v + '</label>').join('');
    return '<div class="card ' + cls + '" id="c-' + esc(c.key) + '">' +
      '<div class="meta"><span class="badge">' + esc(D.datasets[c.dataset]) + '</span><span class="badge">' + esc(c.id) + '</span><span class="badge">' + esc(c.split) + '</span><span class="badge">' + esc(c.type) + '</span><span class="badge">' + esc(c.language) + '</span><span class="badge">' + (c.answerable ? 'has an answer' : 'NO answer expected') + '</span></div>' +
      '<div class="q">' + esc(c.question) + '</div>' +
      (c.answerable ? '<div class="lbl">Expected answer</div><div>' + esc(c.reference_answer) + '</div><div class="lbl">Facts the answer must contain</div><div>' + esc(c.reference_facts.join(' | ')) + '</div>' : '') +
      '<div class="lbl">Evidence in the documents</div>' + evidence(c) + absent(c) +
      '<div class="hint">Origin: ' + esc(c.provenance) + '</div>' +
      '<div class="decide">' + radios + '<button data-clear="' + esc(c.key) + '">clear</button></div>' +
      (s.decision === 'edit' ? '<div class="lbl">Corrected question (leave empty to keep)</div><textarea data-key="' + esc(c.key) + '" data-f="eq">' + esc(s.eq) + '</textarea><div class="lbl">Corrected expected answer (leave empty to keep)</div><textarea data-key="' + esc(c.key) + '" data-f="ea">' + esc(s.ea) + '</textarea>' : '') +
      '<div class="lbl">Notes' + (s.decision === 'reject' ? ' (required: why is it rejected?)' : '') + '</div><textarea data-key="' + esc(c.key) + '" data-f="notes">' + esc(s.notes) + '</textarea>' +
      '</div>';
  }).join('') || '<p class="hint">No question matches the filters.</p>';
  progress();
}
document.addEventListener('input', e => { const t = e.target; if (t.dataset && t.dataset.f && t.tagName === 'TEXTAREA') { (state[t.dataset.key] = st(t.dataset.key))[t.dataset.f] = t.value; save(); } });
document.addEventListener('change', e => { const t = e.target; if (t.dataset && t.dataset.f === 'decision') { (state[t.dataset.key] = st(t.dataset.key)).decision = t.value; save(); render(); } else if (t.id && t.id.startsWith('f-')) render(); });
document.addEventListener('input', e => { if (e.target.id === 'f-text') render(); });
document.addEventListener('click', e => { const k = e.target.dataset && e.target.dataset.clear; if (k) { delete state[k]; save(); render(); } });
$('docs').innerHTML = Object.entries(D.documents).map(([n, t]) => '<details><summary>' + esc(n) + '</summary><pre>' + esc(t) + '</pre></details>').join('');
function csvCell(v) { v = String(v == null ? '' : v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
$('export').addEventListener('click', () => {
  const problems = [];
  D.cards.forEach(c => { const s = st(c.key);
    if (s.decision === 'reject' && !s.notes.trim()) problems.push(c.key + ': a rejection needs a reason in the notes');
    if (s.decision === 'edit' && !s.eq.trim() && !s.ea.trim()) problems.push(c.key + ': an edit needs a corrected question or answer'); });
  if (problems.length && !confirm('These rows will be refused when the sheet is applied:\n\n' + problems.slice(0, 12).join('\n') + (problems.length > 12 ? '\n…' : '') + '\n\nExport anyway?')) return;
  const rows = [D.columns.join(',')];
  D.cards.forEach(c => { const s = st(c.key);
    const r = {dataset:c.dataset, id:c.id, split:c.split, type:c.type, language:c.language, question:c.question, answerable:c.answerable ? 'yes' : 'no', gold:c.gold,
      context:c.evidence.map(e => '[' + e.doc + '] ' + e.before + e.anchor + e.after).join('\n'), reference_answer:c.reference_answer, absence_terms:c.absence_terms.join(', '),
      status:c.status, decision:s.decision, edited_question:s.eq, edited_reference_answer:s.ea, notes:s.notes};
    rows.push(D.columns.map(k => csvCell(r[k])).join(',')); });
  const blob = new Blob([rows.join('\n') + '\n'], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'review_decisions.csv'; a.click();
});
render();
</script></body></html>
"""
