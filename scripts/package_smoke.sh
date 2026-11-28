#!/usr/bin/env bash
# Build the wheel and the source distribution, install the wheel into an empty environment with no
# extras, and check that the command works, that typing information ships and that no test, doc or
# evaluation data leaks into the wheel.
set -euo pipefail

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"
uv build --out-dir "$WORK/dist" >/dev/null
WHEEL="$(ls "$WORK"/dist/*.whl)"
SDIST="$(ls "$WORK"/dist/*.tar.gz)"

# listings go to files first: `grep -q` quits at the first match and would SIGPIPE a producer
# that is still writing, which `pipefail` reports as a failure (exit 141)
unzip -l "$WHEEL" > "$WORK/wheel.txt"
tar -tzf "$SDIST" > "$WORK/sdist.txt"
grep -q "rag_engine/py.typed" "$WORK/wheel.txt"
if grep -E " (tests|eval|docs|baselines)/" "$WORK/wheel.txt" >/dev/null; then
  echo "the wheel contains tests, docs or evaluation data"; exit 1
fi
grep -q "pyproject.toml" "$WORK/sdist.txt"
if grep -E "/(CLAUDE\.md|\.claude/|\.import_linter_cache/|eval_runs/|data/|\.env)" "$WORK/sdist.txt" >/dev/null; then
  echo "the source distribution contains local working files"; exit 1
fi

# A supported interpreter, whatever `python3` happens to be on the machine (the project needs 3.12 or 3.13).
uv venv --quiet --python 3.12 "$WORK/venv"
uv pip install --quiet --python "$WORK/venv/bin/python" "$WHEEL"
"$WORK/venv/bin/rag" --version
"$WORK/venv/bin/python" -c "import rag_engine; print('import ok', rag_engine.__version__)"
cd "$WORK"
"$WORK/venv/bin/rag" init --dir "$WORK/ws" >/dev/null
"$WORK/venv/bin/rag" --config "$WORK/ws/rag.toml" config check
"$WORK/venv/bin/rag" --config "$WORK/ws/rag.toml" status --json >/dev/null
# the API is an extra: without it `serve` says so instead of crashing
if "$WORK/venv/bin/rag" --config "$WORK/ws/rag.toml" serve >"$WORK/serve.log" 2>&1; then
  echo "serve worked without the api extra"; exit 1
fi
grep -q "api extra" "$WORK/serve.log"
echo "package smoke test passed"
