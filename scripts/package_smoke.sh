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

unzip -l "$WHEEL" | grep -q "rag_engine/py.typed"
if unzip -l "$WHEEL" | grep -E " (tests|eval|docs|baselines)/" >/dev/null; then
  echo "the wheel contains tests, docs or evaluation data"; exit 1
fi
tar -tzf "$SDIST" | grep -q "pyproject.toml"

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet "$WHEEL"
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
