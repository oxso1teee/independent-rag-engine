.PHONY: sync lint format typecheck test test-all perf arch audit check

sync:
	uv sync --frozen

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -m unit

test-all:
	uv run pytest -m "not perf" --cov=rag_engine --cov-report=term-missing
	uv run coverage report --include="src/rag_engine/domain/*" --fail-under=90

perf:
	uv run pytest -m perf -s --no-cov

arch:
	uv run lint-imports

audit:
	uv lock --check
	uv run pip-audit

check: lint typecheck arch test-all
