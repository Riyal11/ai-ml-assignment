.PHONY: install lint format typecheck security test

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff format .

typecheck:
	uv run mypy src

security:
	uv run bandit -r src

test:
	uv run pytest
