# args passed to `python -m src`, e.g. make run ARGS='search "lora" --k 5'
ARGS ?= index

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

test:
	uv run pytest

clean:
	rm -rf .mypy_cache .pytest_cache
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug test clean lint lint-strict